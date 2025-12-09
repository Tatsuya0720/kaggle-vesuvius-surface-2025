import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------
# 0. Positional Encoding
# -----------------------------------------------------------
class PositionalEncodingXYZ(nn.Module):
    def __init__(self, num_freqs):
        super().__init__()
        self.num_freqs = num_freqs
        # 2^0, 2^1, ..., 2^(L-1) の周波数バンドを作成
        self.register_buffer("freq_bands", 2.0 ** torch.arange(num_freqs) * torch.pi)

    def forward(self, x):
        # x: (..., 3)
        # output: (..., 3 * 2 * num_freqs)

        # 座標を周波数倍する
        # (..., 3, 1) * (L,) -> (..., 3, L)
        raw = x.unsqueeze(-1) * self.freq_bands

        # sin, cos を適用
        sin_val = torch.sin(raw)
        cos_val = torch.cos(raw)

        # 結合: (..., 3, 2*L)
        # sin(x0), cos(x0), sin(x1)... の順で並べるイメージ
        enc = torch.cat([sin_val, cos_val], dim=-1)

        # 平坦化: (..., 6*L)
        return enc.reshape(*x.shape[:-1], -1)


# -----------------------------------------------------------
# 1. SineLayer (SIRENの構成要素)
# -----------------------------------------------------------
class SineLayer(nn.Module):
    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.linear.in_features, 1 / self.linear.in_features
                )
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.linear.in_features) / self.omega_0,
                    np.sqrt(6 / self.linear.in_features) / self.omega_0,
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class ResBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += residual  # Skip Connection
        out = self.relu(out)
        return out


# -----------------------------------------------------------
# 2. Encoder: 強化版 (座標変換 + ResNet)
# -----------------------------------------------------------
class CTEncoder3D(nn.Module):
    def __init__(self, input_channels=1, feature_dim=64):
        super().__init__()

        # 1. 初期特徴抽出
        self.conv_in = nn.Conv3d(input_channels, 16, kernel_size=3, padding=1)
        self.norm_in = nn.InstanceNorm3d(16)

        # 2. ダウンサンプリング (320 -> 160)
        self.down1 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.InstanceNorm3d(32)

        # 3. ResBlocks (受容野を広げる)
        # ここで層を重ねることで「局所的なノイズ」ではなく「形状」を認識させる
        self.res_blocks1 = nn.Sequential(ResBlock3D(32), ResBlock3D(32))

        # 4. ダウンサンプリング (160 -> 80)
        self.down2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm3d(64)

        # 5. ResBlocks (さらに受容野を広げる)
        self.res_blocks2 = nn.Sequential(
            ResBlock3D(64), ResBlock3D(64), ResBlock3D(64)  # 深くする
        )

        # 6. 出力層 (Feature Dimに合わせる)
        self.conv_out = nn.Conv3d(64, feature_dim, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (B, 1, D, H, W) -> (B, 1, Z, Y, X)

        # --- ★重要: 座標系の整合 ---
        # データセットが (Z, Y, X) で grid_sample が (X, Y, Z) を期待する場合
        # 軸を入れ替えて (B, 1, W, H, D) つまり (X, Y, Z) にする
        # ※ もし Dataset 側で既に transpose している場合は不要ですが、
        #    していない場合はここでやるのが確実です。
        # x = x.permute(0, 1, 4, 3, 2)  # (B, C, Z, Y, X) -> (B, C, X, Y, Z)

        x = F.relu(self.norm_in(self.conv_in(x)))

        x = F.relu(self.norm1(self.down1(x)))
        x = self.res_blocks1(x)

        x = F.relu(self.norm2(self.down2(x)))
        x = self.res_blocks2(x)

        x = self.conv_out(x)

        return x


# -----------------------------------------------------------
# 3. Decoder: 特徴量グリッドから値を拾ってSDFを予測 (SIREN + PosEnc)
# -----------------------------------------------------------
class GeneralizableDeepSDF(nn.Module):
    def __init__(self, feature_dim=64, hidden_dim=64, num_layers=3, xyz_pos_enc_dim=3):
        super().__init__()
        self.encoder = CTEncoder3D(feature_dim=feature_dim)
        self.xyz_pos_enc_dim = xyz_pos_enc_dim

        # Positional Encodingの準備
        if xyz_pos_enc_dim > 0:
            self.pos_enc = PositionalEncodingXYZ(xyz_pos_enc_dim)
            # 入力次元 = 特徴量次元 + (3次元 * 2(sin/cos) * 周波数数)
            input_dim = feature_dim + (6 * xyz_pos_enc_dim)
        else:
            self.pos_enc = None
            input_dim = feature_dim + 3

        # SIREN Decoder
        self.decoder_net = nn.ModuleList()

        # 最初の層
        self.decoder_net.append(
            SineLayer(input_dim, hidden_dim, is_first=True, omega_0=30.0)
        )

        for _ in range(num_layers - 1):
            self.decoder_net.append(
                SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=30.0)
            )

        self.final_layer = nn.Linear(hidden_dim, 1)

        # 最終層の初期化
        with torch.no_grad():
            self.final_layer.weight.uniform_(
                -np.sqrt(6 / hidden_dim) / 30, np.sqrt(6 / hidden_dim) / 30
            )

    def forward(self, ct_volume, query_xyz):
        """
        ct_volume: (B, 1, D, H, W) -> ボクセルデータ
        query_xyz: (B, N_points, 3) -> 推論したい座標 [-1, 1]
        """

        # 1. CTから特徴量グリッドを作成
        # Input: (B, 1, 320, 320, 320) -> Output: (B, 64, 80, 80, 80)
        feature_grid = self.encoder(ct_volume)

        # 2. クエリ座標に対応する特徴量をグリッドから取得 (grid_sample)
        # (B, 1, 1, N_points, 3)
        sample_coords = query_xyz.view(query_xyz.shape[0], 1, 1, -1, 3)

        features = F.grid_sample(
            feature_grid,
            sample_coords,
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        )

        # features: (B, F, 1, 1, N_points) -> (B, N_points, F)
        features_query = features.squeeze(2).squeeze(2).transpose(1, 2)

        # 3. 座標のエンコーディング
        if self.xyz_pos_enc_dim > 0:
            encoded_xyz = self.pos_enc(query_xyz)  # (B, N, 6 * L)
            decoder_input = torch.cat([encoded_xyz, features_query], dim=-1)
        else:
            decoder_input = torch.cat([query_xyz, features_query], dim=-1)

        # 4. SDF予測
        x = decoder_input
        for layer in self.decoder_net:
            x = layer(x)
        output = self.final_layer(x)

        return output
