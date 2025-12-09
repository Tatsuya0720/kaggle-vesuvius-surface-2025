import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------
# 1. Positional Encoding Class (User Defined)
# ---------------------------------------------------------
class PositionalEncodingXYZ(nn.Module):
    def __init__(self, hidden_dim):
        super(PositionalEncodingXYZ, self).__init__()
        self.hidden_dim = hidden_dim
        # 2^n * PI の周波数を作成
        self.register_buffer(
            "n2pi", torch.pow(2, torch.arange(int(hidden_dim))).float() * torch.pi
        )

    def forward(self, x):
        # x: (batch_size, 3) または (batch_size, num_points, 3)
        # 入力が2次元(N, 3)か3次元(B, N, 3)かに関わらず対応できるように次元を処理

        # 各軸を取り出す
        pos_x = x[..., 0]
        pos_y = x[..., 1]
        pos_z = x[..., 2]

        # 周波数成分を掛け合わせるために次元拡張
        # (..., 1) * (hidden_dim) -> (..., hidden_dim)
        pos_x = pos_x.unsqueeze(-1) * self.n2pi
        pos_y = pos_y.unsqueeze(-1) * self.n2pi
        pos_z = pos_z.unsqueeze(-1) * self.n2pi

        # sin, cos を適用して結合
        # output dim: hidden_dim * 2 (sin/cos)
        pos_x = torch.cat([torch.sin(pos_x), torch.cos(pos_x)], dim=-1)
        pos_y = torch.cat([torch.sin(pos_y), torch.cos(pos_y)], dim=-1)
        pos_z = torch.cat([torch.sin(pos_z), torch.cos(pos_z)], dim=-1)

        # 全ての軸を結合: output dim = 6 * hidden_dim
        return torch.cat([pos_x, pos_y, pos_z], dim=-1)


# ---------------------------------------------------------
# 2. SIREN Layer Class
# ---------------------------------------------------------
class SineLayer(nn.Module):
    """
    SIRENのためのSine活性化層
    """

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


# ---------------------------------------------------------
# 3. DeepSDF with SIREN + Positional Encoding
# ---------------------------------------------------------
class DeepSDF(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        latent_code_dim=128,
        num_layers=5,
        xyz_pos_enc_dim=10,  # デフォルト値を少し上げておきます(通常10程度が推奨)
        dropout_prob=0.0,
    ):
        super().__init__()

        # Positional Encodingの定義
        self.positional_encoding = PositionalEncodingXYZ(hidden_dim=xyz_pos_enc_dim)

        self.net = nn.ModuleList()

        # 入力次元の計算
        # PosEncによって各座標軸(x,y,z)が 2 * xyz_pos_enc_dim になるため
        # 合計 3 * 2 * xyz_pos_enc_dim = 6 * xyz_pos_enc_dim
        input_dim = (6 * xyz_pos_enc_dim) + latent_code_dim

        # 最初の層
        self.net.append(SineLayer(input_dim, hidden_dim, is_first=True, omega_0=30.0))

        # 中間層
        for _ in range(num_layers - 2):
            self.net.append(
                SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=1.0)
            )

        # 最終層 (Linear)
        self.final_linear = nn.Linear(hidden_dim, 1)

        # 初期化（最終層）
        with torch.no_grad():
            self.final_linear.weight.uniform_(
                -np.sqrt(6 / hidden_dim) / 30, np.sqrt(6 / hidden_dim) / 30
            )

    def forward(self, latent, xyz):
        # 1. 座標をPositional Encodingで高次元化
        # xyz: (B, 3) -> xyz_encoded: (B, 6 * xyz_pos_enc_dim)
        xyz_encoded = self.positional_encoding(xyz)

        # 2. latentをバッチサイズに合わせて拡張
        # latentが(1, latent_dim)等で来ても(B, latent_dim)に合わせる
        if latent.shape[0] != xyz.shape[0]:
            latent = latent.repeat(xyz.shape[0], 1)

        # 3. 結合
        x = torch.cat([xyz_encoded, latent], dim=-1)

        # 4. SIREN Layers
        for layer in self.net:
            x = layer(x)

        return self.final_linear(x)
