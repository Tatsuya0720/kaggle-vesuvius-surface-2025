import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------
# 1. HashGridEncoder (提示いただいたものそのまま)
# -----------------------------------------------------------
class HashGridEncoder(nn.Module):
    def __init__(
        self,
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        desired_resolution=2048,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.desired_resolution = desired_resolution

        self.b = np.exp(
            (np.log(desired_resolution) - np.log(base_resolution)) / (n_levels - 1)
        )

        self.embeddings = nn.Parameter(
            torch.nn.init.uniform_(
                torch.empty(n_levels * (2**log2_hashmap_size), n_features_per_level),
                a=-1e-4,
                b=1e-4,
            )
        )
        self.primes = [1, 2654435761, 805459861]

    def forward(self, x, level_mask=None):
        # x: (B, N, 3) または (N, 3) など

        # --- 修正点1: 入力の形状を記憶し、(Total_Points, 3) に平坦化する ---
        input_shape = x.shape
        x = x.reshape(-1, 3)
        # -------------------------------------------------------------

        features = []
        for i in range(self.n_levels):
            resolution = np.floor(self.base_resolution * (self.b**i))
            x_scaled = x * resolution
            x0 = torch.floor(x_scaled).long()

            # 微分可能なweight計算
            weights = x_scaled - x0
            fx, fy, fz = weights[:, 0], weights[:, 1], weights[:, 2]

            grid_indices = []
            for dz in [0, 1]:
                for dy in [0, 1]:
                    for dx in [0, 1]:
                        corner = torch.stack(
                            [x0[:, 0] + dx, x0[:, 1] + dy, x0[:, 2] + dz], dim=-1
                        )
                        h = (
                            (corner[:, 0] * self.primes[0])
                            ^ (corner[:, 1] * self.primes[1])
                            ^ (corner[:, 2] * self.primes[2])
                        ) % (2**self.log2_hashmap_size)
                        global_idx = h + (i * (2**self.log2_hashmap_size))
                        grid_indices.append(global_idx)

            indices = torch.stack(grid_indices, dim=-1)
            lookup = self.embeddings[indices]

            c00 = lookup[:, 0] * (1 - fx).unsqueeze(-1) + lookup[:, 1] * fx.unsqueeze(
                -1
            )
            c01 = lookup[:, 2] * (1 - fx).unsqueeze(-1) + lookup[:, 3] * fx.unsqueeze(
                -1
            )
            c10 = lookup[:, 4] * (1 - fx).unsqueeze(-1) + lookup[:, 5] * fx.unsqueeze(
                -1
            )
            c11 = lookup[:, 6] * (1 - fx).unsqueeze(-1) + lookup[:, 7] * fx.unsqueeze(
                -1
            )
            c0 = c00 * (1 - fy).unsqueeze(-1) + c01 * fy.unsqueeze(-1)
            c1 = c10 * (1 - fy).unsqueeze(-1) + c11 * fy.unsqueeze(-1)
            c = c0 * (1 - fz).unsqueeze(-1) + c1 * fz.unsqueeze(-1)

            features.append(c)

        all_features = torch.cat(features, dim=-1)

        if level_mask is not None:
            active_levels = level_mask * self.n_levels
            level_idx = torch.arange(self.n_levels, device=x.device).repeat_interleave(
                self.n_features_per_level
            )
            mask = torch.clamp(active_levels - level_idx, 0.0, 1.0)
            all_features = all_features * mask.unsqueeze(0)

        # --- 修正点2: 元のバッチ形状 (B, N, FeatureDim) に戻す ---
        all_features = all_features.reshape(*input_shape[:-1], -1)
        # ---------------------------------------------------

        return all_features


# -----------------------------------------------------------
# 2. ResBlock & CTEncoder3D (軽量版)
# -----------------------------------------------------------
class ResBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class CTEncoder3D(nn.Module):
    def __init__(self, input_channels=1, feature_dim=64):
        super().__init__()
        self.conv_in = nn.Conv3d(
            input_channels, 16, kernel_size=3, padding=1, bias=False
        )
        self.norm_in = nn.InstanceNorm3d(16)

        self.down1 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(32)
        self.res_blocks1 = nn.Sequential(ResBlock3D(32))

        self.down2 = nn.Conv3d(32, 48, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(48)
        self.res_blocks2 = nn.Sequential(ResBlock3D(48), ResBlock3D(48))

        self.conv_out = nn.Conv3d(48, feature_dim, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (B, 1, D, H, W)
        x = F.relu(self.norm_in(self.conv_in(x)))
        x = F.relu(self.norm1(self.down1(x)))
        x = self.res_blocks1(x)
        x = F.relu(self.norm2(self.down2(x)))
        x = self.res_blocks2(x)
        x = self.conv_out(x)
        return x


# -----------------------------------------------------------
# 3. SineLayer
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


# -----------------------------------------------------------
# 4. GeneralizableDeepSDF (HashGrid + CTEncoder)
# -----------------------------------------------------------
class GeneralizableDeepSDF(nn.Module):
    def __init__(
        self,
        feature_dim=64,
        hidden_dim=64,
        num_layers=3,
        # HashGrid設定
        use_hashgrid=True,
        n_levels=16,
        log2_hashmap_size=19,
    ):
        super().__init__()

        # 1. CT Encoder
        self.encoder = CTEncoder3D(feature_dim=feature_dim)

        # 2. Coordinate Encoder (HashGrid or Raw XYZ)
        self.use_hashgrid = use_hashgrid
        if use_hashgrid:
            self.coord_encoder = HashGridEncoder(
                n_levels=n_levels,
                n_features_per_level=2,
                log2_hashmap_size=log2_hashmap_size,
                base_resolution=16,
                desired_resolution=2048,
            )
            # 入力次元 = CT特徴量 + HashGrid特徴量
            coord_dim = n_levels * 2
        else:
            self.coord_encoder = None
            coord_dim = 3  # raw xyz

        input_dim = feature_dim + coord_dim

        # 3. SIREN Decoder
        self.decoder_net = nn.ModuleList()

        # 最初の層
        # HashGrid特徴量は既に高周波なので、SIRENのomega_0は1.0でOK
        omega_0 = 1.0 if use_hashgrid else 30.0
        self.decoder_net.append(
            SineLayer(input_dim, hidden_dim, is_first=True, omega_0=omega_0)
        )

        for _ in range(num_layers - 1):
            self.decoder_net.append(
                SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0)
            )

        self.final_layer = nn.Linear(hidden_dim, 1)

        with torch.no_grad():
            self.final_layer.weight.uniform_(
                -np.sqrt(6 / hidden_dim) / 30, np.sqrt(6 / hidden_dim) / 30
            )

    def forward(self, ct_volume, query_xyz):
        """
        ct_volume: (B, 1, D, H, W)
        query_xyz: (B, N_points, 3) in [-1, 1]
        """

        # 1. CT特徴量 (Generalization)
        feature_grid = self.encoder(ct_volume)

        sample_coords = query_xyz.view(query_xyz.shape[0], 1, 1, -1, 3)
        features = F.grid_sample(
            feature_grid,
            sample_coords.detach(),  # 勾配切断 (安定化のため)
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        )
        features_query = features.squeeze(2).squeeze(2).transpose(1, 2)  # (B, N, F)

        # 2. 座標特徴量 (High-Frequency Detail)
        if self.use_hashgrid:
            # HashGridは [0, 1] を期待するため正規化
            xyz_norm = (query_xyz + 1.0) / 2.0
            xyz_norm = torch.clamp(xyz_norm, 0.0, 1.0)  # 安全のためClamp

            coord_features = self.coord_encoder(xyz_norm)  # (B, N, L*2)

            # 結合
            decoder_input = torch.cat([coord_features, features_query], dim=-1)
        else:
            # Raw XYZ
            decoder_input = torch.cat([query_xyz, features_query], dim=-1)

        # 3. Decode
        x = decoder_input
        for layer in self.decoder_net:
            x = layer(x)
        output = self.final_layer(x)

        return output
