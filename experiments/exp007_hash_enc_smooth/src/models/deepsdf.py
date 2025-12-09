import numpy as np
import torch
import torch.nn as nn


# -----------------------------------------------------------
# 1. HashGridEncoder (変更なし・前回と同じもの)
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
        features = []
        for i in range(self.n_levels):
            resolution = np.floor(self.base_resolution * (self.b**i))
            x_scaled = x * resolution
            x0 = torch.floor(x_scaled).long()
            weights = x_scaled - x0.float()
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

        return all_features


# -----------------------------------------------------------
# 2. SineLayer (SIRENの構成要素)
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
# 3. Hybrid DeepSDF (HashGrid + SIREN)
# -----------------------------------------------------------
class DeepSDF(nn.Module):
    def __init__(
        self,
        hidden_dim=64,  # SIREN MLPの隠れ層
        latent_code_dim=128,
        num_layers=2,  # 層数は2〜3で十分
        dropout_prob=0.0,
        xyz_pos_enc_dim=0,  # 未使用
        # HashGridパラメータ
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        desired_resolution=2048,
    ):
        super().__init__()

        # Encoder: 空間情報を詳細に保持 (Hash Grid)
        self.encoder = HashGridEncoder(
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            desired_resolution=desired_resolution,
        )

        encoder_output_dim = n_levels * n_features_per_level
        input_dim = encoder_output_dim + latent_code_dim

        # Decoder: 滑らかさを強制 (SIREN)
        # ReLUの代わりにSineLayerを使用します
        self.net = nn.ModuleList()

        # 最初の層:
        # HashGridからの入力は既に正規化された特徴量なので、omega_0は1.0 (または小さい値) でOK
        # ここを30.0にするとHashGridの高周波ノイズが増幅されすぎる危険があります
        self.net.append(SineLayer(input_dim, hidden_dim, is_first=True, omega_0=1.0))

        # 中間層
        for _ in range(num_layers - 1):
            self.net.append(
                SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=1.0)
            )

        # 最終層: SDF値を出力 (Linear)
        self.final_linear = nn.Linear(hidden_dim, 1)

        # 最終層の初期化 (SIREN推奨)
        with torch.no_grad():
            self.final_linear.weight.uniform_(
                -np.sqrt(6 / hidden_dim) / 30, np.sqrt(6 / hidden_dim) / 30
            )

    def forward(self, latent, xyz, level_mask=None):
        # 1. 座標正規化
        xyz_norm = (xyz + 1.0) / 2.0
        xyz_norm = torch.clamp(xyz_norm, 0.0, 1.0)

        # 2. Hash Grid Encoding
        grid_features = self.encoder(xyz_norm, level_mask=level_mask)

        # 3. Latent結合
        if latent.shape[0] != xyz.shape[0]:
            latent = latent.repeat(xyz.shape[0], 1)

        x = torch.cat([grid_features, latent], dim=-1)

        # 4. SIREN MLP Forward
        for layer in self.net:
            x = layer(x)

        return self.final_linear(x)
