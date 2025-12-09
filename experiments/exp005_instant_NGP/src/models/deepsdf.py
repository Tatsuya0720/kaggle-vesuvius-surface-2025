import numpy as np
import torch
import torch.nn as nn


class HashGridEncoder(nn.Module):
    def __init__(
        self,
        n_levels=16,  # 解像度レベル数 (16が標準)
        n_features_per_level=2,  # 各レベルの特徴量次元 (2が標準)
        log2_hashmap_size=19,  # ハッシュテーブルの大きさ (2^19)
        base_resolution=16,  # 最小解像度
        desired_resolution=2048,  # 最大解像度 (詳細度に合わせて調整)
    ):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.desired_resolution = desired_resolution

        # 解像度の成長率を計算 (b)
        self.b = np.exp(
            (np.log(desired_resolution) - np.log(base_resolution)) / (n_levels - 1)
        )

        # 学習可能な埋め込みテーブル (Hash Table)
        # サイズ: (Total Features, 1) -> viewして使う
        self.embeddings = nn.Parameter(
            torch.nn.init.uniform_(
                torch.empty(n_levels * (2**log2_hashmap_size), n_features_per_level),
                a=-1e-4,
                b=1e-4,
            )
        )

        # ハッシュ計算用の大きな素数
        self.primes = [1, 2654435761, 805459861]

    def forward(self, x):
        # x: (B, 3) range in [0, 1] (呼び出し元で正規化済みを想定)

        # 出力用リスト
        features = []

        # 各レベルごとの処理
        for i in range(self.n_levels):
            resolution = np.floor(self.base_resolution * (self.b**i))

            # 1. 座標をグリッド座標へスケーリング
            x_scaled = x * resolution

            # 2. 格子の頂点座標 (floor / ceil)
            x0 = torch.floor(x_scaled).long()
            x1 = x0 + 1

            # 3. 補間のための重み (local coordinates)
            weights = x_scaled - x0.float()  # (B, 3)
            fx, fy, fz = weights[:, 0], weights[:, 1], weights[:, 2]  # (B,)

            # 4. 8つの角のインデックスを作成
            # 000, 001, 010, ..., 111
            grid_indices = []
            for dz in [0, 1]:
                for dy in [0, 1]:
                    for dx in [0, 1]:
                        # 角の座標
                        corner = torch.stack(
                            [x0[:, 0] + dx, x0[:, 1] + dy, x0[:, 2] + dz], dim=-1
                        )

                        # ハッシュ計算 (XOR hashing)
                        # h = (x*p1 ^ y*p2 ^ z*p3) % table_size
                        h = (
                            (corner[:, 0] * self.primes[0])
                            ^ (corner[:, 1] * self.primes[1])
                            ^ (corner[:, 2] * self.primes[2])
                        ) % (2**self.log2_hashmap_size)

                        # レベルごとのオフセットを加算してグローバルインデックスにする
                        global_idx = h + (i * (2**self.log2_hashmap_size))
                        grid_indices.append(global_idx)

            # 5. 特徴量を取得 (B, 8, feature_dim)
            # grid_indices stack -> (8, B) -> (B, 8)
            indices = torch.stack(grid_indices, dim=-1)
            lookup = self.embeddings[indices]  # (B, 8, F)

            # 6. 三重線形補間 (Trilinear Interpolation)
            # c00, c01, c10, c11 (Z軸方向でまずまとめる)
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

        # 全レベルの特徴を結合 (B, n_levels * n_features_per_level)
        return torch.cat(features, dim=-1)


class DeepSDF(nn.Module):
    def __init__(
        self,
        hidden_dim=64,  # Instant-NGPのMLPは小さくて良い (64程度)
        latent_code_dim=128,
        num_layers=2,  # MLPの層数も少なくて良い (2-3層)
        dropout_prob=0.0,
        xyz_pos_enc_dim=0,
        # 以下HashGrid用パラメータ
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        desired_resolution=2048,
    ):
        super().__init__()

        # 1. Encoder (Hash Grid)
        self.encoder = HashGridEncoder(
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            desired_resolution=desired_resolution,
        )

        # エンコーダの出力次元
        encoder_output_dim = n_levels * n_features_per_level

        # MLPへの入力次元 = Grid特徴量 + Latent Code
        input_dim = encoder_output_dim + latent_code_dim

        # 2. Decoder (Small MLP)
        # Instant-NGPでは大きなMLPは不要。ReLUで十分高速・高精度。
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        self.mlp_body = nn.Sequential(*layers)

        # 最終層 (SDF出力)
        self.output_layer = nn.Linear(hidden_dim, 1)

        # 以前のPositional Encoding等は不要になったので削除

    def forward(self, latent, xyz):
        # latent: (B, latent_dim) or (1, latent_dim)
        # xyz: (B, 3) assumed range approximately [-1, 1]

        # 1. 座標の正規化 [-1, 1] -> [0, 1]
        # HashGridは0-1範囲で動作するため
        # ※もしxyzが最初から0-1ならこの行は削除してください
        xyz_norm = (xyz + 1.0) / 2.0

        # 境界クランプ (わずかに誤差ではみ出すのを防ぐ)
        xyz_norm = torch.clamp(xyz_norm, 0.0, 1.0)

        # 2. Grid Encoding
        # (B, 32) (16 levels * 2 dimsの場合)
        grid_features = self.encoder(xyz_norm)

        # 3. Latentとの結合
        if latent.shape[0] != xyz.shape[0]:
            latent = latent.repeat(xyz.shape[0], 1)

        features = torch.cat([grid_features, latent], dim=-1)

        # 4. MLP Forward
        h = self.mlp_body(features)
        out = self.output_layer(h)

        return out
