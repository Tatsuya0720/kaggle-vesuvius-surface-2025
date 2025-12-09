import numpy as np
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    """
    SIRENのためのSine活性化層
    omega_0: 周波数パラメータ。最初の層は30、以降は1などが推奨される
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


class DeepSDF(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        latent_code_dim=128,
        num_layers=5,
        xyz_pos_enc_dim=3,
        dropout_prob=0.0,
    ):
        super().__init__()

        # SIRENではPositional Encodingは不要（層の中で周波数を扱うため）

        self.net = nn.ModuleList()

        # 入力: 座標(3) + Latent(128)
        self.net.append(
            SineLayer(3 + latent_code_dim, hidden_dim, is_first=True, omega_0=30.0)
        )

        for _ in range(num_layers - 2):
            self.net.append(
                SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=1.0)
            )

        # 最終層はSDF値を出力するためLinear (活性化なし)
        self.final_linear = nn.Linear(hidden_dim, 1)

        # 初期化（最終層）
        with torch.no_grad():
            self.final_linear.weight.uniform_(
                -np.sqrt(6 / hidden_dim) / 30, np.sqrt(6 / hidden_dim) / 30
            )

    def forward(self, latent, xyz):
        # latentをバッチサイズに合わせて拡張
        latent = latent.repeat(xyz.shape[0], 1)
        x = torch.cat([xyz, latent], dim=-1)

        for layer in self.net:
            x = layer(x)

        return self.final_linear(x)
