import os
import pickle
import sys
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from experimet_config import Config
from tqdm.auto import tqdm

sys.path.append("../")
# モデル定義もimportが必要です (HashGridEncoderなどのクラス定義が必要なため)
# src/models/deepsdf.py に定義があると仮定します
from src.models.deepsdf import GeneralizableDeepSDF

# create_meshは修正済みのものをimport
from src.utils.mesh_utils import create_mesh


# ==========================================
# 2. Wrapper for create_mesh (HashGrid対応版)
# ==========================================
class DecoderInferenceWrapper(nn.Module):
    """
    create_mesh関数が期待する interface: model(latent, xyz) に合わせるためのラッパー。
    ここでの 'latent' は Encoderが出力した 'feature_grid' です。
    """

    def __init__(self, parent_model):
        super().__init__()
        self.decoder_net = parent_model.decoder_net
        self.final_layer = parent_model.final_layer

        # Positional Encoding (ある場合)
        self.pos_enc = getattr(parent_model, "pos_enc", None)

        # HashGrid Encoder (ある場合)
        self.coord_encoder = getattr(parent_model, "coord_encoder", None)
        self.use_hashgrid = getattr(parent_model, "use_hashgrid", False)

    def forward(self, feature_grid, query_xyz):
        # feature_grid: (1, C, D, H, W)
        # query_xyz: (1, N, 3)

        # 1. Grid Sample (CT特徴量の取得)
        # sample_coords: (1, 1, 1, N, 3)
        sample_coords = query_xyz.view(query_xyz.shape[0], 1, 1, -1, 3)

        features = F.grid_sample(
            feature_grid,
            sample_coords,
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        )
        # (1, F, 1, 1, N) -> (1, N, F)
        features_query = features.squeeze(2).squeeze(2).transpose(1, 2)

        # 2. Coordinate Encoding & Combine
        # ここで HashGrid か PosEnc か Raw XYZ かを分岐

        if self.use_hashgrid and self.coord_encoder is not None:
            # HashGridの場合
            # [0, 1] に正規化してからエンコード
            xyz_norm = (query_xyz + 1.0) / 2.0
            xyz_norm = torch.clamp(xyz_norm, 0.0, 1.0)

            coord_features = self.coord_encoder(xyz_norm)
            decoder_input = torch.cat([coord_features, features_query], dim=-1)

        elif self.pos_enc is not None:
            # Positional Encodingの場合
            encode_xyz = self.pos_enc(query_xyz)
            decoder_input = torch.cat([encode_xyz, features_query], dim=-1)

        else:
            # 何もしない場合 (Raw XYZ)
            decoder_input = torch.cat([query_xyz, features_query], dim=-1)

        # 3. Decode (SIREN)
        x = decoder_input
        for layer in self.decoder_net:
            x = layer(x)
        output = self.final_layer(x)

        return output


# ==========================================
# 3. Main Script
# ==========================================
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


cfg = Config()

# loading mapping dicts
# pickleがない場合は直接globでフォルダ探索してもOK
with open("../outputs/data-split/carfolder2latent_idx.pickle", "rb") as f:
    carfolder2latent_idx = pickle.load(f)

# loading GeneralizableDeepSDF model
# ★重要: train.py と同じパラメータで初期化してください
# 例: use_hashgrid=True にしていたならここでもTrueにする
model = GeneralizableDeepSDF(
    feature_dim=64,
    hidden_dim=cfg.hidden_dim,
    # xyz_pos_enc_dim=cfg.xyz_pos_enc_dim, # PosEncを使う場合
    # use_hashgrid=True,                   # HashGridを使う場合
    # n_levels=16,                         # HashGridパラメータ
).to(cfg.device)

# Load weights
# weight_path = "../outputs/models/deepsdf_encoder.pth"
weight_path = "../outputs/models/deepsdf_encoder_last.pth"
if os.path.exists(weight_path):
    model.load_state_dict(torch.load(weight_path))
    print(f"Loaded model from {weight_path}")
else:
    raise FileNotFoundError(f"Model file not found: {weight_path}")

model.eval()

# Wrap model for inference
inference_model = DecoderInferenceWrapper(model)

# Output directory
output_base_dir = "../outputs/mesh/training-ct-reconstruction"
create_dir(output_base_dir)

# generate 3d data
for carfolder in tqdm(carfolder2latent_idx.keys()):
    carname = carfolder

    # 1. Find CT File
    ct_path_pattern = f"../../../input/03_sdf_dataset/{carname}/ct320.npy"
    ct_files = glob(ct_path_pattern)

    if not ct_files:
        print(f"Skipping {carname}: ct320.npy not found.")
        continue

    ct_file = ct_files[0]

    # 2. Load CT & Encode to Feature Grid
    try:
        # Load numpy
        ct_data = np.load(ct_file)

        # --- ★重要: 学習時と同じ前処理 ---
        # 1. 軸の転置 (Z, Y, X) -> (X, Y, Z)
        # もし学習時の __getitem__ で transpose(2, 1, 0) していたらここでも必須
        ct_data = np.transpose(ct_data, (2, 1, 0))

        # 2. 正規化 0~255 -> 0.0~1.0
        ct_data = ct_data.astype(np.float32) / 255.0

        # To Tensor: (1, 1, D, H, W)
        ct_tensor = torch.from_numpy(ct_data).unsqueeze(0).unsqueeze(0).to(cfg.device)

        # Run Encoder (Once per shape)
        with torch.no_grad():
            feature_grid = model.encoder(ct_tensor)

    except Exception as e:
        print(f"Error processing {carname}: {e}")
        continue

    # 3. Generate Mesh
    output_filename = os.path.join(output_base_dir, carname)

    create_mesh(
        model=inference_model,
        latent=feature_grid,
        output_file=output_filename,
        N=cfg.train_latent_output_resolution,
    )
