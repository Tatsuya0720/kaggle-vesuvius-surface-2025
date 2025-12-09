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
from src.models.deepsdf import GeneralizableDeepSDF

# create_meshは、先ほど修正していただいたバージョンのものをimportしてください
from src.utils.mesh_utils import create_mesh


# ==========================================
# 2. Wrapper for create_mesh
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
        self.pos_enc = parent_model.pos_enc

    def forward(self, feature_grid, query_xyz):
        # feature_grid: (1, C, D, H, W)  <- create_meshから渡される
        # query_xyz: (1, N, 3)           <- create_meshから渡される (先ほどの修正でunsqueeze(0)済み)

        # 1. Grid Sample
        # query_xyzに対応する特徴量をGridから取得
        # sample_coords: (1, 1, 1, N, 3)
        sample_coords = query_xyz.view(query_xyz.shape[0], 1, 1, -1, 3)
        # sample_coords = query_xyz.unsqueeze(1).unsqueeze(1)

        features = F.grid_sample(
            feature_grid,
            sample_coords,
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        )
        # (1, F, 1, 1, N) -> (1, N, F)
        # features_query = features.view(query_xyz.shape[0], -1, feature_grid.shape[1])
        features_query = features.squeeze(2).squeeze(2).transpose(1, 2)

        # 2. Decode (SIREN)
        # (1, N, 3+F)
        encode_xyz = self.pos_enc(query_xyz) if self.pos_enc else query_xyz
        decoder_input = torch.cat([encode_xyz, features_query], dim=-1)
        # decoder_input = query_xyz

        x = decoder_input
        for layer in self.decoder_net:
            x = layer(x)
        output = self.final_layer(x)

        # (1, N, 1) -> (1, N) -> create_mesh側でsqueezeされるのでこれでOK
        return output


# ==========================================
# 3. Main Script
# ==========================================
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


cfg = Config()

# loading mapping dicts
with open("../outputs/data-split/carfolder2latent_idx.pickle", "rb") as f:
    carfolder2latent_idx = pickle.load(f)

# (latent_codes.pth の読み込みは削除しました)

# loading GeneralizableDeepSDF model
# パラメータはtrain.pyの設定に合わせてください
model = GeneralizableDeepSDF(feature_dim=64, hidden_dim=cfg.hidden_dim).to(cfg.device)

# Load weights
weight_path = "../outputs/models/deepsdf_encoder.pth"
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
# carfolder2latent_idx のキー（フォルダ名）を使ってループ
for carfolder in tqdm(carfolder2latent_idx.keys()):
    carname = carfolder

    # 1. Find CT File
    # 実際のデータセットパスに合わせて調整してください
    # 例: ../../../input/02_sdf_dataset_by10/{carname}/ct320.npy
    ct_path_pattern = f"../../../input/03_sdf_dataset/{carname}/ct320.npy"
    ct_files = glob(ct_path_pattern)

    if not ct_files:
        print(f"Skipping {carname}: ct320.npy not found.")
        continue

    ct_file = ct_files[0]

    # 2. Load CT & Encode to Feature Grid
    try:
        # Load numpy
        ct_data = np.load(ct_file).astype(np.float32)
        # To Tensor: (1, 1, D, H, W)
        ct_tensor = torch.from_numpy(ct_data).unsqueeze(0).unsqueeze(0).to(cfg.device)

        # Run Encoder (Once per shape)
        with torch.no_grad():
            feature_grid = model.encoder(ct_tensor)

    except Exception as e:
        print(f"Error processing {carname}: {e}")
        continue

    # 3. Generate Mesh
    # feature_grid を 'latent' 引数として渡します
    output_filename = os.path.join(output_base_dir, carname)

    create_mesh(
        model=inference_model,  # Wrapper model
        latent=feature_grid,  # Encoder output as latent
        output_file=output_filename,
        N=cfg.train_latent_output_resolution,
    )
