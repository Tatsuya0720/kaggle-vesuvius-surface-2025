import os
import pickle
import sys
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configの読み込み
from experimet_config import Config
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

sys.path.append("../")
# DeepSDFDatasetは正規化(0~1)対応済みのものをimportしていると仮定します
from src.datasets.deepsdf_dataset import DeepSDFDataset
from src.models.deepsdf import GeneralizableDeepSDF
from src.utils.metric import DeepSDFLoss, mse
from src.utils.utils import file_id_latent_code_connetction, set_seed

# ==========================================
# 2. Setup
# ==========================================
cfg = Config()
set_seed(42)

# loading training data path
# points_path = glob("../../../input/02_sdf_dataset_by10/**/points.npy")
points_path = glob("../../../input/03_sdf_dataset/**/points.npy")
# points_path = points_path[:1]

# モデル初期化
model = GeneralizableDeepSDF(feature_dim=64, hidden_dim=cfg.hidden_dim).to(cfg.device)

# ダミーのID管理（Dataset互換用）
dummy_codes = torch.zeros(len(points_path), 1)
latent_idx2carfolder, carfolder2latent_idx = file_id_latent_code_connetction(
    dummy_codes, points_path
)

dataset = DeepSDFDataset(
    points_path, carfolder2latent_idx, subsample=cfg.sample_per_scene
)
dataloader = DataLoader(
    dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    num_workers=4,
)

criterion = DeepSDFLoss(delta=cfg.clamping_distance)

# Optimizer: LatentCodeは無いのでモデルパラメータのみ
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.deepsdf_initial_lr)

warmup_steps = int(len(dataloader) * cfg.warmup_epoch)
num_training_steps = int(len(dataloader) * cfg.n_epochs)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps,
)

# save metadata
os.makedirs("../outputs/data-split/", exist_ok=True)
with open("../outputs/data-split/latent_idx2carfolder.pickle", "wb") as f:
    pickle.dump(latent_idx2carfolder, f)

with open("../outputs/data-split/carfolder2latent_idx.pickle", "wb") as f:
    pickle.dump(carfolder2latent_idx, f)


# ==========================================
# 3. Training Loop
# ==========================================
best_val_mse = np.inf
best_model = None
minimum_mse = np.inf

print("Start Training with Generalizable DeepSDF (Simple Loss)...")

for epoch in range(cfg.n_epochs):
    model.train()
    total_loss = []
    total_mse = []
    tq = tqdm(dataloader)

    for data in dataloader:
        # データロード
        xyz_batch = data["points"].float().cuda().chunk(cfg.batch_size)
        sdf_batch = data["sdf"].float().cuda().chunk(cfg.batch_size)

        # CT Volume: Datasetで正規化(0~1)済みと仮定
        ct_batch = data["ct"].float().cuda().unsqueeze(1)  # (B, 1, D, H, W)

        optimizer.zero_grad()
        batch_loss = []
        batch_mse = []

        for i in range(cfg.batch_size):
            # 1サンプルのデータ準備
            onecar_ct = ct_batch[i].unsqueeze(0)  # (1, 1, D, H, W)
            onecar_xyz = xyz_batch[i].squeeze(0).to(cfg.device)  # (N, 3)
            onecar_sdf = sdf_batch[i].squeeze(0).to(cfg.device)  # (N, 1)

            # ※ シンプル学習なので requires_grad_(True) は不要です

            # Forward
            # onecar_xyz: (N, 3) -> モデル内で (1, N, 3) として処理
            onecar_sdf_pred = model(onecar_ct, onecar_xyz.unsqueeze(0))
            onecar_sdf_pred = onecar_sdf_pred.squeeze(0)  # (N, 1)

            # Loss Calculation (Simple Reconstruction Only)
            num_samples = onecar_sdf.shape[0]
            loss = criterion(onecar_sdf_pred, onecar_sdf) / num_samples

            # Backward
            loss.backward()

            # Monitor
            with torch.no_grad():
                monitor_loss = mse(onecar_sdf_pred, onecar_sdf)

            batch_loss.append(loss.item())
            batch_mse.append(monitor_loss.item())

        optimizer.step()
        scheduler.step()

        total_loss.append(np.mean(batch_loss))
        total_mse.append(np.mean(batch_mse))

        tq.update()
        tq.set_postfix(
            {
                "epoch": epoch,
                "loss": f"{np.mean(total_loss):.5f}",
                "mse": f"{np.mean(total_mse):.5f}",
                "min_mse": f"{minimum_mse:.5f}",
            }
        )
    tq.close()

    # Save Best Model
    current_mse = np.mean(total_mse)
    if minimum_mse > current_mse:
        minimum_mse = current_mse
        os.makedirs("../outputs/models/", exist_ok=True)
        torch.save(model.state_dict(), "../outputs/models/deepsdf_encoder.pth")
        print(f"Model saved at epoch {epoch} with MSE: {minimum_mse:.6f}")
