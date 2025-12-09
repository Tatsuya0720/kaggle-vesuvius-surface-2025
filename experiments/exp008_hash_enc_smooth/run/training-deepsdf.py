# import os
# import pickle
# import sys
# from glob import glob

# import numpy as np
# import torch
# import torch.nn as nn
# from experimet_config import Config

# # Eikonal Loss計算のためにautogradが必要
# from torch import autograd
# from torch.utils.data import DataLoader
# from tqdm.auto import tqdm
# from transformers import get_cosine_schedule_with_warmup

# sys.path.append("../")
# from src.datasets.deepsdf_dataset import DeepSDFDataset
# from src.models.deepsdf import DeepSDF
# from src.utils.metric import DeepSDFLoss, mse
# from src.utils.utils import file_id_latent_code_connetction, set_seed

# cfg = Config()
# set_seed(42)

# # --- 追加設定 (Configにない場合はここで調整してください) ---
# LAMBDA_EIKONAL = getattr(cfg, "lambda_eikonal", 0.1)  # Eikonal Lossの重み
# JITTER_SCALE = getattr(cfg, "jitter_scale", 0.005)  # ノイズの強さ
# START_MASK_RATIO = 0.0  # 学習初期のマスク解像度割合

# # loading training data path
# points_path = glob("../../../input/01_sdf_dataset/**/points.npy")
# points_path = points_path[:1]  # テスト時はコメントアウト解除
# latent_codes = nn.Parameter(
#     torch.normal(
#         0, 1e-4, size=(len(points_path), cfg.latent_code_dim)
#     )  # (car_model_num, latent_code_dim)
# )

# # モデル初期化 (HashGrid版のパラメータに合わせて変更)
# # ※前回のコードに合わせて引数を設定しています
# model = DeepSDF(
#     hidden_dim=cfg.hidden_dim,  # MLPの隠れ層次元 (HashGridなら64程度で十分)
#     latent_code_dim=cfg.latent_code_dim,
#     dropout_prob=cfg.dropout_prob,
#     # HashGrid用パラメータ (cfgになければデフォルト値を使用)
#     n_levels=getattr(cfg, "n_levels", 16),
#     n_features_per_level=getattr(cfg, "n_features_per_level", 2),
#     log2_hashmap_size=getattr(cfg, "log2_hashmap_size", 19),
#     base_resolution=getattr(cfg, "base_resolution", 16),
#     desired_resolution=getattr(cfg, "desired_resolution", 2048),
# ).to(cfg.device)

# latent_idx2carfolder, carfolder2latent_idx = file_id_latent_code_connetction(
#     latent_codes, points_path
# )

# dataset = DeepSDFDataset(
#     points_path, carfolder2latent_idx, subsample=cfg.sample_per_scene
# )
# dataloader = DataLoader(
#     dataset,
#     batch_size=cfg.batch_size,
#     shuffle=True,
#     pin_memory=True,
#     drop_last=True,
#     num_workers=20,
# )

# criterion = DeepSDFLoss(delta=cfg.clamping_distance)
# optimizer = torch.optim.Adam(
#     [
#         {
#             "params": model.parameters(),
#             "lr": cfg.deepsdf_initial_lr,
#         },
#         {
#             "params": latent_codes,
#             "lr": cfg.latent_code_inital_lr,
#         },
#     ]
# )

# warmup_steps = int(len(dataloader) * cfg.warmup_epoch)
# num_training_steps = int(len(dataloader) * cfg.n_epochs)

# scheduler = get_cosine_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=warmup_steps,
#     num_training_steps=num_training_steps,
# )

# # save
# os.makedirs("../outputs/data-split/", exist_ok=True)
# with open("../outputs/data-split/latent_idx2carfolder.pickle", "wb") as f:
#     pickle.dump(latent_idx2carfolder, f)

# with open("../outputs/data-split/carfolder2latent_idx.pickle", "wb") as f:
#     pickle.dump(carfolder2latent_idx, f)


# # let's train
# best_val_mse = np.inf
# best_model = None
# minimum_mse = np.inf

# for epoch in range(cfg.n_epochs):
#     model.train()
#     total_loss = []
#     total_mse = []
#     tq = tqdm(dataloader)

#     # --- [Coarse-to-Fine] マスク率の計算 ---
#     # エポックが進むにつれて 0.2 -> 1.0 にマスクを開放していく
#     progress = epoch / cfg.n_epochs
#     mask_ratio = START_MASK_RATIO + (1.0 - START_MASK_RATIO) * progress
#     mask_ratio = min(1.0, mask_ratio)

#     for data in dataloader:
#         # データロード
#         xyz_batch = data["points"].float().cuda().chunk(cfg.batch_size)
#         sdf_batch = data["sdf"].float().cuda().chunk(cfg.batch_size)
#         file_id = data["file_id"]

#         # 今回のバッチ内の全サンプル数（Lossの正規化に使用）
#         # ※勾配累積するので、batch_sizeで割る形になりますが、
#         #  元のコードロジック(num_sdf_samplesで割る)を尊重します。
#         num_sdf_samples = cfg.sample_per_scene * cfg.batch_size

#         optimizer.zero_grad()
#         batch_loss = []
#         batch_mse = []

#         # バッチ内の各データ（車/シーン）ごとに処理
#         for i in range(cfg.batch_size):
#             latent_code = (
#                 latent_codes[carfolder2latent_idx[file_id[i]], :]
#                 .to(cfg.device)
#                 .unsqueeze(0)
#             )  # [1, latent_code_dim]

#             onecar_xyz = xyz_batch[i].squeeze(0).to(cfg.device)  # [sample_per_car, 3]
#             onecar_sdf = sdf_batch[i].squeeze(0).to(cfg.device)

#             # --- [Jittering] 入力ノイズ付加 ---
#             # 点群の隙間を埋めるため、学習時のみ座標を揺らす
#             if True:  # model.training is True inside this loop
#                 noise = torch.randn_like(onecar_xyz) * JITTER_SCALE
#                 onecar_xyz_input = onecar_xyz + noise
#             else:
#                 onecar_xyz_input = onecar_xyz

#             # --- [Eikonal] 勾配計算の準備 ---
#             onecar_xyz_input.requires_grad_(True)

#             # --- Forward (with Mask) ---
#             # modelのforwardにlevel_maskを渡す
#             onecar_sdf_pred = model(
#                 latent_code, onecar_xyz_input, level_mask=mask_ratio
#             )

#             # 1. Reconstruction Loss (SDF Loss)
#             # 元のコード通りサンプル数で正規化
#             loss_recon = criterion(onecar_sdf_pred, onecar_sdf) / num_sdf_samples

#             # Latent Regularization
#             loss_reg = cfg.latent_code_regularization * torch.sum(
#                 torch.norm(latent_code, dim=1)
#             )

#             # 2. --- [Eikonal Loss] ---
#             # 出力(SDF)の入力(XYZ)に対する勾配を計算
#             grads = autograd.grad(
#                 outputs=onecar_sdf_pred,
#                 inputs=onecar_xyz_input,
#                 grad_outputs=torch.ones_like(onecar_sdf_pred),
#                 create_graph=True,
#                 retain_graph=True,
#                 only_inputs=True,
#             )[0]

#             # 勾配ノルムが1になるように制約 ( (|∇f| - 1)^2 )
#             grad_norm = grads.norm(2, dim=-1)
#             loss_eikonal = ((grad_norm - 1.0) ** 2).mean()

#             # Eikonal項には重みをかける (batch_sizeやsample数での割り算のスケール感に注意)
#             # ここではシンプルに重み付け加算します（元Lossが小さい場合、LAMBDAを調整してください）
#             loss = loss_recon + loss_reg + (LAMBDA_EIKONAL * loss_eikonal)

#             loss.backward()

#             # モニタリング用 (JitterなしのMSEを見たい場合は別途推論が必要ですが、簡易的にこれでOK)
#             with torch.no_grad():
#                 monitor_loss = mse(onecar_sdf_pred, onecar_sdf)

#             batch_loss.append(loss.item())
#             batch_mse.append(monitor_loss.item())

#         optimizer.step()
#         scheduler.step()

#         total_loss.append(np.mean(batch_loss))
#         total_mse.append(np.mean(batch_mse))

#         tq.update()
#         tq.set_postfix(
#             {
#                 "epoch": epoch,
#                 "loss": f"{np.mean(total_loss):.5f}",
#                 "mse": f"{np.mean(total_mse):.5f}",
#                 "mask": f"{mask_ratio:.2f}",  # 現在のマスク率を表示
#                 "min_mse": f"{minimum_mse:.5f}",
#             }
#         )
#     tq.close()

#     # Model saving logic
#     current_mse = np.mean(total_mse)
#     if minimum_mse > current_mse:
#         minimum_mse = current_mse
#         os.makedirs("../outputs/models/", exist_ok=True)
#         os.makedirs("../outputs/latent-codes/", exist_ok=True)
#         torch.save(model.state_dict(), "../outputs/models/deepsdf.pth")
#         torch.save(latent_codes, "../outputs/latent-codes/latent_codes.pth")
#         print(f"Model saved at epoch {epoch} with MSE: {minimum_mse:.6f}")


import os
import pickle
import sys
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from experimet_config import Config
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

sys.path.append("../")
from src.datasets.deepsdf_dataset import DeepSDFDataset
from src.models.deepsdf import DeepSDF
from src.utils.metric import DeepSDFLoss, mse
from src.utils.utils import file_id_latent_code_connetction, set_seed

cfg = Config()
set_seed(42)

# loading training data path
points_path = glob("../../../input/02_sdf_dataset_by10/**/points.npy")
# points_path = points_path[:1]
latent_codes = nn.Parameter(
    torch.normal(
        0, 1e-4, size=(len(points_path), cfg.latent_code_dim)
    )  # (car_model_num, latent_code_dim)
)

model = DeepSDF(
    hidden_dim=cfg.hidden_dim,
    xyz_pos_enc_dim=cfg.xyz_pos_enc_dim,
    latent_code_dim=cfg.latent_code_dim,
    dropout_prob=cfg.dropout_prob,
).to(cfg.device)

latent_idx2carfolder, carfolder2latent_idx = file_id_latent_code_connetction(
    latent_codes, points_path
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
    num_workers=20,
)

criterion = DeepSDFLoss(delta=cfg.clamping_distance)
optimizer = torch.optim.Adam(
    [
        {
            "params": model.parameters(),
            "lr": cfg.deepsdf_initial_lr,
        },
        {
            "params": latent_codes,
            "lr": cfg.latent_code_inital_lr,
        },
    ]
)

warmup_steps = int(len(dataloader) * cfg.warmup_epoch)
num_training_steps = int(len(dataloader) * cfg.n_epochs)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps,
)

# save
with open("../outputs/data-split/latent_idx2carfolder.pickle", "wb") as f:
    pickle.dump(latent_idx2carfolder, f)

with open("../outputs/data-split/carfolder2latent_idx.pickle", "wb") as f:
    pickle.dump(carfolder2latent_idx, f)


# let's train
best_val_mse = np.inf
best_model = None
minimum_mse = np.inf

for epoch in range(cfg.n_epochs):
    model.train()
    total_loss = []
    total_mse = []
    tq = tqdm(dataloader)

    for data in dataloader:
        xyz = (
            data["points"].float().cuda().chunk(cfg.batch_size)
        )  # [[sample_per_car, 3], [sample_per_car, 3], ...]]
        sdf = (
            data["sdf"].float().cuda().chunk(cfg.batch_size)
        )  # [[sample_per_car, 1], [sample_per_car, 1], ...]]
        file_id = data["file_id"]  # [batch_car]
        num_sdf_samples = (
            cfg.sample_per_scene * cfg.batch_size
        )  # sample_per_car * batch_car

        optimizer.zero_grad()
        batch_loss = []
        batch_mse = []

        for i in range(cfg.batch_size):
            latent_code = (
                latent_codes[carfolder2latent_idx[file_id[i]], :]
                .to(cfg.device)
                .unsqueeze(0)
            )  # [1, latent_code_dim]
            onecar_xyz = xyz[i].squeeze(0).to(cfg.device)  # [sample_per_car, 3]
            onecar_sdf = sdf[i].squeeze(0).to(cfg.device)
            onecar_sdf_pred = model(latent_code, onecar_xyz)  # [sample_per_car, 1]

            loss = criterion(onecar_sdf_pred, onecar_sdf) / num_sdf_samples
            loss += cfg.latent_code_regularization * torch.sum(
                torch.norm(latent_code, dim=1)
            )

            loss.backward()

            monitor_loss = mse(onecar_sdf_pred, onecar_sdf)
            batch_loss.append(loss.item())
            batch_mse.append(monitor_loss)

        optimizer.step()
        scheduler.step()

        total_loss.append(np.mean(batch_loss))
        total_mse.append(np.mean(batch_mse))

        tq.update()
        tq.set_postfix(
            {
                "epoch": epoch,
                "loss": np.mean(total_loss),
                "mse": np.mean(total_mse),
                "minimum_mse": minimum_mse,
            }
        )
    tq.close()

    if minimum_mse > np.mean(total_mse):
        minimum_mse = np.mean(total_mse)
        torch.save(model.state_dict(), "../outputs/models/deepsdf.pth")
        torch.save(latent_codes, "../outputs/latent-codes/latent_codes.pth")
