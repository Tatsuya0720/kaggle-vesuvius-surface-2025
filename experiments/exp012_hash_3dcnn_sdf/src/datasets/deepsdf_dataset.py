import os
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset


# 負の距離の座標と正の距離の座標をそれぞれsubsample個ずつランダムに取得
def unpack_sdf_samples(points, sdf, subsample=16000):
    points = torch.tensor(points).reshape(-1, 3)
    sdf = torch.tensor(sdf).reshape(-1, 1)
    samples = torch.cat([points, sdf], -1).reshape(-1, 4)

    pos_tensor = samples[samples[:, 3] > 0, :]
    neg_tensor = samples[samples[:, 3] < 0, :]

    half = int(subsample / 2)

    pos_idx = np.arange(0, len(pos_tensor))
    neg_idx = np.arange(0, len(neg_tensor))
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    if len(pos_tensor) < half:
        pos_idx = (
            pos_idx.tolist()
            + np.random.choice(pos_idx, half - len(pos_tensor)).tolist()
        )
    if len(neg_tensor) < half:
        neg_idx = (
            neg_idx.tolist()
            + np.random.choice(neg_idx, half - len(neg_tensor)).tolist()
        )

    sample_pos = pos_tensor[pos_idx[:half], :]
    sample_neg = neg_tensor[neg_idx[:half], :]

    samples = torch.cat([sample_pos, sample_neg], 0)

    xyz = samples[:, :3]
    sdf = samples[:, 3].reshape(-1, 1)

    return xyz, sdf


class DeepSDFDataset(Dataset):
    def __init__(self, points_path_list, name2latent_idx, subsample=16000):
        super().__init__()
        self.subsample = subsample
        self.name2latent_idx = name2latent_idx

        self.idx2training_data = {}

        # read points.npy and sdf.npy
        for i, path in enumerate(points_path_list):
            assert os.path.exists(path), f"{path} does not exist."

            file_id = path.split("/")[-2]
            points_path = deepcopy(path)
            sdf_path = deepcopy(path).replace("points.npy", "sdf.npy")
            ct_path = deepcopy(path).replace("points.npy", "ct320.npy")

            points = np.load(points_path)
            sdf = np.load(sdf_path)
            ct = np.load(ct_path)

            self.idx2training_data[i] = {
                "file_id": file_id,
                "ct": ct,
                "points": points,
                "sdf": sdf,
            }

    def __len__(self):
        return len(self.idx2training_data)

    def __getitem__(self, idx):
        file_id = self.idx2training_data[idx]["file_id"]
        ct = self.idx2training_data[idx]["ct"]
        points = self.idx2training_data[idx]["points"]
        sdf = self.idx2training_data[idx]["sdf"]

        # --- 【修正点】ここで正規化を行います ---
        # 0~255 (uint8等) -> 0.0~1.0 (float32) に変換
        # メモリ効率のため、データを取り出すこのタイミングで行うのが定石です

        ct = ct.astype(np.float32) / 255.0
        # ct = ct.astype(np.float32) / 255.0

        points, sdf = unpack_sdf_samples(points, sdf, subsample=self.subsample)

        return {
            "file_id": file_id,
            "ct": ct,
            "points": points,
            "sdf": sdf,
        }
