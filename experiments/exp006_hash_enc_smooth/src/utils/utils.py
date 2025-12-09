import os
import random

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def file_id_latent_code_connetction(latent_codes, car_points_path):
    assert len(latent_codes) == len(
        car_points_path
    ), "latent_codes and car_points_path must have same length."

    latent_idx2carfolder = {}
    carfolder2latent_idx = {}

    for i in range(len(latent_codes)):
        car_foldername = car_points_path[i].split("/")[-2]
        latent_idx2carfolder[i] = car_points_path[i]
        carfolder2latent_idx[car_foldername] = i

    return latent_idx2carfolder, carfolder2latent_idx
