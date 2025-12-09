import torch


class Config:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # training deepsdf
        self.train_test_ratio = 0.8
        self.hidden_dim = 512
        self.latent_code_dim = 128
        self.xyz_dim = 3
        self.xyz_pos_enc_dim = 3
        self.dropout_prob = 0.001
        self.sample_per_scene = 10000
        self.batch_size = 5
        self.clamping_distance = 1.0
        self.latent_code_regularization = 1e-4
        self.n_epochs = 4002
        self.deepsdf_initial_lr = 1e-3
        self.latent_code_inital_lr = 1e-4
        self.warmup_epoch = 3

        # output training 3d model
        self.train_latent_output_resolution = 256
