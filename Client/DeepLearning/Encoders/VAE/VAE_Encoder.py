import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os

class CustomVAEEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(CustomVAEEncoder, self).__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, features_dim)

        # 1. 获取当前 .py 文件的绝对路径
        current_file_path = os.path.abspath(__file__)
        # 2. 获取该文件所在的目录
        current_dir = os.path.dirname(current_file_path)
        # 3. 拼接得到权重文件的绝对路径
        file_path = os.path.join(current_dir, f"vae_encoder_mu_{features_dim}.pth")

        # 修改后的读取代码
        vae_state_dict = torch.load(file_path)

        self.encoder.load_state_dict(vae_state_dict['encoder'])
        self.fc_mu.load_state_dict(vae_state_dict['fc_mu'])

        #  冻结参数
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.fc_mu.parameters():
            param.requires_grad = False

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hidden = self.encoder(observations)
            mu = self.fc_mu(hidden)
        return mu

print(" Loaded VAE encoder + mu from absolute path.")
