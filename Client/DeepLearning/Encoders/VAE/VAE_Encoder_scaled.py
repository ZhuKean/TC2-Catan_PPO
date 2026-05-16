import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os

class CustomVAEEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(CustomVAEEncoder, self).__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]

        # 保存训练时的归一化常量
        self.obs_max = 7.288927694521257

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
        file_path = os.path.join(current_dir, f"vae_encoder_mu_{features_dim}_scaled.pth")

        # 建议增加 map_location，防止保存的是 GPU 权重但在 CPU 环境运行报错
        vae_state_dict = torch.load(file_path, map_location=torch.device('cpu'))

        self.encoder.load_state_dict(vae_state_dict['encoder'])
        self.fc_mu.load_state_dict(vae_state_dict['fc_mu'])

        # 冻结参数：确保 RL 训练过程中不对 VAE 进行反向传播
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 必须使用 torch 的算子以支持 Tensor 运算
        with torch.no_grad():
            # 1. 处理负数 (对应 np.maximum(obs, 0))
            # 注意：在 Catan 环境中如果出现 -1 (贼) 或 -2，强制转为 0
            x = torch.clamp(observations, min=0.0)

            # 2. 对数缩放 (对应 np.log1p)
            # log(1+x) 压缩长尾分布
            x = torch.log1p(x)

            # 3. 归一化到 [0, 1] (使用训练时记录的 Max)
            # 加上 1e-8 防止极罕见情况下的除零错误
            x = x / (self.obs_max + 1e-8)

            # 4. 喂入已经训练好的 VAE 提取特征
            hidden = self.encoder(x)
            mu = self.fc_mu(hidden)

        return mu


print(f"Loaded VAE encoder with fixed obs_max: {7.288927694521257}")