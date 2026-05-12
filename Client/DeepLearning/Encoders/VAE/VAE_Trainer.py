import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ========== GPU 配置 ========== #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========== 参数配置 ========== #
LATENT_DIM = 64

# ========== Step 1: 加载观察数据 ========== #
obs_path = "./vae_dataset_combined.npy"
assert os.path.exists(obs_path), "请先运行采集脚本并生成 observations.npy"
observations = np.load(obs_path)

# 1. 消除负数带来的影响 (针对你的 -2)
observations = np.maximum(observations, 0)

# 2. 对数缩放：解决 image_821ffb.png 中的长尾问题
# 使用 log(1+x) 确保 0 映射到 0，且大数被极度压缩
obs_log = np.log1p(observations)

# 3. 归一化到 [0, 1]：适配 VAE 最后一层 Sigmoid
# 这样 1463 会变成 1.0，0 还是 0，中间数值按对数曲线分布
obs_max = np.max(obs_log)
observations = obs_log / (obs_max + 1e-8)

print(f"数据处理完毕：当前 Max={observations.max()}, Min={observations.min()}")
observations = torch.tensor(observations, dtype=torch.float32)


# ========== Step 2: 构造 DataLoader ========== #
dataset = TensorDataset(observations)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# ========== Step 3: 定义 VAE 模型 ========== #
class VAE(nn.Module):
    def __init__(self, input_dim=1233, latent_dim=128):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(device)  # 确保随机噪声在同一设备
        return mu + eps * std

    def forward(self, x):
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


# ========== Step 4: 定义损失函数 ========== #
def loss_function(x, x_recon, mu, logvar):
    # 重建损失保持 sum 模式，确保 1233 维特征被重视
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum') / x.shape[0]

    # KL 散度项
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]

    # 改进点：引入更小的 beta (例如 0.1 或 0.01)
    # 这样可以防止 KL 散度过快地把 mu 推向 0，从而保留更多的局势差异特征
    beta = 0.1

    return recon_loss + beta * kl_div


# ========== Step 5: 初始化模型并移动到 GPU ========== #
vae = VAE(latent_dim=LATENT_DIM).to(device)  # 将模型搬运到 GPU
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# ========== Step 6: 开始训练 ========== #
EPOCHS = 30
for epoch in range(EPOCHS):
    vae.train()
    total_loss = 0
    for batch in dataloader:
        x = batch[0].to(device)  # 将每一批次的数据搬运到 GPU

        optimizer.zero_grad()
        x_recon, mu, logvar = vae(x)
        loss = loss_function(x, x_recon, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    # ===== 输出 mu 信息 =====
    print(f"Mu max: {mu.max().item():.4f}, Mu min: {mu.min().item():.4f}")
    print(f"Mu mean: {mu.mean().item():.4f}, Mu std: {mu.std().item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Avg Loss: {avg_loss:.4f}")

# ========== Step 7: 保存模型 ========== #
# 保存时通常将权重移回 CPU，方便在不同环境下加载
save_dict = {
    'encoder': vae.encoder.state_dict(),
    'fc_mu': vae.fc_mu.state_dict()
}
save_path = f"vae_encoder_mu_{LATENT_DIM}.pth"
torch.save(save_dict, save_path)
print(f"Encoder + fc_mu saved to {save_path}")