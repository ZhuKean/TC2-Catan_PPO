import os
import sys
# 1. 【核心修复】将项目根目录加入 sys.path
# 假设你的结构是 .../Client/DeepLearning/Encoders/VAE/CollectGameStatus.py
# 我们需要把 .../Client/ 加入路径，这样 Python 才能识别 'DeepLearning'
current_file_path = os.path.dirname(os.path.abspath(__file__))
# 向上跳三级到达 Client 目录
project_root = os.path.abspath(os.path.join(current_file_path, "..", "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from tqdm import tqdm

from DeepLearning.Environments.DummyEnv import DummyEnv
from DeepLearning.Environments.SelfPlayTrading import SelfPlayTrading
from Agents.AgentRandom2 import AgentRandom2

# === 参数配置 ===
NUM_EPISODES = 500         # 每个环境采集多少局
MAX_STEPS = 200            # 每局最多多少步
OBS_DIM = 1233

# === 用于存储两个环境的 observation ===
all_observations = []

def collect_data(env, env_name):
    local_obs = []
    for ep in tqdm(range(NUM_EPISODES), desc=f"Collecting from {env_name}"):
        obs, _ = env.reset()
        # 打印每一局刚开始时的回合数特征（假设它在 observation 的最后一位）
        print(f"Episode {ep} started. Initial Turn Count: {obs[-1]}")
        for _ in range(MAX_STEPS):
            local_obs.append(obs)
            # 随机选一个合法动作
            valid_actions = [i for i, mask in enumerate(env.action_mask) if mask]
            action = np.random.choice(valid_actions)
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break
    return local_obs


# === Dummy 环境 ===
env_dummy = DummyEnv()
obs_dummy = collect_data(env_dummy, "dummy")
all_observations.extend(obs_dummy)


# === SelfPlayTrading 环境 ===
os.environ["UPDATE_MODELS_DIST"] = "False"
env_selfplay = SelfPlayTrading()
#env_selfplay.opponentModel0 = AgentRandom2("P1", 1)
obs_selfplay = collect_data(env_selfplay, "selfplay")
all_observations.extend(obs_selfplay)

# === 保存为一个合并文件 ===
all_observations = np.array(all_observations)
save_path = "./vae_dataset_combined.npy"
np.save(save_path, all_observations)
print(f" Saved combined dataset to {save_path}, shape: {all_observations.shape}")
