import numpy as np
import matplotlib.pyplot as plt

# ========== 1. 加载数据 ========== #
obs_path = "./vae_dataset_combined.npy"
try:
    observations = np.load(obs_path)
    print(f"成功加载数据，形状为: {observations.shape}")
except FileNotFoundError:
    print("错误：未找到数据文件，请检查路径。")
    exit()

# ========== 2. 基础统计信息 ========== #
obs_min = np.min(observations)
obs_max = np.max(observations)
obs_mean = np.mean(observations)
obs_std = np.std(observations)
zeros_ratio = np.sum(observations == 0) / observations.size

print("--- 全局统计 ---")
print(f"最小值 (Min): {obs_min}")
print(f"最大值 (Max): {obs_max}")
print(f"平均值 (Mean): {obs_mean:.4f}")
print(f"标准差 (Std): {obs_std:.4f}")
print(f"零值占比 (Sparsity): {zeros_ratio:.2%}")

# ========== 3. 维度分析 ========== #
# 检查有多少维度的最大值超过了 1.0（这是导致 Sigmoid 崩溃的关键）
dims_over_one = np.sum(np.max(observations, axis=0) > 1.0)
print(f"数值大于 1.0 的维度数量: {dims_over_one} / {observations.shape[1]}")

# ========== 4. 可视化分布 ========== #
plt.figure(figsize=(12, 5))

# 子图 1: 全局数值分布直方图
plt.subplot(1, 2, 1)
plt.hist(observations.flatten(), bins=50, color='skyblue', edgecolor='black')
plt.title("Overall Value Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.yscale('log') # 使用对数坐标，因为零值通常极多

# 子图 2: 每一维度的最大值分布
plt.subplot(1, 2, 2)
plt.plot(np.max(observations, axis=0), '.', markersize=2, alpha=0.5)
plt.axhline(y=1.0, color='r', linestyle='--', label='Sigmoid Limit (1.0)')
plt.title("Max Value per Dimension")
plt.xlabel("Dimension Index")
plt.ylabel("Max Value")
plt.legend()

plt.tight_layout()
plt.show()

# ========== 5. 抽样检查具体数值 ========== #
print("\n--- 前 10 行的部分原始数据示例 ---")
# 打印前 5 行，前 20 列
print(observations[:5, :20])