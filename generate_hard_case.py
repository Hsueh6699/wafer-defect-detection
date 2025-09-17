import os
import numpy as np
import matplotlib.pyplot as plt

# 確保資料夾存在
os.makedirs("dataset/images/test", exist_ok=True)

# === 生成一個 wafer 基底 (圓形) ===
size = 26
Y, X = np.ogrid[:size, :size]
center = size / 2
dist = np.sqrt((X - center) ** 2 + (Y - center) ** 2)
wafer = (dist < size / 2).astype(int)  # 圓形 wafer

# === 加入隨機雜訊 (模擬 defect) ===
noise = np.random.choice([0, 1], (size, size), p=[0.9, 0.1])  # 10% 隨機黑點
wafer_with_noise = wafer.copy()
wafer_with_noise[noise == 1] = 0  # 在 wafer 上挖洞

# === 加入一條刮痕 (模擬 scratch) ===
for i in range(size):
    if 8 < i < 18:  # 刮痕範圍
        wafer_with_noise[i, i-2:i+2] = 0

# === 加入中心 defect (模擬 center fail) ===
wafer_with_noise[10:16, 10:16] = 0

# === 儲存圖片 ===
plt.imsave("dataset/images/test/hard_case.png", wafer_with_noise, cmap="gray")

print("✅ 已產生難度較高的測試圖片：dataset/images/test/hard_case.png")
