import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob

# === 建立資料夾 ===
def make_dirs(base_path="dataset"):
    os.makedirs(base_path, exist_ok=True)
    img_path = os.path.join(base_path, "images")
    for cls in ["random", "edge", "center", "scratch"]:
        os.makedirs(os.path.join(img_path, cls), exist_ok=True)
    return img_path

# === 產生模擬晶圓 (簡單四種模式) ===
def make_wafer(size=26, base_fail=0.1, pattern="random", strength=0.7, seed=None):
    if seed is not None:
        np.random.seed(seed)

    wafer = np.random.choice([0, 1], size=(size, size), p=[base_fail, 1 - base_fail]).astype(float)

    Y, X = np.ogrid[:size, :size]
    center = size / 2
    dist = np.sqrt((X - center) ** 2 + (Y - center) ** 2)
    mask = dist <= size / 2

    wafer[~mask] = np.nan  # 圓形遮罩

    if pattern == "edge":
        wafer[mask & (dist > size * 0.4)] = np.random.choice(
            [0, 1], p=[strength, 1 - strength], size=np.count_nonzero(mask & (dist > size * 0.4))
        )
    elif pattern == "center":
        wafer[mask & (dist < size * 0.3)] = np.random.choice(
            [0, 1], p=[strength, 1 - strength], size=np.count_nonzero(mask & (dist < size * 0.3))
        )
    elif pattern == "scratch":
        line = np.random.randint(0, size)
        wafer[:, line:line+2] = 0  # 模擬刮痕

    return wafer

# === 畫圖並存檔 ===
def save_wafer(wafer, filepath, title="wafer"):
    plt.imshow(wafer, cmap="coolwarm", interpolation="nearest")
    plt.axis("off")
    plt.title(title)
    plt.savefig(filepath, bbox_inches="tight", dpi=100)
    plt.close()

# === 主程式：產生多片 wafer dataset ===
def generate_dataset(num_per_class=1000, out_dir="dataset"):
    img_dir = make_dirs(out_dir)
    labels = []

    patterns = ["random", "edge", "center", "scratch"]

    for pattern in patterns:
        for i in range(num_per_class):
            w = make_wafer(size=26, base_fail=0.1, pattern=pattern, seed=i)
            filename = f"{pattern}_{i}.png"
            filepath = os.path.join(img_dir, pattern, filename)
            save_wafer(w, filepath, title=pattern)
            labels.append({"file": filename, "label": pattern})

    # 存 CSV 標籤檔
    df = pd.DataFrame(labels)
    df.to_csv(os.path.join(out_dir, "labels.csv"), index=False)

    # 計算總數
    total_images = len(glob.glob(os.path.join(out_dir, "images", "*", "*.png")))
    print(f"✅ 資料集產生完成，共 {total_images} 張 wafer 圖片")

# === 執行 ===
if __name__ == "__main__":
    generate_dataset(num_per_class=1000)  # 會生成 4000 張
