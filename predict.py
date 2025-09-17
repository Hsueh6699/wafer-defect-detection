# pyright: reportMissingImports=false
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# === 修正中文字體 ===
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 或 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 載入模型
model = load_model("wafer_cnn_model.h5")

# 載入 class 名稱
class_names = ["center", "edge", "random", "scratch"]

# 圖片預處理
def load_and_preprocess_image(path):
    img = plt.imread(path)
    if img.ndim == 3:
        img = img.mean(axis=2)  # 灰階
    img = tf.image.resize(img[..., np.newaxis], (26, 26)).numpy()
    img = img.astype("float32") / 255.0
    return img.reshape(1, 26, 26, 1)

# 讓你可以更換測試圖片
img_path = "C:\\Users\\客服專員\\Desktop\\aiboy\\dataset\\images\\test\\hard_case.png"
img = load_and_preprocess_image(img_path)

# 預測
pred = model.predict(img)
pred_class = np.argmax(pred, axis=1)[0]
confidence = np.max(pred)

print(f"模型預測：{class_names[pred_class]} (信心度 {confidence:.2f})")

# 顯示圖片與結果
plt.imshow(img.squeeze(), cmap="gray")
plt.title(f"預測：{class_names[pred_class]} ({confidence:.2f})")
plt.axis("off")
plt.show()
