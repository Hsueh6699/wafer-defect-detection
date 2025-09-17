# type: ignore
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt   # ← 這行要在最前面
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# === 修正中文字體 ===
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# === 讀取標籤檔 ===
labels = pd.read_csv("dataset/labels.csv")
class_names = labels["label"].unique()
class_to_idx = {name: idx for idx, name in enumerate(class_names)}

# === 調整成固定大小 26x26 ===
def load_image(path):
    img = plt.imread(path)
    if img.ndim == 3:  # 如果是彩色圖 → 轉灰階
        img = img.mean(axis=2)
    # resize 到 26x26
    img_resized = tf.image.resize(img[..., np.newaxis], (26, 26)).numpy().squeeze()
    return img_resized

X = []
y = []

for i, row in labels.iterrows():
    img_path = os.path.join("dataset", "images", row["label"], row["file"])
    img = load_image(img_path)
    X.append(img)
    y.append(class_to_idx[row["label"]])

X = np.array(X)
y = np.array(y)

# === 調整形狀 (samples, 26, 26, 1) ===
X = X.reshape(-1, 26, 26, 1).astype("float32")
y = to_categorical(y, num_classes=len(class_names))

print("資料形狀:", X.shape, y.shape)

# === 分割訓練/測試集 ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 建立 CNN 模型 ===
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(26, 26, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === 訓練模型 ===
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_data=(X_test, y_test))

# === 評估模型 ===
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\n測試集準確率: {test_acc:.2f}")

# === 儲存模型 ===
model.save("wafer_cnn_model.h5")
print("模型已儲存為 wafer_cnn_model.h5")

# === 畫出訓練過程 ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='訓練準確率')
plt.plot(history.history['val_accuracy'], label='驗證準確率')
plt.legend()
plt.title("準確率")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='訓練損失')
plt.plot(history.history['val_loss'], label='驗證損失')
plt.legend()
plt.title("損失")

plt.show()
