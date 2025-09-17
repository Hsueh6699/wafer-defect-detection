import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === 修正中文字體 ===
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 微軟正黑體
plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號

# === 讀取標籤檔 ===
labels = pd.read_csv("dataset/labels.csv")

# === 讀取圖片資料 ===
def load_image(path):
    img = plt.imread(path)
    img_gray = img.mean(axis=2) if img.ndim == 3 else img
    return img_gray.flatten()

X = []
y = []

for i, row in labels.iterrows():
    img_path = os.path.join("dataset", "images", row["label"], row["file"])
    X.append(load_image(img_path))
    y.append(row["label"])

X = np.array(X)
y = np.array(y)

print("資料形狀:", X.shape, y.shape)

# === 分割訓練/測試集 ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 建立並訓練 SVM 模型 ===
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# === 預測 ===
y_pred = model.predict(X_test)

# === 評估 ===
print("\n分類報告:")
print(classification_report(y_test, y_pred))

# 混淆矩陣
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("\n混淆矩陣:")
print(cm)

# === 畫出混淆矩陣圖 ===
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel("預測類別")
plt.ylabel("真實類別")
plt.title("SVM 混淆矩陣")
plt.show()