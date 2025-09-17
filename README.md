# 🟢 晶圓缺陷檢測模擬專案

## 📌 專案簡介
本專案模擬半導體製程中常見的晶圓缺陷（`Edge`, `Center`, `Scratch`, `Random`），  
並建立 **傳統機器學習模型 (SVM)** 與 **深度學習模型 (CNN)** 進行分類與檢測，最後透過難度較高的挑戰圖片觀察模型表現。

---

## 🎯 專案目標
- ✅ 自動生成模擬晶圓缺陷資料集  
- ✅ 訓練 SVM 與 CNN 兩種模型進行分類  
- ✅ 比較傳統機器學習與深度學習的效果  
- ✅ 測試模型在「高難度缺陷圖」上的信心度表現  

---

## 🗂️ 資料集
- 總數：**4000 張 wafer 圖片**（四類缺陷，各 1000 張）  
- 影像大小：**26x26**，灰階圖  
- `labels.csv`：包含圖片對應的標籤  

### 資料集結構
dataset/
├── images/
│ ├── center/
│ ├── edge/
│ ├── random/
│ ├── scratch/
└── labels.csv

---

## ⚙️ 方法流程

```mermaid
flowchart TD
    A[資料生成 wafer_map_demo.py] --> B[SVM 訓練 train_svm.py]
    A --> C[CNN 訓練 train_cnn.py]
    B --> D[模型比較與評估]
    C --> D
    D --> E[單張圖片測試 predict.py]
    E --> F[挑戰圖片 generate_hard_case.py]
