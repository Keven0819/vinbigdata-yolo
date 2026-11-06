# VinBigData 胸部 X 光病灶偵測 (YOLO 復刻專案)

這是一個復刻 Kaggle 競賽 [VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection) 的物件偵測專案。

原競賽提供的醫療影像格式為 DICOM，原始檔案大小超過 140 GB，在本機端讀取與訓練成本太高。為了在個人筆電環境完成模型訓練與驗證，本專案改用社群預處理好的 **512x512 PNG 輕量化資料集**（約 2.5 GB），並採用 Ultralytics YOLO 進行 14 種胸腔異常病灶的定位與分類。

## 硬體與開發環境

* **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)
* **OS**: Windows 11 (PowerShell)
* **Python**: 3.10+ (使用 `.venv` 虛擬環境管理)
* **Framework**: PyTorch (CUDA 12.1) + Ultralytics YOLO11

## 專案目錄結構

```text
vinbigdata-yolo/
├── .venv/                   # Python 虛擬環境
├── raw_data/                # 下載解壓後的原始輕量化資料集
├── datasets/                # 轉換後的 YOLO 格式資料集 (train / val)
├── runs/                    # 訓練與推論輸出的權重及圖表紀錄
├── prepare_yolo_data.py     # 資料清理、座標正規化與 Train/Val 切分腳本
├── vinbigdata.yaml          # YOLO 資料集與類別設定檔
├── train.py                 # 模型訓練腳本
├── predict.py               # 驗證集推論與視覺化腳本
└── README.md
```

## 快速執行流程

### 1. 建立環境與安裝套件
```powershell
python -m venv .venv
.venv\Scripts\activate

# 安裝支援 CUDA 的 PyTorch 與相關依賴
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics kaggle pandas opencv-python tqdm scikit-learn
```

### 2. 下載資料集與格式轉換
設定好個人的 Kaggle API Token (`~/.kaggle/kaggle.json`) 後執行：
```powershell
kaggle datasets download -d awsaf49/vinbigdata-512-image-dataset
mkdir raw_data
tar -xf vinbigdata-512-image-dataset.zip -C raw_data

# 轉為 YOLO 標註格式並切分資料集 (80% Train / 20% Val)
python prepare_yolo_data.py
```

### 3. 執行訓練
針對 8GB VRAM 配置，`batch` 設為 16 並開啟混合精度訓練 (`amp=True`)：
```powershell
python train.py
```

### 4. 模型推論測試
自動抓取 `runs/` 目錄下最新的 `best.pt` 對驗證集影像進行病灶框選：
```powershell
python predict.py
```

## 備註與資料集處理細節

* **無病灶樣本處理 (`Class 14: No finding`)**：原始標註中 `class_id = 14` 代表正常無異常的胸片。在轉換腳本中，這些圖片會生成空白的 `.txt` 標註檔作為負樣本（Background images）放進訓練集，用來降低模型的誤報率（False Positive）。
* **重疊標註框問題**：資料集由多位放射科醫師共同標註，同一病灶常有多個重疊的 Bounding Box。目前版本直接將所有標註納入訓練，後續若要進一步提升 mAP，可在前處理階段加入 WBF (Weighted Boxes Fusion) 進行框融合。
