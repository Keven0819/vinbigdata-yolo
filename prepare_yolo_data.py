import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

RAW_DIR = "raw_data/vinbigdata"
CSV_PATH = os.path.join(RAW_DIR, "train.csv")
IMAGES_SRC = os.path.join(RAW_DIR, "train")
OUTPUT_DIR = "datasets/vinbigdata"

os.makedirs(f"{OUTPUT_DIR}/images/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/images/val", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels/val", exist_ok=True)

print("讀取 train.csv...")
df = pd.read_csv(CSV_PATH)

# 取出所有影像 ID 並劃分 80% Train, 20% Val
# (為避免訓練耗時過長，若僅為作業練習，可先抽樣例如 sample_ids = df['image_id'].unique()[:4000])
unique_ids = df["image_id"].unique()
train_ids, val_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)

split_dict = {img_id: "train" for img_id in train_ids}
split_dict.update({img_id: "val" for img_id in val_ids})

# 依影像分組處理
grouped = df.groupby("image_id")

print("正在生成 YOLO 格式檔案與切分資料...")
for img_id, group in tqdm(grouped):
    split = split_dict.get(img_id)
    if not split:
        continue

    # 1. 複製圖片
    img_name = f"{img_id}.png"
    src_img = os.path.join(IMAGES_SRC, img_name)
    dst_img = os.path.join(OUTPUT_DIR, "images", split, img_name)

    if os.path.exists(src_img):
        shutil.copy(src_img, dst_img)

    # 2. 生成對應的 label txt 檔
    label_path = os.path.join(OUTPUT_DIR, "labels", split, f"{img_id}.txt")
    with open(label_path, "w") as f:
        for _, row in group.iterrows():
            cid = int(row["class_id"])
            # Class 14 是 "No finding"，代表沒有異常，txt 留空作為背景負樣本
            if cid == 14 or pd.isna(row["x_min"]):
                continue

            # 原 CSV 包含原始寬高，進行 0~1 正規化
            orig_w, orig_h = row["width"], row["height"]
            x_min, y_min = row["x_min"], row["y_min"]
            x_max, y_max = row["x_max"], row["y_max"]

            x_center = ((x_min + x_max) / 2.0) / orig_w
            y_center = ((y_min + y_max) / 2.0) / orig_h
            box_w = (x_max - x_min) / orig_w
            box_h = (y_max - y_min) / orig_h

            # 防止座標些微超出 [0, 1] 範圍
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            box_w = max(0.0, min(1.0, box_w))
            box_h = max(0.0, min(1.0, box_h))

            f.write(f"{cid} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

print("資料集轉換完成！目錄：datasets/vinbigdata")
