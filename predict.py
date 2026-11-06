import glob
import os

from ultralytics import YOLO

# 1. 自動在 runs 資料夾下尋找所有的 best.pt
weight_files = glob.glob("runs/**/best.pt", recursive=True)

if not weight_files:
    raise FileNotFoundError(
        "找不到任何 best.pt！請確認 `python train.py` 是否有成功跑完至少 1 個 epoch。"
    )

# 取出最新產生的一個 best.pt
latest_weight = max(weight_files, key=os.path.getctime)
print(f"成功找到並載入模型權重: {latest_weight}")

model = YOLO(latest_weight)

# 2. 從驗證集挑選前 3 張圖片做推論
sample_images = glob.glob("datasets/vinbigdata/images/val/*.png")[:3]

if not sample_images:
    raise FileNotFoundError(
        "在 datasets/vinbigdata/images/val/ 找不到圖片，請確認資料轉換步驟是否成功。"
    )

# 3. 進行推論並存檔
results = model.predict(
    source=sample_images,
    conf=0.25,  # 信心度門檻值
    save=True,  # 儲存繪製標註框的影像
    project="runs/predict",
    name="val_result",
)

print(f"推論完成！請前往查看預測結果圖片：{results[0].save_dir}")
