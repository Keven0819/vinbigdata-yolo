from ultralytics import YOLO


def main():
    # 載入預訓練模型 (可選 yolo11s.pt、yolo11n.pt 或 yolov8s.pt)
    model = YOLO("yolo11s.pt")

    # 開始訓練
    model.train(
        data="vinbigdata.yaml",
        epochs=30,  # 學校作業通常 30~50 epoch 即可觀察到不錯的收斂趨勢
        imgsz=512,  # 與輕量化資料集原生大小匹配
        batch=16,  # 8GB VRAM 建議設 16，若出現 CUDA Out of Memory 請降至 8
        device=0,  # 指定使用 GPU (0)
        workers=4,  # 資料載入核心數
        amp=True,  # 自動啟用混合精度 FP16，降低顯存並加速訓練
        project="runs/train",
        name="vinbigdata_exp",
    )


if __name__ == "__main__":
    main()
