from ultralytics import RTDETR #do pip install ultralytics
from pathlib import Path
import torch

dataset_yaml = "dataset.yaml" #prepared dataset after running prepare_dataset.py
base_model = "rtdetr-x.pt"  #rt-detr model
save_dir = "runs/rtdetr_pong" #save training output
run = "rtdetr-x_v1"

if __name__ == '__main__':
    assert Path(dataset_yaml).exists(), \
        "dataset.yaml not found"

    print(f"device: ", 0 if torch.cuda.is_available() else "cpu")
    print(f'rt-detr model: ', base_model)

    model = RTDETR(base_model)

    results = model.train(
        data = dataset_yaml, epochs = 100, batch = 8, imgsz = 640, workers = 0, lr0 = 1e-4, lrf = 0.01, warmup_epochs = 5, patience = 30,
        device = 0 if torch.cuda.is_available() else "cpu", project = save_dir, name = run, exist_ok = True, hsv_h = 0.0, hsv_s = 0.0,   # no change to hue/saturation/brightness
        hsv_v = 0.0, fliplr= 0.0, flipud= 0.0, mosaic= 0.0, mixup = 0.0, degrees = 0.0, translate = 0.1, scale = 0.3,
        plots = True, save = True, save_period = 10, verbose = True,
    )

    print("\n" + "="*60)
    print(" TRAINING COMPLETE")
    print("="*60)
    best_ckpt = Path(save_dir) / run / "weights" / "best.pt"
    print(f"Best checkpoint: {best_ckpt}")
    print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    print("="*60)
    print("\n Training done, run inference")