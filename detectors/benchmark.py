import torch
from ultralytics import YOLO, RTDETR
from pathlib import Path

script_dir = Path(__file__).parent

yolo_path = script_dir / "YOLOv8nano" / "weights" / "pong_YOLObest.pt"
rtdetr_path = script_dir / "RT-DETR" / "yolo26n"
yaml_path = script_dir / "RT-DETR" / "dataset.yaml"

print("loading in models to be compared")
yolo = YOLO(str(yolo_path))
rtdetr = RTDETR(str(rtdetr_path))

if torch.cuda.is_available():
    print("warming up GPU")
    yolo.model.to("cuda")
    rtdetr.model.to("cuda")

    dummy_input = torch.zeros((1, 3, 640, 640), device="cuda")

    for _ in range(20):
        _ = yolo.model(dummy_input)
        _ = rtdetr.model(dummy_input)

print("running yolo val")
yolo_metrics = yolo.val(data = str(yaml_path), verbose = False)

print("running rtdetr val")
rtdetr_metrics = rtdetr.val(data = str(yaml_path), verbose = False)

yoloMap50 = yolo_metrics.box.map50
rtdetrMap50 = rtdetr_metrics.box.map50

yoloMap5095 = yolo_metrics.box.map
rtdetrMap5095 = rtdetr_metrics.box.map

yoloInferSpeed = yolo_metrics.speed['inference']
rtdetrInferSpeed = rtdetr_metrics.speed['inference']

#enforce FPS rule in report
yolo_fps = 1000.0 / yoloInferSpeed if yoloInferSpeed > 0 else 0.0
rtdetr_fps = 1000.0 / rtdetrInferSpeed if rtdetrInferSpeed > 0 else 0.0

yolo_budget_pass = "PASS" if yoloInferSpeed <= 16.66 else "FAIL (TOO SLOW)"
rtdetr_budget_pass = "PASS" if rtdetrInferSpeed <= 16.66 else "FAIL (TOO SLOW)"


#construct tables
print("\n" + "="*50)
print(f"{'METRIC':<20} | {'YOLOv8 Nano':<12} | {'RT-DETR':<12}")
print(f"{'mAP 50':<20} | {yoloMap50:<12.4f} | {rtdetrMap50:<12.4f}")
print(f"{'mAP 50-95':<20} | {yoloMap5095:<12.4f} | {rtdetrMap5095:<12.4f}")
print(f"{'inference Speed':<20} | {yoloInferSpeed:<12.4f} | {rtdetrInferSpeed:<12.4f}")
print(f"{'Derived Throughput':<25} | {f'{yolo_fps:.2f} FPS':<15} | {f'{rtdetr_fps:.2f} FPS':<15}")
print(f"{'16.66ms RL Budget':<25} | {yolo_budget_pass:<15} | {rtdetr_budget_pass:<15}")
print("="*75)
