from ultralytics import YOLO, RTDETR
from pathlib import Path

script_dir = Path(__file__).parent

yolo_path = script_dir / "YOLOv8nano" / "weights" / "my_model.pt"
rtdetr_path = script_dir / "RT-DETR" / "yolo26n"
yaml_path = script_dir / "RT-DETR" / "dataset.yaml"

print("loading in models to be compared")
yolo = YOLO(str(yolo_path))
rtdetr = RTDETR(str(rtdetr_path))

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


#construct tables
print("\n" + "="*50)
print(f"{'METRIC':<20} | {'YOLOv8 Nano':<12} | {'RT-DETR':<12}")
print(f"{'mAP 50':<20} | {yoloMap50:<12.4f} | {rtdetrMap50:<12.4f}")
print(f"{'mAP 50-95':<20} | {yoloMap5095:<12.4f} | {rtdetrMap5095:<12.4f}")
print(f"{'inference Speed':<20} | {yoloInferSpeed:<12.4f} | {rtdetrInferSpeed:<12.4f}")
