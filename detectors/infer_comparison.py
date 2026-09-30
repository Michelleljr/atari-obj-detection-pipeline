import sys
import subprocess
from pathlib import Path


try:
    from ultralytics import YOLO, RTDETR
except ModuleNotFoundError:
    print("\n📦 'ultralytics' not found. Auto-installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    from ultralytics import YOLO, RTDETR

    print("Installation complete. Resuming inference...\n")

import cv2
import numpy as np
import torch
import jax
import jaxatari
from jaxatari.wrappers import PixelAndObjectObsWrapper, AtariWrapper


CURRENT_GAME = "venture"
mode = "live"

UNIVERSAL_NAMES = {
    0: "player", 1: "enemy", 2: "projectile", 3: "collectible",
    4: "structure", 5: "neutral", 6: "enemy_projectile"
}

UNIVERSAL_COLOURS = {
    0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 165, 0),
    3: (0, 215, 255), 4: (219, 55, 170), 5: (180, 180, 180),
    6: (255, 255, 255)
}

script_dir = Path(__file__).parent
yolo_path = script_dir / "YOLOv8nano" / "weights" / f"{CURRENT_GAME}.pt"
#rtdetr_path = script_dir.parent / "RT-DETR" / "yolo26n"  # Adjust placeholder path

conf = 0.40
iou_threshold = 0.50
img_size = 640
device = 0 if torch.cuda.is_available() else "cpu"

print(f"Loading models onto device: {device}...")
yolo = YOLO(str(yolo_path)).to(device)
rtdetr = yolo
print("Models loaded successfully!\n")


def draw_annotated_frame(frame_bgr, result, title):
    img = frame_bgr.copy()

    # Draw header bar for the title
    cv2.rectangle(img, (0, 0), (img.shape[1], 35), (30, 30, 30), -1)
    cv2.putText(img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if result.boxes is None or len(result.boxes) == 0:
        return img

    for box in result.boxes:
        cls_id = int(box.cls.item())
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        colour = UNIVERSAL_COLOURS.get(cls_id, (255, 255, 255))
        label = f"{UNIVERSAL_NAMES.get(cls_id, str(cls_id))} {confidence:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), colour, -1)
        cv2.putText(img, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return img


if mode == "live":
    print(f"Starting live side-by-side JaxAtari game: {CURRENT_GAME}")
    print("Press 'q' to quit.")

    base_env = jaxatari.make(CURRENT_GAME)
    atari_env = AtariWrapper(base_env)
    env = PixelAndObjectObsWrapper(atari_env)

    rng = jax.random.PRNGKey(0)
    rng, reset_key = jax.random.split(rng)
    current_obs, state = env.reset(reset_key)

    frame_count = 0
    infer_frames = 3
    last_combined = None



    while True:
        rng, action_key = jax.random.split(rng)
        action = jax.random.randint(action_key, shape=(), minval=0, maxval=18)
        current_obs, state, reward, stopped, truncated, info = env.step(state, action)
        frame_count += 1

        image_stack, _ = current_obs
        pixels = np.array(image_stack[-1])
        frame_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)


        display = cv2.resize(frame_bgr, (480, 630), interpolation=cv2.INTER_NEAREST)

        if frame_count % infer_frames == 0:
            res_yolo = yolo.predict(display, imgsz=img_size, conf=conf, device=device, verbose=False)[0]
            res_rtdetr = rtdetr.predict(display, imgsz=img_size, conf=conf, device=device, verbose=False)[0]

            frame_yolo = draw_annotated_frame(display, res_yolo, "YOLOv8 Nano")
            frame_rtdetr = draw_annotated_frame(display, res_rtdetr, "RT-DETR")

            last_combined = np.hstack((frame_yolo, frame_rtdetr))

        # Show the video feed
        if last_combined is not None:
            cv2.imshow("JaxAtari: YOLO vs RT-DETR", last_combined)
        else:
            blank_start = np.hstack((display, display))
            cv2.imshow("JaxAtari: YOLO vs RT-DETR", blank_start)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if stopped or truncated:
            rng, reset_key = jax.random.split(rng)
            current_obs, state = env.reset(reset_key)

    cv2.destroyAllWindows()
    print("Live session ended.")