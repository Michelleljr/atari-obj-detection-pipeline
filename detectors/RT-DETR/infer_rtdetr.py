import cv2
import numpy as np
from pathlib import Path
from ultralytics import RTDETR
import torch
import jax

mode = "live"          # switch between already captured frames detections and live game detections [static/live]
saved_checkpoint = r"detectors\runs\detect\runs\rtdetr_pong\rtdetr-x_v1\weights\best.pt" #replace with your own trained checkpoint path
static_dir = "dataset/images/val"    # folder of pngs (static mode)
static_output = "runs/rtdetr_inferences"
conf = 0.40            # confidence threshold
iou_threshold = 0.50            # IoU threshold for RT-DETR
img_size = 640
device = 0 if torch.cuda.is_available() else "cpu"

# Colours per class (BGR) — matches class order in dataset.yaml
# class_colours = {
#     0: (0,   255, 100),   # player— green
#     1: (0,   100, 255),   # enemy— orange-red
#     2: (255, 255,   0),   # bullet— cyan
#     3: (200,   0, 255),   # ufo— purple
#     4: (255, 165,   0),   # barricade— blue
# }
# class_names = {0: "player", 1: "enemy", 2: "bullet", 3: "ufo", 4: "barricade"} # for spaceinvaders
class_colours = {
    0: (0,   255, 100),   # player— green
    1: (0,   100, 255),   # enemy— orange-red
    2: (255, 255,   255), # ball— white
}
class_names = {0: "player", 1: "enemy", 2: "ball"} # for pong

print(f"Loading checkpoint: {saved_checkpoint}")
model = RTDETR(saved_checkpoint)
model.to(device)
print("Model loaded\n")


def draw_boxes(frame_bgr, result):
    """Draw RT-DETR boxes on a BGR frame and return it."""
    img = frame_bgr.copy()
    if result.boxes is None or len(result.boxes) == 0:
        return img

    for box in result.boxes:
        cls_id  = int(box.cls.item())
        conf    = float(box.conf.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        colour = class_colours.get(cls_id, (255, 255, 255))
        label  = f"{class_names.get(cls_id, str(cls_id))} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), colour, -1)
        cv2.putText(img, label, (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return img


# static mode: run on already captured frames and save outputs with bounding boxes
if mode == "static":
    out_dir = Path(static_output)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_paths = sorted(Path(static_dir).glob("*.png"))
    print(f"Running on {len(img_paths)} frames → {out_dir}\n")

    for img_path in img_paths:
        frame_bgr = cv2.imread(str(img_path))
        results   = model.predict(frame_bgr, imgsz=img_size,
                                  conf=conf, iou=iou_threshold,
                                  device=device, verbose=False)
        annotated = draw_boxes(frame_bgr, results[0])
        cv2.imwrite(str(out_dir / img_path.name), annotated)
        print(f"  {img_path.name} — {len(results[0].boxes)} detections")

    print(f"\nDone — annotated frames saved to {out_dir}")


# live mode: run on live JaxAtari game and display detections in a cv2 window
elif mode == "live":
    import jaxatari
    from jaxatari.wrappers import PixelAndObjectObsWrapper, AtariWrapper
    
    game_name = "pong"  # replace with the game name of your choice

    print(f"Starting live JaxAtari game: {game_name}, press 'q' to quit")
    base_env = jaxatari.make(game_name)
    atari_env = AtariWrapper(base_env)
    env = PixelAndObjectObsWrapper(atari_env)

    rng = jax.random.PRNGKey(0)
    rng, reset_key = jax.random.split(rng)
    current_obs, state = env.reset(reset_key)

    frame_count = 0
    infer_frames = 3         # run RT-DETR every 3 frames
    last_annotated = None

    while True:
        rng, action_key = jax.random.split(rng)
        action = jax.random.randint(action_key, shape=(), minval=0, maxval=18)
        current_obs, state, reward, stopped, truncated, info = env.step(state, action)
        frame_count += 1

        image_stack, _ = current_obs
        pixels    = np.array(image_stack[-1])          # RGB
        frame_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

        # Upscale for visibility (Atari is only 160×210)
        display = cv2.resize(frame_bgr, (480, 630), interpolation=cv2.INTER_NEAREST)

        if frame_count % infer_frames == 0:
            # results = model.predict(frame_bgr, imgsz=img_size,
            #                         conf=conf, iou=iou_threshold,
            #                         device=device, verbose=False)
            # Draw on the upscaled version by re-running on display frame
            results_big = model.predict(display, imgsz=img_size,
                                        conf=conf, iou=iou_threshold,
                                        device=device, verbose=False)
            last_annotated = draw_boxes(display, results_big[0])

        if last_annotated is not None:
            cv2.imshow(f"RT-DETR-X — {game_name}", last_annotated)
        else:
            cv2.imshow(f"RT-DETR-X — {game_name}", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if stopped or truncated:
            rng, reset_key = jax.random.split(rng)
            current_obs, state = env.reset(reset_key)

    cv2.destroyAllWindows()
    print("Live session ended.")
