import sys
import subprocess

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    print("\n📦 'ultralytics' not found in this environment. Auto-installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])

    from ultralytics import YOLO

    print("Installation complete. Resuming inference...\n")

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
import jax

CURRENT_GAME = "hauntedhouse"
mode = "live"  # switch between already captured frames detections and live game detections [static/live/manual]

UNIVERSAL_NAMES = {
    0: "player", 1: "enemy", 2: "projectile", 3: "collectible",
    4: "structure", 5: "neutral", 6: "enemy_projectile"
}

UNIVERSAL_COLOURS = {
    0: (0, 255, 0),  # green   — player
    1: (0, 0, 255),  # red     — enemy
    2: (255, 165, 0),  # light blue  — projectile
    3: (0, 215, 255),  # gold    — collectible
    4: (219, 55, 170),  # purple    — structure
    5: (180, 180, 180),  # gray    — neutral
    6: (255, 255, 255)  # white   — enemy_projectile
}


GAME_REGISTRY = {
    "pong": {

        "names_override": {0: "player", 1: "enemy", 2: "ball"},
        "colours_override": {0: (0, 255, 100), 1: (0, 100, 255), 2: (255, 255, 255)}
    }
}

config = GAME_REGISTRY.get(CURRENT_GAME, {})


WEIGHTS_FILE = f"{CURRENT_GAME}.pt"
class_names = config.get("names_override", UNIVERSAL_NAMES)
class_colours = config.get("colours_override", UNIVERSAL_COLOURS)

script_dir = Path(__file__).parent
saved_checkpoint = script_dir / "weights" / WEIGHTS_FILE

static_dir = "dataset/images/val"  # folder of pngs (static mode)
static_output = "runs/yolov8n_inferences"
conf = 0.40 # confidence threshold
iou_threshold = 0.50 # IoU threshold for RT-DETR
img_size = 640
device = 0 if torch.cuda.is_available() else "cpu"

print(f"Loading checkpoint: {saved_checkpoint}")
model = YOLO(saved_checkpoint)
model.to(device)
print("Model loaded\n")


def draw_boxes(display_bgr, result, scale_factor=3.0):
    """Draw boxes on upscaled display window while running inference on native resolution."""
    img = display_bgr.copy()
    if result.boxes is None or len(result.boxes) == 0:
        return img

    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf_val = float(box.conf.item())

        # Native resolution coordinates (160x210)
        x1_nat, y1_nat, x2_nat, y2_nat = box.xyxy[0].tolist()

        # Filter bottom HUD false positives (y >= 135 on native frame)
        if CURRENT_GAME == "hauntedhouse" and y1_nat >= 135:
            continue

        # Scale coordinates up by 3x to match display size (480x630)
        x1 = int(x1_nat * scale_factor)
        y1 = int(y1_nat * scale_factor)
        x2 = int(x2_nat * scale_factor)
        y2 = int(y2_nat * scale_factor)

        colour = class_colours.get(cls_id, (255, 255, 255))
        label = f"{class_names.get(cls_id, str(cls_id))} {conf_val:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
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
        results = model.predict(frame_bgr, imgsz=img_size,
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

    game_name = CURRENT_GAME

    print(f"Starting live JaxAtari game: {game_name}, press 'q' to quit")
    base_env = jaxatari.make(game_name)
    atari_env = AtariWrapper(base_env)
    env = PixelAndObjectObsWrapper(atari_env)

    rng = jax.random.PRNGKey(0)
    rng, reset_key = jax.random.split(rng)
    current_obs, state = env.reset(reset_key)

    frame_count = 0
    infer_frames = 3  # run RT-DETR every 3 frames
    last_annotated = None

    while True:
        rng, action_key = jax.random.split(rng)
        action = jax.random.randint(action_key, shape=(), minval=0, maxval=18)
        current_obs, state, reward, stopped, truncated, info = env.step(state, action)
        frame_count += 1

        image_stack, _ = current_obs
        pixels = np.array(image_stack[-1])  # RGB
        frame_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

        # Upscale for visibility (Atari is only 160×210)
        display = cv2.resize(frame_bgr, (480, 630), interpolation=cv2.INTER_NEAREST)

        if frame_count % infer_frames == 0:
            # Run prediction on native 160x210 frame
            results = model.predict(frame_bgr, imgsz=320,
                                    conf=conf, iou=iou_threshold,
                                    device=device, verbose=False)
            # Draw scaled boxes on display window
            last_annotated = draw_boxes(display, results[0], scale_factor=3.0)

        my_window_name = " YOLOv8 Inference"

        if last_annotated is not None:
            cv2.imshow(my_window_name, last_annotated)
        else:
            cv2.imshow(my_window_name, display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if stopped or truncated:
            rng, reset_key = jax.random.split(rng)
            current_obs, state = env.reset(reset_key)

    cv2.destroyAllWindows()
    print("Live session ended.")


elif mode == "manual":
    import jaxatari
    from jaxatari.wrappers import PixelAndObjectObsWrapper, AtariWrapper
    import jax.numpy as jnp

    game_name = CURRENT_GAME
    print(f"\n🎮 MANUAL CONTROL MODE STARTED: {game_name.upper()}")
    print("Controls:")
    print("  W / Up    -> Move UP")
    print("  S / Down  -> Move DOWN")
    print("  A / Left  -> Move LEFT")
    print("  D / Right -> Move RIGHT")
    print("  Spacebar  -> FIRE")
    print("  Q         -> Quit Game\n")

    base_env = jaxatari.make(game_name)
    atari_env = AtariWrapper(base_env)
    env = PixelAndObjectObsWrapper(atari_env)

    KEY_MAPPING = {
        ord("w"): 2, 82: 2,  # 'w' or Up Arrow
        ord("s"): 5, 84: 5,  # 's' or Down Arrow
        ord("a"): 4, 81: 4,  # 'a' or Left Arrow
        ord("d"): 3, 83: 3,  # 'd' or Right Arrow
        32: 1,  # Spacebar (FIRE)
    }

    rng = jax.random.PRNGKey(0)
    rng, reset_key = jax.random.split(rng)
    current_obs, state = env.reset(reset_key)

    frame_count = 0
    infer_frames = 3
    last_annotated = None
    my_window_name = "YOLOv8 Inference - Manual Play"

    cv2.namedWindow(my_window_name)

    while True:
        frame_count += 1

        image_stack, _ = current_obs
        pixels = np.array(image_stack[-1])
        frame_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        display = cv2.resize(frame_bgr, (480, 630), interpolation=cv2.INTER_NEAREST)

        if frame_count % infer_frames == 0:
            # Run prediction on native 160x210 frame
            results = model.predict(frame_bgr, imgsz=320,
                                    conf=conf, iou=iou_threshold,
                                    device=device, verbose=False)
            # Draw scaled boxes on display window
            last_annotated = draw_boxes(display, results[0], scale_factor=3.0)

        if last_annotated is not None:
            cv2.imshow(my_window_name, last_annotated)
        else:
            cv2.imshow(my_window_name, display)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break

        action_int = KEY_MAPPING.get(key, 0)
        action = jnp.array(action_int, dtype=jnp.int32)

        current_obs, state, reward, stopped, truncated, info = env.step(state, action)

        if stopped or truncated:
            print("Game Over! Resetting environment...")
            rng, reset_key = jax.random.split(rng)
            current_obs, state = env.reset(reset_key)

    cv2.destroyAllWindows()
    print("Manual session ended.")