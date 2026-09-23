import json
import cv2
import jax
import jax.numpy as jnp
import jaxatari
from jaxatari.wrappers import AtariWrapper, YOLOObjectCentricWrapper
from pathlib import Path

REGISTRY_PATH = Path("praktikum_obj_detection_team") / "quirks_registry.json"
game_name = "breakout"
YOLO_MODEL_PATH = Path("detectors") / "YOLOv8nano" / "weights" / f"{game_name}.pt"

def load_registry(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

registry = load_registry(REGISTRY_PATH)

base_env = jaxatari.make(game_name)
atari_env = AtariWrapper(base_env)
wrapped_env = YOLOObjectCentricWrapper(
    env=atari_env,
    yolo_model_path=YOLO_MODEL_PATH,
    quirks_registry=registry,
    game_name=game_name,
    frame_stack_size=4,
    frame_skip=4,
    conf_threshold=0.40,
    imgsz=640,
    iou_threshold=0.50,
    display_size=(480, 640),
    clip_reward=True,
    autoreset=True,
)

class_colours = {
    0: (0, 255, 0),        # green  — player
    1: (0, 0, 255),        # red    — enemy
    2: (255, 165, 0),      # orange — projectile
    3: (0, 215, 255),      # gold   — collectible
    4: (219, 55, 170),     # purple — structure
    5: (180, 180, 180),    # gray   — neutral
    6: (255, 255, 255),    # white  — enemy projectile
}

def draw_boxes(frame_bgr, result):
    img = frame_bgr.copy()

    if result is None or result.boxes is None or len(result.boxes) == 0:
        return img

    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = map(int,box.xyxy[0].tolist())
        colour = class_colours.get(cls_id,(255, 255, 255))

        # YOLO's class name
        name = wrapped_env.model.names.get(cls_id,str(cls_id))
        label = f"{name} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), colour, -1)
        cv2.putText(img, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return img

# Key maps for manual play
KEY_MAPPING = {
    ord("w"): 2, 82: 2,
    ord("s"): 5, 84: 5,
    ord("a"): 4, 81: 4,
    ord("d"): 3, 83: 3,
    32: 1,
}

rng = jax.random.PRNGKey(0)
rng, reset_key = jax.random.split(rng)
obs_stack, state = wrapped_env.reset(reset_key)

print("\nInitial observation shape:", obs_stack.shape)
print("Observation dtype:", obs_stack.dtype)

window_name = f"Manual Play — {game_name}"

print(f"\nYOLO layout for {game_name}:")

for class_id in sorted(wrapped_env.class_offsets):
    print(f"  class {class_id}: {wrapped_env.class_names[class_id]} "
        f"-> offset={wrapped_env.class_offsets[class_id]}, "
        f"size={wrapped_env.class_slot_sizes[class_id]}")

print(f"  total features={wrapped_env.num_features}")
# cv2.namedWindow(window_name)
# cv2.resizeWindow(window_name, DISPLAY_SIZE[0], DISPLAY_SIZE[1])

while True:
    if wrapped_env.last_results is not None:
        annotated = draw_boxes(wrapped_env.last_frame, wrapped_env.last_results)
    else:
        annotated = wrapped_env.last_frame

    if annotated is not None:
        cv2.imshow(window_name, annotated)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break

    action = jnp.array(KEY_MAPPING.get(key, 0), dtype=jnp.int32)
    obs_stack, state, reward, terminated, truncated, info = wrapped_env.step(state, action)
    if terminated or truncated:
        print("Game Over, resetting...")
        rng, reset_key = jax.random.split(rng)
        obs_stack, state = wrapped_env.reset(reset_key)

cv2.destroyAllWindows()
