import cv2
import jax
import jax.numpy as jnp
import jaxatari
from jaxatari.wrappers import PixelAndObjectObsWrapper, AtariWrapper, YOLOObjectCentricWrapper
# from detectors.infer_yolov8n import draw_boxes  # reuse from play.py

game_name = "frostbite"
base_env = jaxatari.make(game_name)
atari_env = AtariWrapper(base_env)
pixel_env = PixelAndObjectObsWrapper(atari_env)

CLASS_MAP = {
    0: (0, 8, 1),      # bailey: 1 instance × 8 fields
    1: (8, 8, 12),     # obstacles: up to 12 instances × 8 fields = 96
    2: (104, 8, 1),    # bear: 1 instance × 8 fields
}

class_colours = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 165, 0)}

class_names = {
    0: "bailey",
    1: "obstacle",
    2: "bear",
}

wrapped_env = YOLOObjectCentricWrapper(
    env=pixel_env,
    yolo_model_path=r"C:\\Users\\anush\\Documents\\atari-obj-detection-pipeline\\detectors\\YOLOv8nano\\weights\\frostbite.pt",
    class_map=CLASS_MAP,          # your {class_id: (slot_start, slot_size)}
    num_features=5 + 5*12 + 5,
    frame_stack_size=4,
    conf_threshold=0.40,
)

def draw_boxes(frame_bgr, result):
    """Draw RT-DETR boxes on a BGR frame and return it."""
    img = frame_bgr.copy()
    if result.boxes is None or len(result.boxes) == 0:
        return img

    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        colour = class_colours.get(cls_id, (255, 255, 255))
        label = f"{class_names.get(cls_id, str(cls_id))} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), colour, -1)
        cv2.putText(img, label, (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return img

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

window_name = "Manual Play — YOLO Wrapper"
cv2.namedWindow(window_name)

while True:
    annotated = draw_boxes(wrapped_env.last_frame, wrapped_env.last_results) \
        if wrapped_env.last_results is not None else wrapped_env.last_frame
    cv2.imshow(window_name, annotated)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break

    action = jnp.array(KEY_MAPPING.get(key, 0), dtype=jnp.int32)
    obs_stack, state, reward, terminated, truncated, info = wrapped_env.step(state, action)

    if terminated or truncated:
        print("Game Over! Resetting...")
        rng, reset_key = jax.random.split(rng)
        obs_stack, state = wrapped_env.reset(reset_key)

cv2.destroyAllWindows()