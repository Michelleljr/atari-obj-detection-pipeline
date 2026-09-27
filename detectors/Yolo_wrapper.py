import json
import cv2
import jax
import jax.numpy as jnp
import jaxatari
from jaxatari.wrappers import AtariWrapper, YOLOObjectCentricWrapper
from pathlib import Path
import pygame
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = SCRIPT_DIR.parent / "praktikum_obj_detection_team" / "quirks_registry.json"
game_name = "kingkong"
YOLO_MODEL_PATH = SCRIPT_DIR.parent / "detectors" / "YOLOv8nano" / "weights" / f"{game_name}.pt"

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
    conf_threshold=0.15 if game_name == "hauntedhouse" else 0.40,
    imgsz=320 if game_name == "hauntedhouse" else 640,
    iou_threshold=0.50,
    display_size=(480, 640),
    clip_reward=True,
    autoreset=True,
)


class_colours = {
    0: (0,   255,   0),    # green   — player
    1: (255,   0,   0),    # red     — enemy
    2: (0, 165,   255),    # light blue  — projectile
    3: (255,   215, 0),    # gold    — collectible
    4: (170, 55,  219),    # purple    — structure
    5: (180, 180, 180),    # gray    — neutral
    6: (255, 255, 255)     # white   — enemy_projectile
}

def draw_boxes(frame_rgb, result):
    if frame_rgb is None:
        return None

    img = frame_rgb.copy()

    if result is None or result.boxes is None or len(result.boxes) == 0:
        return img

    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = map(int,box.xyxy[0].tolist())

        if game_name == "hauntedhouse" and y1 >= 405:
            continue

        colour = class_colours.get(cls_id,(0, 255, 0))

        # YOLO's class name
        name = wrapped_env.model.names.get(cls_id,str(cls_id))
        label = f"{name} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), colour, -1)
        cv2.putText(img, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return img


native_h, native_w = 210, 160
if hasattr(wrapped_env, "observation_space"):
    # If frame shape is available, grab real (H, W)
    shape = getattr(wrapped_env, "frame_shape", (210, 160))
    native_h, native_w = shape[0], shape[1]


TARGET_HEIGHT = 630
SCALE_FACTOR = TARGET_HEIGHT / native_h
DISPLAY_WIDTH = int(native_w * SCALE_FACTOR)
DISPLAY_HEIGHT = TARGET_HEIGHT

pygame.init()
screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
pygame.display.set_caption(f"JAXAtari YOLO Evaluation — {game_name.upper()}")
clock = pygame.time.Clock()

rng = jax.random.PRNGKey(42)
rng, reset_key = jax.random.split(rng)
obs_stack, state = wrapped_env.reset(reset_key)

running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    action_val = 0  # Default NOOP

    if keys[pygame.K_w] or keys[pygame.K_UP]:
        action_val = 2  # UP
    elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
        action_val = 5  # DOWN
    elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
        action_val = 4  # LEFT
    elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        action_val = 3  # RIGHT
    elif keys[pygame.K_SPACE]:
        action_val = 1  # FIRE

    prev_raw_frame = None

    if keys[pygame.K_q] or keys[pygame.K_ESCAPE]:
        running = False

    # Step Environment with YOLO Wrapper
    action = jnp.array(action_val, dtype=jnp.int32)
    obs_stack, state, reward, terminated, truncated, info = wrapped_env.step(state, action)

    # Render Annotated Frame to Pygame Display
    if wrapped_env.last_frame is not None:
        current_frame = wrapped_env.last_frame

        # Max-pool consecutive frames
        if prev_raw_frame is not None and prev_raw_frame.shape == current_frame.shape:
            pooled_frame = np.maximum(current_frame, prev_raw_frame)
        else:
            pooled_frame = current_frame

        prev_raw_frame = current_frame.copy()

        # Pass the pooled frame into box drawing function
        annotated_rgb = draw_boxes(pooled_frame, wrapped_env.last_results)

        pygame_ready_array = jnp.transpose(annotated_rgb, (1, 0, 2))
        surf = pygame.surfarray.make_surface(pygame_ready_array)
        scaled_surf = pygame.transform.scale(surf, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        screen.blit(scaled_surf, (0, 0))

    pygame.display.flip()
    clock.tick(60)

    if terminated or truncated:
        print("Game Over! Resetting environment...")
        rng, reset_key = jax.random.split(rng)
        obs_stack, state = wrapped_env.reset(reset_key)

pygame.quit()
