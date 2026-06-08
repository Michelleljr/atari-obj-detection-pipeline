import os
import json
import cv2
import numpy as np
import jax
import jaxatari
from jaxatari.wrappers import PixelAndObjectObsWrapper, AtariWrapper

REGISTRY_PATH = "quirks_registry.json"

TARGET_FRAMES  = 10
DEBUG_MODE     = True

RUN_ALL_GAMES      = False
SINGLE_GAME_TARGET = "alien"

# Atari screen dimensions
SCREEN_W = 160.0
SCREEN_H = 210.0

# =============================================================================
#  GLOBAL CLASS MAP
# =============================================================================

GLOBAL_CLASSES = {
    0: "player",
    1: "enemy",
    2: "projectile",
    3: "collectible",
    4: "structure",
    5: "neutral",
}

# Debug colours per class ID
CLASS_COLORS = {
    0: (0,   255,   0),    # green   — player
    1: (0,   0,   255),    # red     — enemy
    2: (255, 165,   0),    # orange  — projectile
    3: (0,   215, 255),    # gold    — collectible
    4: (219, 55,  170),    # purple    — structure
    5: (180, 180, 180),    # gray    — neutral
}

# identifies how the obj is stored
TYPE_BORDER = {
    "entity": (255,   0, 255),   # magenta
    "grid":   (255,   255, 0),   # cyan
}

with open(REGISTRY_PATH, "r") as f:
    QUIRKS_REGISTRY = json.load(f)


def extract_entity(obj_data, entry):
    """
    Named-tuple entity with .x .y .width .height fields.

    JAX stacks frames on axis-0, most-recent first, so we use index [0]
    to get the current frame — NOT [-1] which is the oldest frame.
    """
    if not (hasattr(obj_data, 'x') and hasattr(obj_data, 'y')):
        return []

    x_off = entry.get("x_offset") or 0
    y_off = entry.get("y_offset") or 0

    try:
        active = np.atleast_1d(obj_data.active[0]) if hasattr(obj_data, 'active') else None
        xs = np.atleast_1d(obj_data.x[0])
        ys = np.atleast_1d(obj_data.y[0])
        ws = np.atleast_1d(obj_data.width[0])
        hs = np.atleast_1d(obj_data.height[0])
    except (AttributeError, IndexError):
        return []

    boxes = []
    for i in range(len(xs)):
        if active is not None and not bool(active[i]):
            continue
        x = float(xs[i]) + x_off
        y = float(ys[i]) + y_off
        w = float(ws[i])
        h = float(hs[i])
        if (x == 0 and y == 0) or w == 0 or h == 0:
            continue
        boxes.append((x, y, w, h))
    return boxes


def extract_grid(obj_data, entry):
    """
    Raw flat / 2-D JAX array where each element represents one tile.
    Non-zero (matching active_value) cells become bounding boxes.

    Tune grid_origin_x/y and cell_w/h in the registry
    """
    origin_x  = entry.get("grid_origin_x") or 0
    origin_y  = entry.get("grid_origin_y") or 0
    cell_w    = entry.get("cell_w") or 8
    cell_h    = entry.get("cell_h") or 8
    grid_cols = entry.get("grid_cols")
    active_v  = entry.get("active_value") if entry.get("active_value") is not None else 1.0

    if not grid_cols or grid_cols == "TODO":
        print(f"      [SKIP] grid_cols not set — update registry and re-run")
        return []
    try:
        arr = np.array(obj_data).reshape(-1)
    except Exception:
        return []

    # 3-D stacked frames: (frames, rows, cols) — unwrap to frame 0
    raw_shape = entry.get("_raw_shape") or []
    if len(raw_shape) >= 3:
        try:
            arr = np.array(obj_data[0]).reshape(-1)
        except Exception:
            pass

    boxes = []
    for idx, val in enumerate(arr):
        fval = float(val)
        if fval == 0.0 and active_v != 0.0:
            continue
        if active_v != 1.0 and fval != active_v:
            continue
        col = idx % int(grid_cols)
        row = idx // int(grid_cols)
        px  = origin_x + col * cell_w
        py  = origin_y + row * cell_h
        boxes.append((px, py, cell_w, cell_h))
    return boxes


def to_yolo(class_id, x, y, w, h):
    xc = min((x + w / 2.0) / SCREEN_W, 1.0)
    yc = min((y + h / 2.0) / SCREEN_H, 1.0)
    wn = min(w / SCREEN_W, 1.0)
    hn = min(h / SCREEN_H, 1.0)
    return f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"

def draw_debug_box(frame, x, y, w, h, class_id, obj_type, obj_name):
    fill_color   = CLASS_COLORS.get(class_id, (255, 255, 255))
    border_color = TYPE_BORDER.get(obj_type, (255, 255, 255))
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

    # Outer border encodes storage type (entity=magenta, grid=cyan)
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), border_color, 1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), fill_color, 1)

    class_name = GLOBAL_CLASSES.get(class_id, str(class_id))
    label = f"{class_name} | {obj_name}"
    cv2.putText(frame, label, (x1, max(y1 - 4, 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, fill_color, 1, cv2.LINE_AA)



games_to_run = list(QUIRKS_REGISTRY.keys()) if RUN_ALL_GAMES else [SINGLE_GAME_TARGET]

print("=" * 50)
print(" DATA EXTRACTOR")
print(f" Mode: {'ALL GAMES' if RUN_ALL_GAMES else SINGLE_GAME_TARGET.upper()}")
print(f" Frames per game: {TARGET_FRAMES}")
print(f" Debug overlay: {DEBUG_MODE}")
print("=" * 50)

for game_name in games_to_run:
    print(f"\n{'─' * 50}")
    print(f" {game_name.upper()}")
    print(f"{'─' * 50}")

    if game_name not in QUIRKS_REGISTRY:
        print(f"  [SKIP] Not in registry — run auto_explorer first.")
        continue

    objects_cfg = QUIRKS_REGISTRY[game_name].get("objects", {})

    # Only extract objects that have been fully assigned
    active_cfg = {
        name: entry for name, entry in objects_cfg.items()
        if isinstance(entry.get("class_id"), int)
    }
    skipped = [n for n in objects_cfg if n not in active_cfg]

    if not active_cfg:
        print(f"  [SKIP] No class_ids assigned yet — edit registry first.")
        continue

    if skipped:
        print(f"  [INFO] Skipping unassigned: {skipped}")

    obj_summary = [f"{n}(cls={e['class_id']})" for n, e in active_cfg.items()]
    print(f"  Extracting {len(active_cfg)} objects: {obj_summary}")


    base_dir = "quirks_debug" if DEBUG_MODE else "dataset"
    output_folder = f"{base_dir}/{game_name}"
    os.makedirs(output_folder, exist_ok=True)

    try:
        base_env  = jaxatari.make(game_name)
        atari_env = AtariWrapper(base_env)
        env       = PixelAndObjectObsWrapper(atari_env)

        rng = jax.random.PRNGKey(42)
        rng, reset_key = jax.random.split(rng)
        current_obs, state = env.reset(reset_key)

        frame_count       = 0
        saved_count       = 0
        MAX_IDLE_FRAMES   = 3000
        frames_since_save = 0

        while saved_count < TARGET_FRAMES and frames_since_save < MAX_IDLE_FRAMES:
            rng, action_key = jax.random.split(rng)
            action = jax.random.randint(action_key, shape=(),
                                        minval=0, maxval=env.action_space().n)

            current_obs, state, reward, stopped, truncated, info = env.step(state, action)
            frame_count       += 1
            frames_since_save += 1

            if frame_count % 20 == 0:
                image_stack, obs_stack = current_obs
                pixels    = np.array(image_stack[0])
                frame_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

                objects_dict = (obs_stack._asdict() if hasattr(obs_stack, '_asdict')
                                else obs_stack.__dict__)

                yolo_lines = []

                for obj_name, entry in active_cfg.items():
                    obj_data = objects_dict.get(obj_name)
                    if obj_data is None:
                        continue

                    class_id = entry["class_id"]
                    obj_type = entry.get("detected_type", "entity")

                    if obj_type == "entity":
                        boxes = extract_entity(obj_data, entry)
                    elif obj_type == "grid":
                        boxes = extract_grid(obj_data, entry)
                    else:
                        # unknown_array — needs manual classification in registry
                        continue

                    for (x, y, w, h) in boxes:
                        yolo_lines.append(to_yolo(class_id, x, y, w, h))
                        if DEBUG_MODE:
                            draw_debug_box(frame_bgr, x, y, w, h,
                                           class_id, obj_type, obj_name)

                if yolo_lines:
                    base = f"{output_folder}/frame_{saved_count:05d}"
                    with open(f"{base}.txt", "w") as f:
                        f.write("\n".join(yolo_lines))
                    cv2.imwrite(f"{base}.png", frame_bgr)
                    saved_count       += 1
                    frames_since_save  = 0
                    print(f"  [{saved_count:>4}/{TARGET_FRAMES}] "
                          f"{len(yolo_lines)} annotations saved")
                else:
                    print(f"  [frame {frame_count}] no visible objects — skipping")

            if stopped or truncated:
                rng, reset_key = jax.random.split(rng)
                current_obs, state = env.reset(reset_key)

        if frames_since_save >= MAX_IDLE_FRAMES:
            print(f"  [WARNING] Timed out — saved {saved_count}/{TARGET_FRAMES} frames.")
        else:
            print(f"  Done — {saved_count} frame pairs saved to {output_folder}/")

    except Exception as e:
        print(f"  [CRITICAL] {e}")
        continue


classes_yaml_path = "classes.yaml"
with open(classes_yaml_path, "w") as f:
    f.write("# YOLOv8 class definitions — generated by data_extractor.py\n")
    f.write(f"nc: {len(GLOBAL_CLASSES)}\n")
    f.write("names:\n")
    for class_id in sorted(GLOBAL_CLASSES):
        f.write(f"  {class_id}: {GLOBAL_CLASSES[class_id]}\n")

print(f"\n{'=' * 50}")
print(" PIPELINE COMPLETE")
print(f" Dataset written to: dataset/")
print(f" Class definitions:  {classes_yaml_path}")
print("=" * 50)