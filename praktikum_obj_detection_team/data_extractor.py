import os
import json
import cv2
import numpy as np
import jax
import jaxatari
from jaxatari.wrappers import PixelAndObjectObsWrapper, AtariWrapper


REGISTRY_PATH = "quirks_registry.json"

TARGET_FRAMES  = 100
DEBUG_MODE     = False
DEBUG_LABELS   = True #false = hide text

RUN_ALL_GAMES      = False
SINGLE_GAME_TARGET = "asteroids"

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
    2: (255, 165,   0),    # light blue  — projectile
    3: (0,   215, 255),    # gold    — collectible
    4: (219, 55,  170),    # purple    — structure
    5: (180, 180, 180),    # gray    — neutral
}

# identifies how the obj is stored
TYPE_BORDER = {
    "entity": (255,   0, 255),   # magenta
    "xy_pairs":      (255, 100,   0),   # orange-red
    "flat_per_row":  ( 64, 255, 255),   # cyan
    "true_2d_grid":  (255, 255,   0),   # yellow
    "static_entity": (  255, 0, 0),   # blue
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

    w_pad = entry.get("w_padding") or 0
    h_pad = entry.get("h_padding") or 0

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

        if xs[i] <= 1 or ys[i] <= 1:
            continue

        x = float(xs[i]) + x_off
        y = float(ys[i]) + y_off
        w = float(ws[i]) + w_pad
        h = float(hs[i]) + h_pad

        if (x == 0 and y == 0) or w == 0 or h == 0:
            continue
        boxes.append((x, y, w, h))
    return boxes


def extract_xy_pairs(obj_data, entry):
    try:
        x_off = entry.get("x_offset") or 0
        y_off = entry.get("y_offset") or 0
        cw = entry.get("obj_w") or entry.get("cell_w") or 8
        ch = entry.get("obj_h") or entry.get("cell_h") or 8
        boxes = []
        for pair in obj_data[0]:
            x = float(pair[0]) + x_off
            y = float(pair[1]) + y_off
            if float(pair[0]) > 1 and float(pair[1]) > 1:
                boxes.append((x, y, cw, ch))
        return boxes
    except Exception:
        return []


def extract_true_2d_grid(obj_data, entry):
    """[4, R, C] — maps tile layouts dynamically based on row/column indices."""
    try:
        grid = obj_data[0]

        # 🟢 1. The Transpose Fix: Flip the mangled RAM matrix back to reality
        if entry.get("transpose_grid"):
            grid = grid.T

        rows, cols = grid.shape
        boxes = []

        cw = entry.get("cell_w") or (SCREEN_W / cols)
        ch = entry.get("cell_h") or (SCREEN_H / rows)

        # Separate the jump distance from the box width
        step_x = entry.get("step_x") or cw
        step_y = entry.get("step_y") or ch

        ox, oy = entry.get("grid_origin_x", 0), entry.get("grid_origin_y", 0)

        av = entry.get("active_value")
        amin = entry.get("active_min")

        for r in range(rows):
            for c in range(cols):
                val = float(grid[r, c])

                is_active = False
                if amin is not None:
                    if val >= float(amin):
                        is_active = True
                elif av is not None:
                    if val == float(av):
                        is_active = True
                else:
                    if val == 1.0:
                        is_active = True

                if is_active:
                    # 🟢 Draw the box at the Step coordinate, using the Cell size
                    boxes.append((ox + (c * step_x), oy + (r * step_y), cw, ch))
        return boxes
    except Exception:
        return []


def extract_flat_per_row(obj_data, entry):
    """
    [4, 210] — splits continuous vertical contours into manageable chunks
    so extreme bounding box aspect ratios do not break YOLOv8 training.
    """
    x_off = entry.get("x_offset") or 0
    y_off = entry.get("y_offset") or 0
    obj_w = entry.get("obj_w") or 8
    active_v = entry.get("active_value")
    max_run = int(entry.get("max_run_height") or 32)

    try:
        row_vals = np.array(obj_data[0], dtype=float)
    except Exception:
        return []

    if active_v is not None:
        active_mask = row_vals == float(active_v)
    else:
        active_mask = row_vals > 1

    boxes = []

    def _flush_segment(run_start, run_xs):
        for chunk_start in range(0, len(run_xs), max_run):
            chunk = run_xs[chunk_start : chunk_start + max_run]
            x_min = float(min(chunk))
            x_max = float(max(chunk))
            y_top = run_start + chunk_start
            boxes.append((
                x_min + x_off,
                float(y_top) + y_off,
                (x_max - x_min) + float(obj_w),
                float(len(chunk)),
            ))

    in_run = False
    run_start = 0
    run_xs = []

    for y_idx in range(len(row_vals)):
        if bool(active_mask[y_idx]):
            if not in_run:
                in_run = True
                run_start = y_idx
                run_xs = []
            run_xs.append(float(row_vals[y_idx]))
        else:
            if in_run:
                in_run = False
                if run_xs:
                    _flush_segment(run_start, run_xs)
                run_xs = []

    if in_run and run_xs:
        _flush_segment(run_start, run_xs)

    return boxes

def apply_custom_game_patches(game_name, pixels, yolo_lines, frame_bgr):
    """
    Handles hardcoded computer vision workarounds for games where
    JAXAtari has missing or incomplete RAM mappings.
    """
    if game_name == "enduro":
        y_min, y_max = 140, 155
        car_strip = pixels[y_min:y_max, :, :]
        brightness = np.sum(car_strip, axis=2)
        y_coords, x_coords = np.where(brightness > 400)

        if len(x_coords) > 0:
            car_x = int(np.min(x_coords))
            car_w = int(np.max(x_coords) - car_x)
            car_h = y_max - y_min

            # Map directly to class 0 (Player)
            yolo_lines.append(to_yolo(0, car_x, y_min, car_w, car_h))

            if DEBUG_MODE:
                draw_debug_box(frame_bgr, car_x, y_min, car_w, car_h,
                               0, "entity", "player_car")
        # TODO: any future games with same issue

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

    if DEBUG_LABELS:
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
                    class_id = entry["class_id"]
                    obj_type = entry.get("detected_type", "entity")

                    # Handle static entities directly from JSON without checking RAM
                    if obj_type == "static_entity":
                        boxes = entry.get("static_boxes") or []
                    else:
                        # Dynamic entities must exist in RAM
                        obj_data = objects_dict.get(obj_name)
                        if obj_data is None:
                            continue

                        if obj_type == "entity":
                            boxes = extract_entity(obj_data, entry)
                        elif obj_type == "grid" or obj_type == "true_2d_grid":
                            boxes = extract_true_2d_grid(obj_data, entry)
                        elif obj_type == "xy_pairs":
                            boxes = extract_xy_pairs(obj_data, entry)
                        elif obj_type == "flat_per_row":
                            boxes = extract_flat_per_row(obj_data, entry)
                        elif obj_type == "slot_flags":
                            continue
                        else:
                            continue

                    for (x, y, w, h) in boxes:
                        yolo_lines.append(to_yolo(class_id, x, y, w, h))
                        if DEBUG_MODE:
                            draw_debug_box(frame_bgr, x, y, w, h,
                                           class_id, obj_type, obj_name)

                apply_custom_game_patches(game_name, pixels, yolo_lines, frame_bgr)

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