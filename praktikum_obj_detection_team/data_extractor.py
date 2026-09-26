import os
import json
import cv2
import numpy as np
import jax
import jaxatari
from jaxatari.wrappers import PixelAndObjectObsWrapper, AtariWrapper


REGISTRY_PATH = "quirks_registry.json"

TARGET_FRAMES  = 100
DEBUG_MODE     = True
DEBUG_LABELS   = True #false = hide text

RUN_ALL_GAMES      = False
TARGET_GAMES = ["hauntedhouse"]

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
    6: "enemy_projectile"
}

# Debug colours per class ID
CLASS_COLORS = {
    0: (0,   255,   0),    # green   — player
    1: (0,   0,   255),    # red     — enemy
    2: (255, 165,   0),    # light blue  — projectile
    3: (0,   215, 255),    # gold    — collectible
    4: (219, 55,  170),    # purple    — structure
    5: (180, 180, 180),    # gray    — neutral
    6: (255, 255, 255)     # white   — enemy_projectile
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

        if entry.get("transpose_grid"):
            grid = grid.T

        rows, cols = grid.shape
        boxes = []

        cw = entry.get("cell_w") or (SCREEN_W / cols)
        ch = entry.get("cell_h") or (SCREEN_H / rows)

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

def classify_hauntedhouse_enemy(crop_rgb):
    """
    Inspects cropped sprite pixels to determine enemy type by color.
    Returns: 'ghost', 'bat', 'spider', or 'unknown'
    """
    if crop_rgb.size == 0:
        return 'unknown'

    # Spider / Tarantula (Green / Yellow-Green OR Orange / Yellow-Orange)
    # Green check: High Green, lower Red & Blue
    green_pixels = np.sum((crop_rgb[:, :, 1] > 140) & (crop_rgb[:, :, 2] < 120))
    # Orange / Yellow check: High Red & High Green, Low Blue
    orange_pixels = np.sum((crop_rgb[:, :, 0] > 150) &
                           (crop_rgb[:, :, 1] > 100) &
                           (crop_rgb[:, :, 2] < 80))

    if green_pixels > 4 or orange_pixels > 4:
        return 'spider'

    # Bat (Crimson Red: High Red, Low Green & Blue)
    red_pixels = np.sum((crop_rgb[:, :, 0] > 140) &
                        (crop_rgb[:, :, 1] < 80) &
                        (crop_rgb[:, :, 2] < 80))
    if red_pixels > 5:
        return 'bat'

    # Ghost (Off-white / Cyan / Gray: High RGB across all 3 channels)
    white_pixels = np.sum((crop_rgb[:, :, 0] > 150) &
                          (crop_rgb[:, :, 1] > 150) &
                          (crop_rgb[:, :, 2] > 150))
    if white_pixels > 6:
        return 'ghost'

    return 'spider' if (green_pixels + orange_pixels) > 0 else 'ghost'


def apply_custom_game_patches(game_name, pixels, yolo_lines, frame_bgr):
    """
    Handles hardcoded computer vision workarounds for games where
    JAXAtari has missing or incomplete RAM mappings.
    """
    detected_enemy_type = None  # FIX: Always initialize at function start

    if game_name == "enduro":
        y_min, y_max = 140, 155
        car_strip = pixels[y_min:y_max, :, :]
        brightness = np.sum(car_strip, axis=2)
        y_coords, x_coords = np.where(brightness > 400)

        if len(x_coords) > 0:
            car_x = int(np.min(x_coords))
            car_w = int(np.max(x_coords) - car_x)
            car_h = y_max - y_min

            patch_h, patch_w = pixels.shape[:2]
            yolo_lines.append(to_yolo(0, car_x, y_min, car_w, car_h, patch_h, patch_w))

            if DEBUG_MODE:
                draw_debug_box(frame_bgr, car_x, y_min, car_w, car_h,
                               0, "entity", "player_car")

    elif game_name == "mspacman":
        patch_h, patch_w = pixels.shape[:2]
        player_color = np.array([210, 164, 74])
        lower_p = np.clip(player_color - 25, 0, 255)
        upper_p = np.clip(player_color + 25, 0, 255)
        p_mask = cv2.inRange(pixels, lower_p, upper_p)

        p_contours, _ = cv2.findContours(p_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in p_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 4 <= w <= 16 and 4 <= h <= 16 and y < 176:
                yolo_lines.append(to_yolo(0, x, y, w, h, patch_h, patch_w))
                if DEBUG_MODE:
                    draw_debug_box(frame_bgr, x, y, w, h, 0, "entity", "player_cv")

        palette_colors = [
            np.array([210, 164, 116]),
            np.array([228, 111, 111]),
            np.array([200, 200, 80]),
            np.array([214, 214, 214])
        ]

        combined_pellet_mask = np.zeros(pixels.shape[:2], dtype=np.uint8)

        for color in palette_colors:
            lower = np.clip(color - 15, 0, 255)
            upper = np.clip(color + 15, 0, 255)
            mask = cv2.inRange(pixels, lower, upper)
            combined_pellet_mask = cv2.bitwise_or(combined_pellet_mask, mask)

        pel_contours, _ = cv2.findContours(combined_pellet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in pel_contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if 2 <= w <= 12 and 1 <= h <= 6 and y < 176:
                yolo_lines.append(to_yolo(3, x, y, w, h, patch_h, patch_w))
                if DEBUG_MODE:
                    draw_debug_box(frame_bgr, x, y, w, h, 3, "entity", "pellet_cv")

    elif game_name == "phoenix":
        patch_h, patch_w = pixels.shape[:2]

        lower_s = np.array([200, 200, 200])
        upper_s = np.array([255, 255, 255])
        s_mask = cv2.inRange(pixels, lower_s, upper_s)

        s_contours, _ = cv2.findContours(s_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in s_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 14 <= w <= 40 and 14 <= h <= 40 and y > 150:
                yolo_lines.append(to_yolo(4, x, y, w, h, patch_h, patch_w))
                if DEBUG_MODE:
                    draw_debug_box(frame_bgr, x, y, w, h, 4, "entity", "shield_cv")

        ep_contours, _ = cv2.findContours(s_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in ep_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 4 and 3 <= h <= 15 and y > 25:
                pad = 2
                px = max(0, x - pad)
                py = max(0, y - pad)
                pw = w + (pad * 2)
                ph = h + (pad * 2)

                yolo_lines.append(to_yolo(6, px, py, pw, ph, patch_h, patch_w))
                if DEBUG_MODE:
                    draw_debug_box(frame_bgr, px, py, pw, ph, 6, "entity", "enemy_proj_cv")

    elif game_name == "hauntedhouse":
        patch_h, patch_w = pixels.shape[:2]

        yolo_lines[:] = [line for line in yolo_lines if not line.startswith("0 ")]

        # Check if the room background is currently white (lightning/flash effect)
        # Check average brightness of top corners
        top_left_bg = pixels[5:20, 5:20]
        is_white_bg = np.mean(top_left_bg) > 200

        claimed_mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
        claimed_mask[135:, :] = 255  # Lock out bottom HUD completely

        #PLAYER EYES EXTRACTION
        if not is_white_bg:
            # Normal dark/blue room -> Eyes are WHITE pixels
            white_mask = cv2.inRange(pixels, np.array([200, 200, 200]), np.array([255, 255, 255]))
            white_mask[135:, :] = 0
            y_indices, x_indices = np.where(white_mask > 0)
        else:
            # Flash/White room -> Eyes invert to BLACK pixels
            black_eye_mask = cv2.inRange(pixels, np.array([0, 0, 0]), np.array([30, 30, 30]))
            black_eye_mask[135:, :] = 0
            y_indices, x_indices = np.where(black_eye_mask > 0)

        # Look for small eye clusters
        if len(x_indices) >= 4:
            x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
            y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))

            ew = (x_max - x_min) + 1
            eh = (y_max - y_min) + 1

            if 4 <= ew <= 18 and 3 <= eh <= 12:
                pad = 2
                px = max(0, x_min - pad)
                py = max(0, y_min - pad)
                pw = ew + (pad * 2)
                ph = eh + (pad * 2)

                yolo_lines.append(to_yolo(0, px, py, pw, ph, patch_h, patch_w))
                claimed_mask[py:py + ph, px:px + pw] = 255

                if DEBUG_MODE:
                    draw_debug_box(frame_bgr, px, py, pw, ph, 0, "entity", "player_eyes_cv")

        hsv = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)

        if not is_white_bg:
            # Normal room: ignore blue walls and black background
            blue_mask = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([140, 255, 255]))
            black_mask = cv2.inRange(pixels, np.array([0, 0, 0]), np.array([15, 15, 15]))
            ignore_mask = cv2.bitwise_or(blue_mask, black_mask)
            fg_mask = cv2.bitwise_not(ignore_mask)
        else:
            # White room: ignore white walls (RGB > 200)
            white_wall_mask = cv2.inRange(pixels, np.array([200, 200, 200]), np.array([255, 255, 255]))
            fg_mask = cv2.bitwise_not(white_wall_mask)

        # Subtract claimed region (HUD + Player eyes)
        fg_mask[claimed_mask > 0] = 0

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if 5 <= w <= 25 and 6 <= h <= 30:
                if not (w <= 12 and h <= 10 and x > 110):  # Skip static room artifact
                    sprite_crop = pixels[y:y + h, x:x + w]
                    detected_enemy_type = classify_hauntedhouse_enemy(sprite_crop)

                    yolo_lines.append(to_yolo(1, x, y, w, h, patch_h, patch_w))

                    if DEBUG_MODE:
                        draw_debug_box(frame_bgr, x, y, w, h, 1, "entity", "enemies_cv")
    return detected_enemy_type


def to_yolo(class_id, x, y, w, h, actual_h, actual_w):
    xc = min((x + w / 2.0) / actual_w, 1.0)
    yc = min((y + h / 2.0) / actual_h, 1.0)
    wn = min(w / actual_w, 1.0)
    hn = min(h / actual_h, 1.0)
    return f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"

def draw_debug_box(frame, x, y, w, h, class_id, obj_type, obj_name):
    fill_color   = CLASS_COLORS.get(class_id, (255, 255, 255))
    border_color = TYPE_BORDER.get(obj_type, (255, 255, 255))
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), border_color, 1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), fill_color, 1)

    if DEBUG_LABELS:
        class_name = GLOBAL_CLASSES.get(class_id, str(class_id))
        label = f"{class_name} | {obj_name}"
        cv2.putText(frame, label, (x1, max(y1 - 4, 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, fill_color, 1, cv2.LINE_AA)


games_to_run = list(QUIRKS_REGISTRY.keys()) if RUN_ALL_GAMES else TARGET_GAMES

print("=" * 50)
print(" DATA EXTRACTOR")
print(f" Mode: {'ALL GAMES' if RUN_ALL_GAMES else f'BATCH ({len(TARGET_GAMES)} games)'}")
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
        TOTAL_ENEMY_QUOTA = TARGET_FRAMES // 2
        EMPTY_QUOTA       = TARGET_FRAMES - TOTAL_ENEMY_QUOTA
        MIN_PER_ID        = int(TARGET_FRAMES * 0.10)

        enemy_id_counts   = {'ghost': 0, 'bat': 0, 'spider': 0}
        enemy_saved_total = 0
        empty_saved_total = 0

        while saved_count < TARGET_FRAMES and frames_since_save < MAX_IDLE_FRAMES:
            rng, action_key, chance_key = jax.random.split(rng, 3)
            roll = float(jax.random.uniform(chance_key))

            if roll < 0.50:
                action = 1
            else:
                action = jax.random.randint(action_key, shape=(),
                                            minval=0, maxval=env.action_space().n)

            current_obs, state, reward, stopped, truncated, info = env.step(state, action)
            frame_count       += 1
            frames_since_save += 1

            if game_name == "hauntedhouse" and frame_count % 100 == 0:
                rng, reset_key = jax.random.split(rng)
                current_obs, state = env.reset(reset_key)

            if frame_count % 20 == 0:
                image_stack, obs_stack = current_obs
                pixels    = np.array(image_stack[0])
                frame_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

                actual_h, actual_w = pixels.shape[:2]

                objects_dict = (obs_stack._asdict() if hasattr(obs_stack, '_asdict')
                                else obs_stack.__dict__)

                yolo_lines = []

                for obj_name, entry in active_cfg.items():
                    class_id = entry["class_id"]
                    obj_type = entry.get("detected_type", "entity")

                    if obj_type == "static_entity":
                        boxes = entry.get("static_boxes") or []
                    else:
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
                        if w < 8:
                            pad_x = (8 - w) // 2
                            x = max(0, x - pad_x)
                            w = 8
                        if h < 8:
                            pad_y = (8 - h) // 2
                            y = max(0, y - pad_y)
                            h = 8

                        yolo_lines.append(to_yolo(class_id, x, y, w, h, actual_h, actual_w))
                        if DEBUG_MODE:
                            draw_debug_box(frame_bgr, x, y, w, h,
                                           class_id, obj_type, obj_name)

                # FIX: Called only ONCE and store returned string safely
                detected_enemy_type = apply_custom_game_patches(game_name, pixels, yolo_lines, frame_bgr)

                if yolo_lines:
                    has_enemy = any(line.startswith("1 ") for line in yolo_lines)

                    if game_name == "hauntedhouse":
                        if has_enemy:
                            if enemy_saved_total >= TOTAL_ENEMY_QUOTA:
                                continue

                            e_type = detected_enemy_type or 'ghost'

                            if e_type not in enemy_id_counts:
                                enemy_id_counts[e_type] = 0

                            other_types_needed = sum(
                                max(0, MIN_PER_ID - enemy_id_counts.get(t, 0))
                                for t in ['ghost', 'bat', 'spider'] if t != e_type
                            )
                            enemy_budget_left = TOTAL_ENEMY_QUOTA - enemy_saved_total

                            if frames_since_save < 1500:
                                if enemy_budget_left <= other_types_needed and enemy_id_counts[e_type] >= MIN_PER_ID:
                                    continue

                            enemy_id_counts[e_type] += 1
                            enemy_saved_total += 1

                        else:
                            if empty_saved_total >= EMPTY_QUOTA:
                                continue
                            empty_saved_total += 1

                    base = f"{output_folder}/frame_{saved_count:05d}"
                    with open(f"{base}.txt", "w") as f:
                        f.write("\n".join(yolo_lines))
                        frame_bgr = cv2.resize(frame_bgr, (640, 840), interpolation=cv2.INTER_NEAREST)
                        cv2.imwrite(f"{base}.png", frame_bgr)
                    saved_count       += 1
                    frames_since_save  = 0

                    if game_name == "hauntedhouse":
                        tag = f"ENEMY ({detected_enemy_type})" if has_enemy else "EMPTY"
                        print(f"  [{saved_count:>4}/{TARGET_FRAMES}] Saved ({tag:<14s}) | "
                              f"Breakdown: {enemy_id_counts} | Empty: {empty_saved_total}/{EMPTY_QUOTA}")

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