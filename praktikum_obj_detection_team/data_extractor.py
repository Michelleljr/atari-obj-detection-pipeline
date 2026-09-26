import os
import json
import cv2
import numpy as np
import jax
import jaxatari
from jaxatari.wrappers import PixelAndObjectObsWrapper, AtariWrapper

from game_patches import GAME_PATCHES
from extractor_utils import (
    extract_entity,
    extract_xy_pairs,
    extract_true_2d_grid,
    extract_flat_per_row
)

REGISTRY_PATH = "quirks_registry.json"

TARGET_FRAMES  = 20
DEBUG_MODE     = True
DEBUG_LABELS   = False #false = hide text

RUN_ALL_GAMES  = False
TARGET_GAMES   = ["mspacman"]

# =============================================================================
#  GLOBAL CLASS MAP & COLOR DEFS
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

CLASS_COLORS = {
    0: (0,   255,   0),    # green   — player
    1: (0,   0,   255),    # red     — enemy
    2: (255, 165,   0),    # light blue  — projectile
    3: (0,   215, 255),    # gold    — collectible
    4: (219, 55,  170),    # purple    — structure
    5: (180, 180, 180),    # gray    — neutral
    6: (255, 255, 255)     # white   — enemy_projectile
}

TYPE_BORDER = {
    "entity":        (255,   0, 255),   # magenta
    "xy_pairs":      (255, 100,   0),   # orange-red
    "flat_per_row":  ( 64, 255, 255),   # cyan
    "true_2d_grid":  (255, 255,   0),   # yellow
    "static_entity": (255,   0,   0),   # blue
}

with open(REGISTRY_PATH, "r") as f:
    QUIRKS_REGISTRY = json.load(f)


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
                        elif obj_type in ["grid", "true_2d_grid"]:
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

                # Custom Game Patch Routing
                detected_enemy_type = None
                if game_name in GAME_PATCHES:
                    detected_enemy_type = GAME_PATCHES[game_name](
                        pixels, yolo_lines, frame_bgr, to_yolo, draw_debug_box, DEBUG_MODE
                    )

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

                    # Save text labels and image frame
                    base = f"{output_folder}/frame_{saved_count:05d}"
                    with open(f"{base}.txt", "w") as f:
                        f.write("\n".join(yolo_lines))
                        frame_bgr_resized = cv2.resize(frame_bgr, (640, 840), interpolation=cv2.INTER_NEAREST)
                        cv2.imwrite(f"{base}.png", frame_bgr_resized)

                    saved_count       += 1
                    frames_since_save  = 0

                    if game_name == "hauntedhouse":
                        tag = f"ENEMY ({detected_enemy_type})" if has_enemy else "EMPTY"
                        print(f"  [{saved_count:>4}/{TARGET_FRAMES}] Saved ({tag:<14s}) | "
                              f"Breakdown: {enemy_id_counts} | Empty: {empty_saved_total}/{EMPTY_QUOTA}")
                    else:
                        num_objs = len(yolo_lines)
                        print(f"  [{saved_count:>4}/{TARGET_FRAMES}] Saved frame_{saved_count-1:05d}.png ({num_objs} objects detected)")

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