import os
import json
import jax
import jaxatari
from jaxatari.wrappers import PixelAndObjectObsWrapper, AtariWrapper

REGISTRY_PATH  = "quirks_registry.json"


# =============================================================================
#  GLOBAL CLASS MAP
#
#  0  player       — the character(s) the human controls
#  1  enemy        — anything that can kill/harm the player
#  2  projectile   — bullets, missiles, bombs (player or enemy)
#  3  collectible  — items that give points, power-ups, rewards
#  4  structure    — platforms, blocks, walls, static grid elements
#  5  neutral      — moving objects that are neither threat nor reward
# =============================================================================

GLOBAL_CLASSES = {
    0: "player",
    1: "enemy",
    2: "projectile",
    3: "collectible",
    4: "structure",
    5: "neutral",
}

# ---------------------------------------------------------------------------
# Keyword rules for auto-assigning class IDs.
# Checked in order — first match wins.
# Extend this list freely if you find object names that don't resolve well.
# ---------------------------------------------------------------------------

CLASS_KEYWORDS = [
    # class 0 — player
    (0, ["player", "bailey", "chicken", "human", "hook"]),

    # class 1 — enemy
    (1, ["enemy", "enemies", "otto", "bear", "shark", "spider",
         "centipede", "flea", "scorpion", "alien", "kong", "monkeys",
         "mothership", "byte_bat", "rock_muncher", "radar_mortar",
         "kamikaze", "bouncer", "chasing", "ghost", "monster", "chaser", "asteroids", "tentacles"]),

    # class 2 — projectile
    (2, ["bullet", "missile", "shot", "spell", "plasma",
         "bomb", "laser", "torpedo", "fireball", "projectile",
         "detonator", "homing", "ball", "player_spell", "spear"]),

    # class 3 — collectible
    (3, ["collectible", "fruit", "child", "bell", "princess",
         "item", "coin", "rejuvenator", "energy_pod", "fish",
         "score_item", "kill_item", "prize", "chest", "key",
         "power", "pellet", "letter", "target_word", "oxygen_line"]),

    # class 4 — structure
    (4, ["block", "platform", "ladder", "wall", "grid", "path",
         "bank", "installation", "forcefield", "densepack",
         "mountain", "lane_blocker", "road", "ice", "obstacle",
         "board", "cube", "bumper", "flipper", "plunger",
         "rollover", "hole", "spinner", "door", "portal",
         "rope", "conveyor", "completed_rect", "walked"]),

    # class 5 — neutral
    (5, ["truck", "car", "jet", "chopper", "cloud", "ufo",
         "falling_rock", "meteoroid", "debris", "disc", "boat"]),
]


def auto_assign_class(obj_name):
    """
    Walk CLASS_KEYWORDS and return the first matching class ID.
    Returns None if nothing matches —  'REVIEW' in the JSON.
    """
    name_lower = obj_name.lower()
    for class_id, keywords in CLASS_KEYWORDS:
        for kw in keywords:
            if kw in name_lower:
                return class_id
    return None


def detect_type(obj_data):
    """
    Returns {"detected_type": str, "raw_shape": list|None}
    """
    if obj_data is None:
        return {"detected_type": "non_spatial", "raw_shape": None}

    # 1. PRESERVE THIS: Standard JAXAtari entities with explicit coordinates
    if hasattr(obj_data, 'x') or hasattr(obj_data, 'xy'):
        return {"detected_type": "entity", "raw_shape": None}

    # 2. THE NEW ARRAY CLASSIFIERS
    if hasattr(obj_data, 'shape'):
        shape = list(obj_data.shape)
        total = 1
        for s in shape:
            total *= s

        # Ignore non-spatial scalars or tiny boolean arrays
        if len(shape) == 0 or total <= 8:
            return {"detected_type": "non_spatial", "raw_shape": shape}

        # Pattern 1: slot_flags [4, N] (No spatial data, skip rendering)
        if len(shape) == 2 and shape[0] == 4 and shape[1] < 210:
            return {"detected_type": "slot_flags", "raw_shape": shape}

        # Pattern 2: xy_pairs [4, N, 2] (Raw coordinates)
        if len(shape) == 3 and shape[0] == 4 and shape[2] == 2:
            return {"detected_type": "xy_pairs", "raw_shape": shape}

        # Pattern 3: true_2d_grid [4, rows, cols]
        if len(shape) == 3 and shape[0] == 4 and shape[2] != 2:
            return {"detected_type": "true_2d_grid", "raw_shape": shape}

        # Pattern 4: flat_per_row [4, 210] (Scanlines)
        if len(shape) == 2 and shape[1] == 210:
            return {"detected_type": "flat_per_row", "raw_shape": shape}

        # Catch anything weird that survived
        return {"detected_type": "unknown_array", "raw_shape": shape}

    return {"detected_type": "non_spatial", "raw_shape": None}



def build_entry(obj_name, detected_type, raw_shape):
    class_id = auto_assign_class(obj_name)
    class_id_value = class_id if class_id is not None else "REVIEW"

    # XY pairs are entities, true_2d_grids are grids
    is_entity = detected_type in ["entity", "xy_pairs"]
    is_grid = detected_type == "true_2d_grid"

    # Infer grid_cols from shape where possible
    if is_grid and raw_shape:
        if len(raw_shape) == 1:
            grid_cols = "TODO"   # flat 1-D: total cells known, cols unknown
        else:
            grid_cols = raw_shape[-1]   # (rows, cols) or (frames, rows, cols)
    else:
        grid_cols = None

    return {
        "class_id":      class_id_value,   # int if matched, "REVIEW" if not
        "detected_type": detected_type,


        "x_offset": 0    if is_entity else None,
        "y_offset": 0    if is_entity else None,

        # ── grid fields (if array) ──────────────────────────────────
        # Tune grid_origin_* and cell_w/h with the debug overlay.
        # grid_cols is auto-inferred where possible, set manually for flat 1-D.
        "grid_origin_x": 0    if is_grid else None,
        "grid_origin_y": 0    if is_grid else None,
        "cell_w":        8    if is_grid else None,
        "cell_h":        8    if is_grid else None,
        "grid_cols":     grid_cols,
        "active_value":  1.0  if is_grid else None,

        "_raw_shape":    raw_shape,
    }


# ---------------------------------------------------------------------------
# Games list
# ---------------------------------------------------------------------------
RUN_ALL_GAMES      = False
SINGLE_GAME_TARGET = "namethisgame"

ATARI_57 = [
    "alien", "amidar", "asterix", "asteroids", "atlantis",
    "bankheist", "beamrider", "berzerk", "blackjack", "breakout",
    "centipede", "choppercommand", "defender", "enduro", "fishingderby",
    "flagcapture", "freeway", "frostbite", "galaxian", "gravitar",
    "hauntedhouse", "humancannonball", "kangaroo", "kingkong", "lasergates",
    "montezumarevenge", "mspacman", "namethisgame", "phoenix", "pitfall",
    "pong", "privateeye", "qbert", "riverraid", "seaquest", "sirlancelot",
    "skiing", "slotmachine", "spaceinvaders", "spacewar", "surround",
    "tennis", "tetris", "timepilot", "tron", "turmoil", "venture",
    "videocube", "videopinball", "wordzapper", "yarsrevenge", "zaxxon",
]

AVAILABLE_GAMES = [
    'amidar', 'alien', 'asterix', 'asteroids', 'atlantis', 'bankheist',
    'beamrider', 'berzerk', 'blackjack', 'breakout', 'centipede',
    'choppercommand', 'enduro', 'fishingderby', 'flagcapture', 'freeway',
    'frostbite', 'galaxian', 'gravitar', 'hauntedhouse', 'humancannonball',
    'kangaroo', 'kingkong', 'lasergates', 'namethisgame', 'phoenix',
    'pong', 'qbert', 'riverraid', 'seaquest', 'sirlancelot', 'skiing',
    'slotmachine', 'spaceinvaders', 'spacewar', 'tennis', 'tetris',
    'timepilot', 'tron', 'turmoil', 'venture', 'videocube', 'videopinball',
    'wordzapper', 'mspacman', 'montezumarevenge',
]

TARGET_FRAMES  = 300   # frames explored per game to discover all objects

master_registry = {}

if os.path.exists(REGISTRY_PATH):
    with open(REGISTRY_PATH, "r") as f:
        master_registry = json.load(f)


games_to_run = AVAILABLE_GAMES if RUN_ALL_GAMES else [SINGLE_GAME_TARGET]

for game_name in games_to_run:
    print(f"\n{'=' * 50}")
    print(f" SCANNING: {game_name.upper()}")
    print(f"{'=' * 50}")

    try:
        base_env  = jaxatari.make(game_name)
        atari_env = AtariWrapper(base_env)
        env       = PixelAndObjectObsWrapper(atari_env)

        rng = jax.random.PRNGKey(42)
        rng, reset_key = jax.random.split(rng)
        current_obs, state = env.reset(reset_key)

        frame_count = 0
        discovered  = {}   # obj_name -> {detected_type, raw_shape}

        while frame_count < TARGET_FRAMES:
            rng, action_key = jax.random.split(rng)
            action = jax.random.randint(
                action_key, shape=(), minval=0, maxval=env.action_space().n
            )

            current_obs, state, reward, stopped, truncated, info = env.step(state, action)
            frame_count += 1

            image_stack, obs_stack = current_obs
            objects_dict = (obs_stack._asdict() if hasattr(obs_stack, '_asdict')
                            else obs_stack.__dict__)

            for obj_name, obj_data in objects_dict.items():
                if obj_name in discovered:
                    continue
                meta = detect_type(obj_data)
                if meta["detected_type"] == "non_spatial":
                    continue
                discovered[obj_name] = meta
                assigned = auto_assign_class(obj_name)
                class_label = (f"class {assigned} ({GLOBAL_CLASSES[assigned]})"
                               if assigned is not None else "REVIEW")
                print(f"  [{frame_count:>4}]  {meta['detected_type'].upper():14s}  "
                      f"{obj_name:<30s}  -> {class_label}"
                      + (f"  shape={meta['raw_shape']}" if meta['raw_shape'] else ""))

            if stopped or truncated:
                rng, reset_key = jax.random.split(rng)
                current_obs, state = env.reset(reset_key)

        print(f"\n  Done. {len(discovered)} spatial objects found.")

        # Count how many need manual review
        needs_review = [n for n in discovered
                        if auto_assign_class(n) is None]
        if needs_review:
            print(f"  *** REVIEW NEEDED for: {needs_review}")

        existing_objects = master_registry.get(game_name, {}).get("objects", {})
        game_objects = {}

        # Bring over EVERYTHING from the old registry (offsets, static entities, manual classes)
        for old_name, old_entry in existing_objects.items():
            game_objects[old_name] = old_entry

        # Add ONLY the new objects we discovered this run
        for obj_name, meta in discovered.items():
            if obj_name not in game_objects:
                game_objects[obj_name] = build_entry(
                    obj_name, meta["detected_type"], meta["raw_shape"]
                )

        # Fetch any manually tuned data you've already saved for this game
        existing_objects = master_registry.get(game_name, {}).get("objects", {})
        final_game_objects = {}

        # 1. Keep absolutely everything from your current file intact
        for old_name, old_entry in existing_objects.items():
            final_game_objects[old_name] = old_entry

         # 2. Only add completely new variables found during this scan
        for obj_name, entry_data in game_objects.items():
            if obj_name not in final_game_objects:
                final_game_objects[obj_name] = entry_data

        master_registry[game_name] = {"objects": final_game_objects}

        # Write after every game so a crash doesn't lose earlier work
        with open(REGISTRY_PATH, "w") as f:
            json.dump(master_registry, f, indent=4)
        print(f"  Registry saved -> {REGISTRY_PATH}")

    except Exception as e:
        print(f">>> [CRITICAL] Failed on {game_name}: {e}")
        continue

print(f"\n{'=' * 50}")
print(" SCAN COMPLETE — REVIEW SUMMARY")
print(f"{'=' * 50}")
for game_name, game_data in master_registry.items():
    flagged = [
        name for name, entry in game_data["objects"].items()
        if entry["class_id"] == "REVIEW"
    ]
    if flagged:
        print(f"  {game_name}: {flagged}")
unknown_types = [
    f"{g}/{n}"
    for g, gd in master_registry.items()
    for n, e in gd["objects"].items()
    if e["detected_type"] == "unknown_array"
]
if unknown_types:
    print(f"\n  unknown_array objects (inspect manually): {unknown_types}")
print("\nAll done. Edit quirks_registry.json to fix any REVIEW entries,")
print("then tune grid_origin_*/cell_w/h for grid objects using data_extractor's debug overlay.")


