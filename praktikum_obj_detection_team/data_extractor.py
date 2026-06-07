import os
import cv2
import numpy as np
import jax
import jaxatari
from jaxatari.wrappers import PixelAndObjectObsWrapper, AtariWrapper

TARGET_FRAMES = 10
DEBUG_MODE = True #toggle to false if used purely for frame extraction

GAME_REGISTRY = {
    "amidar": {
        "player_gorilla": 0, "player_paint_roller": 0, "enemy": 1,
        "walked_on_paths": 4, "completed_rectangles": 4, "paths": 4
    },
    "alien": {
        "player": 0, "enemies": 1, "enemies_killable": 1,
        "score_item_position": 3, "kill_item_position": 3
    },
    "asterix": {
        "player": 0, "enemies": 1, "collectibles": 3
    },
    "asteroids": {
        "player": 0, "asteroids": 1, "missiles": 2
    },
    "atlantis": {
        "enemy": 1, "plasma": 2, "bullet": 2, "installations_alive": 4
    },
    "bankheist": {
        "player": 0, "enemies": 1, "dynamite": 2, "banks": 4
    },
    "beamrider": {
        "player": 0, "chasing_meteoroids": 1, "mothership": 1, "bouncer": 1,
        "white_ufo": 1, "kamikaze": 1, "falling_rocks": 1, "enemy_shots": 2,
        "player_shots": 2, "rejuvenator": 3, "coins": 3, "lane_blockers": 4,
        "white_ufo_pattern_timer": 5, "white_ufo_pattern_id": 5
    },
    "berzerk": {
        "player": 0, "otto": 1, "enemies": 1, "enemy_bullets": 2, "player_bullet": 2
    },
    "blackjack": {
        # no spatial objects
    },
    "breakout": {
        "player": 0, "ball": 2, "blocks": 4
    },
    "centipede": {
        "player": 0, "spider": 1, "centipede": 1, "flea": 1,
        "scorpion": 1, "player_spell": 2, "mushrooms": 4
    },
    "choppercommand": {
        "player": 0, "trucks": 1, "jets": 1, "choppers": 1,
        "player_missiles": 2, "enemy_missiles": 2
    },
    "enduro": {
        "enemy_positions": 1, "road_features": 4
    },
    "fishingderby": {
        "hook_p1": 0, "shark": 1, "fish": 3
    },
    "flagcapture": {
        "player": 0, "grid": 4
    },
    "freeway": {
        "chicken": 0, "car": 1
    },
    "frostbite": {
        "bailey": 0, "bear": 1, "obstacles": 4, "ice_grid": 4
    },
    "galaxian": {
        "player": 0, "aliens": 1, "missiles": 2, "bombs": 2
    },
    "hauntedhouse": {
        "player": 0, "enemies": 1, "items": 3
    },
    "humancannonball": {
        "human": 0, "water_tower": 4, "cannon": 4
    },
    "kangaroo": {
        "player": 0, "monkeys": 1, "thrown_coconuts": 2, "falling_coconut": 2,
        "child": 3, "bell": 3, "fruits": 3, "platforms": 4, "ladders": 4
    },
    "kingkong": {
        "player": 0, "kong": 1, "bombs": 2, "princess": 3
    },
    "lasergates": {
        "player": 0, "byte_bat": 1, "rock_muncher": 1, "radar_mortar": 1,
        "player_missile": 2, "rock_muncher_missile": 2, "radar_mortar_missile": 2,
        "homing_missile": 2, "detonator": 3, "energy_pod": 3, "forcefields": 4,
        "densepack": 4, "upper_mountains": 4, "lower_mountains": 4
    }
}

RUN_ALL_GAMES = True
SINGLE_GAME_TARGET = "spaceinvaders"

ACTIVE_GAMES = [
    'amidar', 'alien', 'asterix', 'asteroids', 'atlantis', 'bankheist',
    'beamrider', 'berzerk', 'blackjack', 'breakout', 'centipede',
    'choppercommand', 'enduro', 'fishingderby', 'flagcapture', 'freeway',
    'frostbite', 'galaxian', 'gravitar', 'hauntedhouse', 'humancannonball',
    'kangaroo', 'kingkong', 'lasergates'
]

MY_GAMES = ACTIVE_GAMES if RUN_ALL_GAMES else [SINGLE_GAME_TARGET]

print("Starting Extraction...")


for GAME_NAME in MY_GAMES:
    print(f"\n{'=' * 50}")
    print(f" INITIALIZING PIPELINE: {GAME_NAME.upper()}")
    print(f"{'=' * 50}")

    if GAME_NAME not in GAME_REGISTRY:
        print(f">>> [WARNING] Skipping {GAME_NAME}: No mapping found in GAME_REGISTRY.")
        continue

    class_mapping = GAME_REGISTRY[GAME_NAME]

    if not class_mapping:
        print(f">>> [INFO] Skipping {GAME_NAME}: Game has no spatial objects mapped (Logic Game).")
        continue

    #folder setup
    output_folder = f"captured_{TARGET_FRAMES}frames_{GAME_NAME}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    try:
        # start the game up
        print(f"Starting game {GAME_NAME}")
        base_env = jaxatari.make(GAME_NAME)
        atari_env = AtariWrapper(base_env)
        env = PixelAndObjectObsWrapper(atari_env)

        #pure jax setup, jax requires random keys for everything it does
        rng = jax.random.PRNGKey(42)
        rng, reset_key = jax.random.split(rng)

        current_obs, state = env.reset(reset_key)

        frame_count = 0
        saved_count = 0

        MAX_IDLE_FRAMES = 3000
        frames_since_last_save = 0

        print(f"Starting Extraction for {GAME_NAME}...")

        while saved_count < TARGET_FRAMES and frames_since_last_save < MAX_IDLE_FRAMES:
            rng, action_key = jax.random.split(rng)
            action = jax.random.randint(action_key, shape=(), minval=0, maxval=env.action_space().n)

            current_obs, state, reward, stopped, truncated, info = env.step(state, action)
            frame_count += 1
            frames_since_last_save += 1

            if frame_count % 20 == 0:
                base_filename = f"{output_folder}/frame_{saved_count}"

                image_stack, obs_stack = current_obs
                pixels = np.array(image_stack[-1])
                frame_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)


                yolo_lines = []

                if hasattr(obs_stack, '_asdict'):
                    objects_dict = obs_stack._asdict()
                else:
                    objects_dict = obs_stack.__dict__


                for obj_name, obj_data in objects_dict.items():

                    class_id = class_mapping.get(obj_name, -1)

                    if class_id == -1:
                        continue

                    # Map the class IDs
                    # if obj active but has no jax flag, try/except it
                    # ensure object has x/y properties before mathing
                    if not (hasattr(obj_data, 'x') and hasattr(obj_data, 'y')):
                        continue

                    # JAX stores coordinates directly as .x, .y, .width, .height arrays
                    try:
                        #some objects may already have active array tracking if alive
                        is_active = True
                        if hasattr(obj_data, 'active'):
                            active_array = np.atleast_1d(obj_data.active[-1])
                        else:
                            active_array = None

                        x_array = np.atleast_1d(obj_data.x[-1])
                        y_array = np.atleast_1d(obj_data.y[-1])
                        w_array = np.atleast_1d(obj_data.width[-1])
                        h_array = np.atleast_1d(obj_data.height[-1])

                    except AttributeError:
                        continue



                    for i in range(len(x_array)):
                        # If the object has an active flag and it's False, skip
                        if active_array is not None and not bool(active_array[i]):
                            continue

                        x = float(x_array[i])
                        y = float(y_array[i])
                        w = float(w_array[i])
                        h = float(h_array[i])

                        # Skip dead/hidden objects
                        if (x == 0 and y == 0) or w == 0 or h == 0:
                            continue

                        # YOLO Math (Normalized 0.0 to 1.0)
                        x_center = min((x + (w / 2.0)) / 160.0, 1.0)
                        y_center = min((y + (h / 2.0)) / 210.0, 1.0)
                        w_norm = min(w / 160.0, 1.0)
                        h_norm = min(h / 210.0, 1.0)

                        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

                        if DEBUG_MODE:
                            cv2.rectangle(frame_bgr, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 255), 2)
                            # text label
                            cv2.putText(frame_bgr, str(class_id), (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                        (255, 0, 255), 1)

                #savve only if objects found
                if yolo_lines:
                    with open(f"{base_filename}.txt", "w") as f:
                        f.write("\n".join(yolo_lines))
                    cv2.imwrite(f"{base_filename}.png", frame_bgr)
                    saved_count += 1
                    frames_since_last_save = 0

                    print(f"  -> Generated pair {saved_count}/{TARGET_FRAMES}: {len(yolo_lines)} objects found")
                else:
                    print(f"  -> Frame {frame_count} skipped (no mapped objects visible)")

            if stopped or truncated:
                rng, reset_key = jax.random.split(rng)
                current_obs, state = env.reset(reset_key)


        if frames_since_last_save >= MAX_IDLE_FRAMES:
            print(f">>> [WARNING] Extraction timed out for {GAME_NAME}. Stuck on empty screens. Saved {saved_count}/{TARGET_FRAMES} frames.")

    except Exception as e:
        print(f">>> [CRITICAL] Pipeline failed on {GAME_NAME}!")
        print(f">>> Error Details: {e}")
        print(f">>> Skipping {GAME_NAME} and continuing to next game...")

print("\n" + "=" * 50)
print(" GLOBAL PIPELINE COMPLETE")
print("=" * 50)

