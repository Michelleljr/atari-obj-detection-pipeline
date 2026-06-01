import os
import cv2
import numpy as np
import jax
import jaxatari
from jaxatari.wrappers import PixelAndObjectObsWrapper, AtariWrapper

GAME_NAME = "spaceinvaders"
TARGET_FRAMES = 10
DEBUG_MODE = True #toggle to false if used purely for frame extraction

#folder setup
output_folder = f"captured_{TARGET_FRAMES}frames_{GAME_NAME}"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#start the game up
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
total_extracted_objects = 0

# all roles found fromm the matrix_scanner
space_invaders_mapping = {
    "player": 0,
    "enemies": 1,
    "player_bullet": 2,
    "enemy_bullets": 2,
    "ufo": 3,
    "barricade_health": 4
}

pong_mapping = {
    "player": 5,
    "enemy": 6,
    "ball": 7
}

# Automatically select the right dictionary
if GAME_NAME == "spaceinvaders":
    class_mapping = space_invaders_mapping
elif GAME_NAME == "pong":
    class_mapping = pong_mapping
else:
    raise ValueError("Game mapping not found!")

print("Starting Extraction...")

while saved_count < TARGET_FRAMES:
    rng, action_key = jax.random.split(rng)
    action = jax.random.randint(action_key, shape=(), minval=0, maxval=18)

    current_obs, state, reward, stopped, truncated, info = env.step(state, action)
    frame_count += 1

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

            # --- THE FINAL, CLEAN PARSER ---
        for obj_name, obj_data in objects_dict.items():
            # -----------------------------------------------------
            # THE DEBUG ENGINE: Uncomment these lines if a new game
            # breaks the parser to expose the underlying JAX keys and data within the specific object.(x-ray)
            # -----------------------------------------------------
            # if isinstance(obj_data, dict):
            #     keys = obj_data.keys()
            # else:
            #     keys = dir(obj_data)
            # print(f"DEBUG [{obj_name}]: {list(keys)}")
            # -----------------------------------------------------

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

                #if GAME_NAME == "spaceinvaders" and class_id == 0:
                    x = x - 7.5

                #for quirk detection
                if DEBUG_MODE:
                    # Draw boxes directly on image
                    # (255, 0, 255) is Hot Pink in BGR format. Thickness is set to 2.
                    cv2.rectangle(frame_bgr, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 255), 2)

                # YOLO Math (Normalized 0.0 to 1.0)
                x_center = (x + (w / 2.0)) / 160.0
                y_center = (y + (h / 2.0)) / 210.0
                w_norm = w / 160.0
                h_norm = h / 210.0



                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        with open(f"{base_filename}.txt", "w") as f:
            f.write("\n".join(yolo_lines))

        cv2.imwrite(f"{base_filename}.png", frame_bgr)

        saved_count += 1
        print(f"Auto-generated pair {saved_count}/{TARGET_FRAMES}: {len(yolo_lines)} objects found!")

    if stopped or truncated:
        rng, reset_key = jax.random.split(rng)
        current_obs, state = env.reset(reset_key)

print("\n" + "="*50)
print(" PIPELINE COMPLETE")
print("="*50)