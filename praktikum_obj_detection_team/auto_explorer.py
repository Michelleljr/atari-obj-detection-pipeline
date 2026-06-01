import os
import json
import jax
import jaxatari
from jaxatari.wrappers import PixelAndObjectObsWrapper, AtariWrapper

ATARI_57 = [
    "alien", "amidar", "asterix", "asteroids", "atlantis",
    "bankheist", "beamrider", "berzerk", "blackjack", "breakout",
    "centipede", "choppercommand", "enduro", "fishingderby", "flagcapture",
    "freeway", "frostbite", "galaxian", "gravitar", "hauntedhouse",
    "humancannonball", "kangaroo", "kingkong", "lasergates", "montezumarevenge",
    "mspacman", "namethisgame", "phoenix", "pong", "qbert",
    "riverraid", "seaquest", "sirlancelot", "skiing", "slotmachine",
    "spaceinvaders", "spacewar", "tennis", "tetris", "timepilot",
    "tron", "turmoil", "venture", "videocube", "videopinball",
    "wordzapper"
]

TARGET_FRAMES = 500

master_registry = {}

for game_name in ATARI_57:
    print(f"\n{'=' * 40}")
    print(f" SCANNING: {game_name.upper()}")
    print(f"{'=' * 40}")

    try:
        # start up the game
        base_env = jaxatari.make(game_name)
        atari_env = AtariWrapper(base_env)
        env = PixelAndObjectObsWrapper(atari_env)

        # PRNG Setup
        rng = jax.random.PRNGKey(42)
        rng, reset_key = jax.random.split(rng)
        current_obs, state = env.reset(reset_key)

        frame_count = 0
        discovered_objects = set()

        while frame_count < TARGET_FRAMES:
            rng, action_key = jax.random.split(rng)
            action = jax.random.randint(action_key, shape=(), minval=0, maxval=18)

            # Take a step in the environment
            current_obs, state, reward, stopped, truncated, info = env.step(state, action)
            frame_count += 1

            # Split the JAX tuple
            image_stack, obs_stack = current_obs

            # Force it into dictionary
            if hasattr(obs_stack, '_asdict'):
                objects_dict = obs_stack._asdict()
            else:
                objects_dict = obs_stack.__dict__

            # The Memory Filter
            for obj_name, obj_data in objects_dict.items():
                if obj_name not in discovered_objects:
                    discovered_objects.add(obj_name)
                    print(f"[{frame_count}/{TARGET_FRAMES}] New Object Discovered: {obj_name}")

            # if game ends/player dies, reset the game
            if stopped or truncated:
                rng, reset_key = jax.random.split(rng)
                current_obs, state = env.reset(reset_key)

        print(f"\nExploration Over! Found {len(discovered_objects)} unique objects.")


        # Loop through your discovered set and attach the default math
        master_registry[game_name] = {"objects": {}}

        for obj_name in discovered_objects:
            master_registry[game_name]["objects"][obj_name] = {
                "class_id": "TODO",
                "x_offset": 0,
                "y_offset": 0
            }

    except Exception as e:
        print(f">>> [CRITICAL] Pipeline failed on {game_name}!")
        print(f">>> Error Details: {e}")
        print(f">>> Skipping {game_name} and continuing pipeline...")
        continue  # Abort this specific game and jump to the next one

        # 3. The Incremental Save (Safely outside the try/except, but inside the loop)
    with open("quirks_registry.json", "w") as outfile:
        json.dump(master_registry, outfile, indent=4)

    print(f"Registry successfully updated with {game_name}!")

    print(f"\nPIPELINE COMPLETE! The quirks_registry.json file is fully populated with {game_name} information.")