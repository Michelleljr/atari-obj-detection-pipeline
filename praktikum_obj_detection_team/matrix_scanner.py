import jax
import jaxatari
from jaxatari.wrappers import PixelAndObjectObsWrapper, AtariWrapper

#  Start the game up
print("Booting Matrix Scanner...")
base_env = jaxatari.make("spaceinvaders")  # game to be scanned
atari_env = AtariWrapper(base_env)
env = PixelAndObjectObsWrapper(atari_env)

# Get exactly ONE frame
rng = jax.random.PRNGKey(42)
rng, reset_key = jax.random.split(rng)
current_obs, state = env.reset(reset_key)

image_stack, obs_stack = current_obs
if hasattr(obs_stack, '_asdict'):
    objects_dict = obs_stack._asdict()
else:
    objects_dict = obs_stack.__dict__

#  Execute the "role call" (what roles are there)
print("\n" + "=" * 40)
print(" 1. THE ROLL CALL (ALL OBJECTS)")
print("=" * 40)
for obj_name in objects_dict.keys():
    print(f"- {obj_name}")

# Execute the "X-Ray" (deep diver per role)
print("\n" + "=" * 40)
print(" 2. THE X-RAY (OBJECT PROPERTIES)")
print("=" * 40)
for obj_name, obj_data in objects_dict.items():
    # Find the variables hidden inside the object
    if isinstance(obj_data, dict):
        keys = obj_data.keys()
    else:
        # Filter out built-in Python junk (like __class__) so it's readable
        keys = [k for k in dir(obj_data) if not k.startswith('_')]

    print(f"[{obj_name}] contains: {list(keys)}")

print("\nScanner complete. No files were written.")