import time
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import jaxatari
import numpy as np
from jaxatari.wrappers import AtariWrapper, YOLOObjectCentricWrapper

# --- Paths Setup ---
SCRIPT_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = SCRIPT_DIR.parent / "praktikum_obj_detection_team" / "quirks_registry.json"
WEIGHTS_DIR = SCRIPT_DIR.parent / "detectors" / "YOLOv8nano" / "weights"

TARGET_GAMES = ["pong", "mspacman", "spaceinvaders"]


def load_registry(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


registry = load_registry(REGISTRY_PATH)


def evaluate_local_wrapper_latency(game_name, num_steps=500):
    """Measures pure runtime execution overhead inside YOLOObjectCentricWrapper."""
    model_path = WEIGHTS_DIR / f"{game_name}.pt"
    if not model_path.exists():
        print(f"[SKIP] Weight file missing for '{game_name}' at: {model_path}")
        return None

    base_env = jaxatari.make(game_name)
    atari_env = AtariWrapper(base_env)

    wrapped_env = YOLOObjectCentricWrapper(
        env=atari_env,
        yolo_model_path=model_path,
        quirks_registry=registry,
        game_name=game_name,
        frame_stack_size=4,
        frame_skip=4,
        conf_threshold=0.40,  # Operational threshold
        imgsz=640,
        iou_threshold=0.50,
        autoreset=True,
    )

    rng = jax.random.PRNGKey(42)
    rng, reset_key = jax.random.split(rng)
    obs_stack, state = wrapped_env.reset(reset_key)

    full_step_latencies = []
    num_actions = int(wrapped_env.action_space().n)

    # Warm-up step
    rng, action_key = jax.random.split(rng)
    action_val = jax.random.randint(action_key, shape=(), minval=0, maxval=num_actions)
    action = jnp.array(action_val, dtype=jnp.int32)
    obs_stack, state, reward, terminated, truncated, info = wrapped_env.step(state, action)

    wall_clock_start = time.perf_counter()

    for _ in range(num_steps):
        rng, action_key = jax.random.split(rng)
        action_val = jax.random.randint(action_key, shape=(), minval=0, maxval=num_actions)
        action = jnp.array(action_val, dtype=jnp.int32)

        # Time total end-to-end wrapper step execution
        t0 = time.perf_counter()
        obs_stack, state, reward, terminated, truncated, info = wrapped_env.step(state, action)
        t1 = time.perf_counter()

        full_step_latencies.append((t1 - t0) * 1000.0)

    total_wall_time = time.perf_counter() - wall_clock_start
    avg_step_latency = float(np.mean(full_step_latencies))
    effective_fps = (num_steps * wrapped_env.frame_skip) / total_wall_time

    return {
        "game": game_name,
        "wrapper_step_ms": avg_step_latency,
        "fps": effective_fps,
        "frame_skip": wrapped_env.frame_skip
    }


if __name__ == "__main__":
    results = []
    print("\n" + "=" * 65)
    print(" BENCHMARKING END-TO-END WRAPPER SYSTEM OVERHEAD ")
    print("=" * 65)

    for game in TARGET_GAMES:
        res = evaluate_local_wrapper_latency(game, num_steps=500)
        if res:
            results.append(res)

    print("\n" + "=" * 65)
    print(f"{'Target Game':<16} | {'Wrapper Step (ms)':<18} | {'Throughput (FPS)':<16} | {'Frame Skip (k)':<12}")
    print("-" * 65)
    for r in results:
        print(f"{r['game']:<16} | {r['wrapper_step_ms']:<18.2f} | {r['fps']:<16.2f} | {r['frame_skip']:<12}")
    print("=" * 65)