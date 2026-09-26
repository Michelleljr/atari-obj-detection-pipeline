import unittest
import json
from pathlib import Path
import jax
import jaxatari
from jaxatari.wrappers import AtariWrapper, YOLOObjectCentricWrapper

SCRIPT_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = SCRIPT_DIR / "praktikum_obj_detection_team" / "quirks_registry.json"
WEIGHTS_DIR = SCRIPT_DIR / "detectors" / "YOLOv8nano" / "weights"

ASSIGNED_GAMES = [
    'freeway', 'frostbite', 'galaxian', 'hauntedhouse',
    'humancannonball', 'kangaroo', 'kingkong', 'lasergates',
    'montezumarevenge', 'mspacman', 'namethisgame', 'phoenix', 'pong'
]


class TestYOLOPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        assert REGISTRY_PATH.exists(), f"Registry missing at {REGISTRY_PATH}"
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            cls.registry = json.load(f)

    def test_registry_contains_assigned_games(self):
        """Ensure all assigned games exist in quirks_registry.json."""
        for game in ASSIGNED_GAMES:
            with self.subTest(game=game):
                self.assertIn(game, self.registry, f"{game} missing from registry!")

    def test_wrapper_initialization_and_step(self):
        """Test wrapper reset and 1-step execution on a lightweight sample game."""
        test_game = "pong"
        weights_path = WEIGHTS_DIR / f"{test_game}.pt"

        if not weights_path.exists():
            self.skipTest(f"Weights for {test_game} not found at {weights_path}")

        base_env = jaxatari.make(test_game)
        atari_env = AtariWrapper(base_env)
        wrapped_env = YOLOObjectCentricWrapper(
            env=atari_env,
            yolo_model_path=weights_path,
            quirks_registry=self.registry,
            game_name=test_game,
            frame_stack_size=4,
            frame_skip=4,
            autoreset=True
        )

        rng = jax.random.PRNGKey(42)
        obs, state = wrapped_env.reset(rng)
        self.assertIsNotNone(obs)

        # Step environment
        action = jax.numpy.array(0, dtype=jax.numpy.int32)
        next_obs, next_state, reward, term, trunc, info = wrapped_env.step(state, action)
        self.assertIsNotNone(next_obs)


if __name__ == "__main__":
    unittest.main()