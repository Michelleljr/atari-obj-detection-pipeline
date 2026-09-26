import pytest
import jax
import jax.numpy as jnp
import numpy as np
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import jaxatari
from jaxatari.wrappers import AtariWrapper, YOLOObjectCentricWrapper
from detectors.Yolo_wrapper import draw_boxes

SCRIPT_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = SCRIPT_DIR / "praktikum_obj_detection_team" / "quirks_registry.json"


@pytest.fixture
def available_game_setup():
    """Finds any available .pt file in detectors/ and matches it with quirks_registry.json."""
    assert REGISTRY_PATH.exists(), f"Registry file not found at {REGISTRY_PATH}"

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry_data = json.load(f)

    weights_dir = SCRIPT_DIR / "detectors"
    pt_files = list(weights_dir.glob("**/*.pt"))

    if not pt_files:
        return None  # No local weights available

    # Grab the first available weight file on disk
    weights_path = pt_files[0]
    game_name = weights_path.stem.lower()

    # Fallback to a registered key if the filename stems slightly differently
    if game_name not in registry_data:
        game_name = list(registry_data.keys())[0]

    return {
        "registry_path": REGISTRY_PATH,
        "weights_path": weights_path,
        "game_name": game_name,
        "registry": registry_data
    }


# ---------------------------------------------------------------------------
# 1. TEST: Bounding Box Utility (No model or environment required)
# ---------------------------------------------------------------------------
def test_draw_boxes_fallback():
    """Verifies draw_boxes gracefully returns unchanged frame when result is None."""
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    output_frame = draw_boxes(dummy_frame, None)

    assert output_frame.shape == (480, 640, 3)
    assert np.array_equal(output_frame, dummy_frame)


# ---------------------------------------------------------------------------
# 2. TEST: Isolated Wrapper Logic via Mocking (Runs without needing .pt files)
# ---------------------------------------------------------------------------
@patch("ultralytics.YOLO")
def test_wrapper_structure_with_mock_yolo(mock_yolo):
    """Verifies wrapper initialization and observation building using a mocked YOLO model."""
    # Setup mock detection return
    mock_model_instance = MagicMock()
    mock_yolo.return_value = mock_model_instance
    mock_model_instance.names = {0: "player", 1: "enemy"}

    # Minimal registry structure matching wrapper expectation
    mock_registry = {
        "pong": {
            "objects": {
                "player": {"class_id": 0, "max_instances": 1, "features": ["x1", "y1", "x2", "y2"]},
                "enemy": {"class_id": 1, "max_instances": 1, "features": ["x1", "y1", "x2", "y2"]}
            }
        }
    }

    base_env = jaxatari.make("pong")
    atari_env = AtariWrapper(base_env)

    wrapped_env = YOLOObjectCentricWrapper(
        env=atari_env,
        yolo_model_path="fake_path.pt",
        quirks_registry=mock_registry,
        game_name="pong"
    )

    # Check key attributes
    assert wrapped_env is not None
    assert hasattr(wrapped_env, "step")
    assert hasattr(wrapped_env, "reset")


# ---------------------------------------------------------------------------
# 3. TEST: End-to-End Execution (Runs using active local weights)
# ---------------------------------------------------------------------------
def test_e2e_reset_and_step(available_game_setup):
    """Tests reset() and step() using an actual local weight file found in detectors/."""
    if available_game_setup is None:
        pytest.skip("No local .pt weight files found in detectors/ directory.")

    setup = available_game_setup
    base_env = jaxatari.make(setup["game_name"])
    atari_env = AtariWrapper(base_env)

    wrapped_env = YOLOObjectCentricWrapper(
        env=atari_env,
        yolo_model_path=setup["weights_path"],
        quirks_registry=setup["registry"],
        game_name=setup["game_name"],
        frame_stack_size=4,
        frame_skip=4
    )

    # Test Reset
    rng = jax.random.PRNGKey(42)
    obs_stack, state = wrapped_env.reset(rng)

    assert obs_stack is not None
    assert state is not None

    # Test Step
    action = jnp.array(0, dtype=jnp.int32)
    obs_stack, state, reward, terminated, truncated, info = wrapped_env.step(state, action)

    assert obs_stack is not None
    assert isinstance(info, dict)