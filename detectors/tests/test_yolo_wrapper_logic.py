import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import jax.numpy as jnp

from jaxatari.wrappers import YOLOObjectCentricWrapper


def fake_box(class_id, confidence, xyxy):
    """Minimal substitute for a YOLO box."""
    return SimpleNamespace(
        cls=np.array(class_id),
        conf=np.array(confidence),
        xyxy=np.array([xyxy], dtype=np.float32),
    )


class TestYOLOWrapperLogic(unittest.TestCase):

    def setUp(self):
        # bypass initialization so no weights or real game are needed.
        self.wrapper = YOLOObjectCentricWrapper.__new__(
            YOLOObjectCentricWrapper
        )

        self.wrapper.conf_threshold = 0.40
        self.wrapper.imgsz = 640
        self.wrapper.iou_threshold = 0.50
        self.wrapper.device = "cpu"

        # 2 classes, each with [x, y, width, height, present].
        self.wrapper.class_offsets = {0: 0, 1: 5}
        self.wrapper.class_slot_sizes = {0: 5, 1: 5}
        self.wrapper.num_features = 10

        self.wrapper.frame_stack_size = 4
        self.wrapper.frame_skip = 4
        self.wrapper.clip_reward = True
        self.wrapper.autoreset = False

        self.wrapper.model = Mock()
        self.wrapper._env = Mock()
        self.frame = np.zeros((210, 160, 3), dtype=np.uint8)

    def set_detections(self, boxes):
        result = SimpleNamespace(boxes=boxes)
        self.wrapper.model.predict.return_value = [result]
        return result

    def make_state(self):
        stack = jnp.arange(40, dtype=jnp.float32).reshape(4, 10)
        return SimpleNamespace(
            atari_state=SimpleNamespace(
                env_state="initial",
                key=jnp.array([0, 42], dtype=jnp.uint32),
            ),
            obs_stack=stack,
        )

    def configure_step(self, rewards, terminated=False, truncated=False):
        next_state = SimpleNamespace(
            env_state="next",
            key=jnp.array([0, 43], dtype=jnp.uint32),
        )

        # end the episode on the last supplied step if requested.
        self.wrapper._env.step.side_effect = [
            (
                None,
                next_state,
                reward,
                terminated and i == len(rewards) - 1,
                truncated and i == len(rewards) - 1,
                {},
            )
            for i, reward in enumerate(rewards)
        ]

        newest = np.full(10, 99, dtype=np.float32)
        self.wrapper._prep_frame = Mock(return_value=self.frame)
        self.wrapper._detections_to_flat_obs = Mock(
            return_value=(newest, "detections")
        )
        return newest, next_state

    def test_empty_detections_return_zero_observation(self):
        self.set_detections([])

        flat, _ = self.wrapper._detections_to_flat_obs(self.frame)

        np.testing.assert_array_equal(flat, np.zeros(10))
        self.assertEqual(flat.dtype, np.float32)

    def test_highest_confidence_box_wins_for_same_class(self):
        # supply lower confidence first.
        self.set_detections([
            fake_box(0, 0.50, [0, 0, 10, 10]),
            fake_box(0, 0.90, [10, 20, 30, 60]),
        ])

        flat, _ = self.wrapper._detections_to_flat_obs(self.frame)

        # Center=(20,40), width=20, height=40, present=1.
        np.testing.assert_array_equal(
            flat[:5], [20, 40, 20, 40, 1]
        )
        np.testing.assert_array_equal(flat[5:], np.zeros(5))

    def test_unknown_and_low_confidence_boxes_are_ignored(self):
        self.set_detections([
            fake_box(0, 0.39, [10, 20, 30, 60]),
            fake_box(99, 0.99, [10, 20, 30, 60]),
        ])

        flat, _ = self.wrapper._detections_to_flat_obs(self.frame)

        np.testing.assert_array_equal(flat, np.zeros(10))

    def test_classes_fill_the_correct_slots(self):
        self.set_detections([
            fake_box(1, 0.80, [40, 50, 60, 70]),
            fake_box(0, 0.70, [10, 20, 30, 60]),
        ])

        flat, _ = self.wrapper._detections_to_flat_obs(self.frame)

        np.testing.assert_array_equal(
            flat,
            [20, 40, 20, 40, 1, 50, 60, 20, 20, 1],
        )

    def test_step_shifts_stack_and_appends_new_observation(self):
        state = self.make_state()
        newest, _ = self.configure_step([0, 0, 0, 0])

        obs, new_state, *_ = self.wrapper.step(state, 0)

        np.testing.assert_array_equal(
            np.asarray(obs)[:-1],
            np.asarray(state.obs_stack)[1:],
        )
        np.testing.assert_array_equal(np.asarray(obs)[-1], newest)
        np.testing.assert_array_equal(
            np.asarray(new_state.obs_stack), np.asarray(obs)
        )
        self.assertEqual(self.wrapper._env.step.call_count, 4)

    def test_reward_is_clipped_after_accumulation(self):
        self.configure_step([2, -1, 2, 0])

        _, _, reward, *_ = self.wrapper.step(self.make_state(), 0)

        self.assertEqual(reward, 1.0)

    def test_reward_is_not_clipped_when_disabled(self):
        self.wrapper.clip_reward = False
        self.configure_step([2, -1, 2, 0])

        _, _, reward, *_ = self.wrapper.step(self.make_state(), 0)

        self.assertEqual(reward, 3.0)

    def test_frame_skip_stops_at_termination(self):
        self.configure_step([0, 2], terminated=True)

        _, _, reward, terminated, truncated, _ = self.wrapper.step(
            self.make_state(), 0
        )

        self.assertEqual(self.wrapper._env.step.call_count, 2)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, 1.0)

    def test_autoreset_returns_fresh_observation(self):
        self.wrapper.autoreset = True
        _, terminal_state = self.configure_step(
            [1], terminated=True
        )

        reset_obs = jnp.zeros((4, 10), dtype=jnp.float32)
        reset_state = object()
        self.wrapper.reset = Mock(
            return_value=(reset_obs, reset_state)
        )

        obs, state, reward, terminated, _, _ = self.wrapper.step(
            self.make_state(), 0
        )

        self.wrapper.reset.assert_called_once()
        np.testing.assert_array_equal(
            np.asarray(self.wrapper.reset.call_args.args[0]),
            np.asarray(terminal_state.key),
        )
        self.assertIs(state, reset_state)
        np.testing.assert_array_equal(np.asarray(obs), reset_obs)
        self.assertTrue(terminated)
        self.assertEqual(reward, 1.0)

        # no detection should run on the terminal frame after autoreset.
        self.wrapper._detections_to_flat_obs.assert_not_called()


if __name__ == "__main__":
    unittest.main()