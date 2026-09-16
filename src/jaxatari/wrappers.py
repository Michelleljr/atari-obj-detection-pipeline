"""Jaxatari Wrappers"""
from absl.logging import info

import functools
import types
import warnings
from typing import Any, Dict, Tuple, Union, Optional, Callable
from dataclasses import is_dataclass, asdict

import chex
import cv2
from flax import struct
import jax
import jax.image as jim
import jax.numpy as jnp
from jax import flatten_util
import torch
from jaxatari.environment import EnvState, JAXAtariAction as Action
import jaxatari.spaces as spaces
import numpy as np
from jaxatari.rendering.jax_rendering_utils import RendererConfig

from ultralytics import YOLO

class JaxatariWrapper(object):
    """Base class for JAXAtark wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)
    
class MultiRewardWrapper(JaxatariWrapper):
    """
    Allows providing multiple reward functions to be computed at every step.
    Apply this wrapper directly after the base environment, before any other wrappers.
    """

    def __init__(self, env, reward_funcs: list[Callable]):
        super().__init__(env)
        assert isinstance(reward_funcs, list) and len(reward_funcs) > 0, "reward_funcs must be a non-empty list of callables" 
        self._reward_funcs = reward_funcs

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: EnvState, state: EnvState) -> chex.Array:
        """Compute multiple rewards based on the provided reward functions."""
        if self._reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self._reward_funcs]
        )
        return rewards

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: EnvState, action: int) -> Tuple[chex.Array, EnvState, float, bool, Dict]: 
        obs, new_state, reward, done, info = self._env.step(state, action)
        all_rewards = self._get_all_rewards(state, new_state)
        # Convert info to dict: handle NamedTuple (has _asdict) or dataclass (use asdict)
        if hasattr(info, '_asdict'):
            info = info._asdict()
        elif is_dataclass(info):
            info = asdict(info)
        info["all_rewards"] = all_rewards
        return obs, new_state, reward, done, info 

@struct.dataclass
class AtariState:
    env_state: Any
    key: chex.PRNGKey
    step: int
    prev_action: int
    
class AtariWrapper(JaxatariWrapper):
    """
    Wrapper for Atari environments that applies Atari-specific control logic.
    Returns single-step, single-frame observations from the wrapped base env.
    Args:
        env: The environment to wrap.
        sticky_actions: Sticky action probability in [0, 1]. Defaults to 0.25.
        episodic_life: Loss of life -> terminated. Does not reset the environment. Defaults to True.
        first_fire: Take FIRE action on reset. Defaults to True.
        noop_max: Max number of no-op actions to take on reset. Defaults to 30.
        full_action_space: Use full action space of 18 actions. Defaults to False (minimal action set).
        max_frames_per_episode: Maximum number of frames per episode before truncation. Defaults to 108,000 (30 minutes at 60fps).

    Note: Typically, this wrapper is followed by PixelObsWrapper, ObjectCentricWrapper or PixelAndObjectCentricWrapper.
    Frame-skipping, max-pooling, frame-stacking and reward clipping are handled in those.
    """
    def __init__(self, env, sticky_actions: float = 0.25, episodic_life: bool = True, first_fire: bool = True, noop_max: int = 30, full_action_space: bool = False, max_frames_per_episode: int = 108_000):
        super().__init__(env)
        self._env = env
        self.sticky_actions = float(np.clip(sticky_actions, 0.0, 1.0))
        self.episodic_life = episodic_life
        self.first_fire = first_fire
        self.noop_reset = False if noop_max == 0 else True
        self.noop_max = noop_max
        self.full_action_space = full_action_space
        self.max_frames_per_episode = max_frames_per_episode

        # --- 1) HANDLE FULL ACTION SPACE LOGIC ---
        # If requested, swap the environment's (minimal) action set for the full identity set.
        # This keeps each game env "clean" while enabling a central switch for experimentation.
        if self.full_action_space and hasattr(self._env, "ACTION_SET"):
            # Overwrite the instance attribute with [0, 1, ... 17]
            self._env.ACTION_SET = jnp.arange(18, dtype=jnp.int32)

        # --- 2) RESOLVE CORRECT 'FIRE' ACTION INDEX ---
        # The wrapped env expects an *index* into ACTION_SET (agent action), not the ALE action constant.
        self.fire_action_index: int = int(Action.FIRE)  # fallback if env doesn't expose ACTION_SET
        self.first_fire = first_fire

        if hasattr(self._env, "ACTION_SET"):
            # Convert to numpy for search (safe in __init__)
            action_set_np = np.array(self._env.ACTION_SET)
            fire_indices = np.where(action_set_np == int(Action.FIRE))[0]
            if len(fire_indices) > 0:
                self.fire_action_index = int(fire_indices[0])
            else:
                # Game has no FIRE action (e.g. Freeway).
                # Disable first_fire to prevent sending a random command by mistake.
                self.first_fire = False

        self._observation_space = self._env.observation_space()

    def observation_space(self) -> spaces.Space:
        """Returns the single-frame base observation space."""
        return self._observation_space
    
    def image_space(self) -> spaces.Box:
        """Returns the image space."""
        return self._env.image_space()

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, AtariState]:
        # Split keys for all potential random operations
        env_key, wrapper_key, noop_key = jax.random.split(key, 3)
        obs, env_state = self._env.reset(env_key)
        step = jnp.array(0, dtype=jnp.int32)
        prev_action = jnp.array(0, dtype=jnp.int32)

        # ========== NOOP RESET ==========
        def perform_noop_reset(carry):
            # This function will be executed if self.noop_reset is True
            env_state, obs, step = carry
            # Generate the random number of no-op steps to take.
            num_noops = jax.random.randint(noop_key, shape=(), minval=0, maxval=self.noop_max + 1)

            def noop_body_fn(i, loop_carry):
                current_obs, current_env_state = loop_carry
                # We always compute the next step for static graph tracing...
                next_obs, next_env_state, reward, done, info = self._env.step(current_env_state, Action.NOOP)
                # Might be done from the no-op step, in which case we reset. 
                next_obs, next_env_state = jax.lax.cond(
                    done,
                    lambda: self._env.reset(env_key),
                    lambda: (next_obs, next_env_state) 
                )
                return next_obs, next_env_state

            # Loop for the static maximum number of no-ops.
            final_obs, final_env_state = jax.lax.fori_loop(0, num_noops, noop_body_fn, (obs, env_state)) 
            
            # Update the step counter by the dynamic number of no-ops performed.
            final_step = step + num_noops
            return final_env_state, final_obs, final_step

        # Use lax.cond to conditionally apply the whole no-op block based on the static self.noop_reset flag.
        env_state, obs, step = jax.lax.cond(
            self.noop_reset,
            lambda carry: perform_noop_reset(carry),
            lambda carry: carry,
            (env_state, obs, step)
        )

        # ========== FIRST FIRE ==========
        def perform_first_fire(carry):
            env_state, obs, step, _ = carry
            fire_obs, fire_env_state, _, _, _ = self._env.step(env_state, self.fire_action_index)
            return fire_env_state, fire_obs, step + 1, self.fire_action_index

        def identity_fire(carry):
            return carry
        
        # Conditionally apply the fire action based on the static self.first_fire flag.
        env_state, obs, step, prev_action = jax.lax.cond(
            self.first_fire,
            perform_first_fire,
            identity_fire,
            (env_state, obs, step, prev_action)
        )

        return obs, AtariState(env_state, wrapper_key, step, prev_action)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: AtariState, action: int) -> Tuple[chex.Array, AtariState, float, bool, bool, Dict[Any, Any]]:
        step_key, next_state_key = jax.random.split(state.key)

        use_sticky_action = jax.random.uniform(step_key, shape=()) < self.sticky_actions
        new_action = jnp.where(use_sticky_action, state.prev_action, action)
        obs, new_env_state, reward, env_done, infos = self._env.step(state.env_state, new_action)

        terminated = env_done
        if self.episodic_life:
            # If the player has lost a life, we consider the episode done
            if hasattr(state.env_state, "lives"):
                condition = jnp.logical_and(state.env_state.lives > 0, new_env_state.lives < state.env_state.lives)
                terminated = jnp.logical_or(terminated, condition)
            elif hasattr(state.env_state, "lives_lost"):
                terminated = jnp.logical_or(terminated, new_env_state.lives_lost > state.env_state.lives_lost)

        if hasattr(infos, '_asdict'):
            # It's a namedtuple or similar, convert to dict
            info_items = infos._asdict().items()
        elif is_dataclass(infos):
            # It's a dataclass, convert to dict
            info_items = asdict(infos).items()
        else:
            # It's already a dict
            info_items = infos.items()

        info_dict = {k: v for k, v in info_items}

        next_state = AtariState(new_env_state, next_state_key, state.step + 1, new_action)

        # store actual done - not affected by episodic life
        info_dict["env_done"] = env_done

        # store actual reward in info dict before clipping
        info_dict["env_reward"] = reward

        truncated = (state.step + 1 >= self.max_frames_per_episode)
        return obs, next_state, reward, terminated, truncated, info_dict


@struct.dataclass
class ObjectCentricState:
    atari_state: AtariState
    obs_stack: jax.Array

class ObjectCentricWrapper(JaxatariWrapper):
    """
    Wrapper for Atari environments that returns stacked object-centric observations.
    The output observation is a 2D array of shape (frame_stack_size, num_features).
    Apply this wrapper after the AtariWrapper!
    """

    def __init__(self, env, frame_stack_size: int = 4, frame_skip: int = 4, clip_reward: bool = True):
        super().__init__(env)
        assert isinstance(env, AtariWrapper), "ObjectCentricWrapper must be applied after AtariWrapper"
        self.frame_stack_size = frame_stack_size
        self.frame_skip = frame_skip
        self.clip_reward = clip_reward

        # Calculate exact bounds for the flattened single-frame observation.
        single_frame_space = self._env.observation_space()
        lows, highs = [], []

        for leaf_space in jax.tree.leaves(single_frame_space):
            if isinstance(leaf_space, spaces.Box):
                low_arr = np.broadcast_to(leaf_space.low, leaf_space.shape).flatten()
                high_arr = np.broadcast_to(leaf_space.high, leaf_space.shape).flatten()
                lows.append(low_arr)
                highs.append(high_arr)
            elif isinstance(leaf_space, spaces.Discrete):
                lows.append(np.array([0], dtype=leaf_space.dtype))
                highs.append(np.array([leaf_space.n - 1], dtype=leaf_space.dtype))
            else:
                raise TypeError(f"Unsupported space type for flattening: {type(leaf_space)}")

        if not lows:
            raise ValueError("The observation space appears to be empty or contain unsupported types.")

        single_frame_lows = np.concatenate(lows)
        single_frame_highs = np.concatenate(highs)

        self._observation_space = spaces.Box(
            low=single_frame_lows,
            high=single_frame_highs,
            shape=(self.frame_stack_size, int(single_frame_lows.shape[0])),
            dtype=jnp.float32
        )

    def observation_space(self) -> spaces.Box:
        """Returns a Box space for the flattened observation."""
        return self._observation_space

    @functools.partial(jax.jit, static_argnums=(0,))
    def _flatten_single_obs(self, obs):
        """Flatten a single object-centric observation using ravel_pytree."""
        return flatten_util.ravel_pytree(obs)[0].astype(jnp.float32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey
    ) -> Tuple[chex.Array, ObjectCentricState]:
        obs, atari_state = self._env.reset(key)
        flat_obs = self._flatten_single_obs(obs)
        obs_stack = jnp.stack([flat_obs] * self.frame_stack_size)
        return obs_stack, ObjectCentricState(atari_state, obs_stack)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: ObjectCentricState,
        action: int,
    ) -> Tuple[chex.Array, ObjectCentricState, float, bool, bool, Dict[Any, Any]]:

        def body_fn(carry, _):
            atari_state, action = carry
            obs, new_atari_state, reward, terminated, truncated, info = self._env.step(atari_state, action)
            return (new_atari_state, action), (obs, reward, terminated, truncated, info)

        (atari_state, _), (obs, rewards, terminations, truncations, infos) = jax.lax.scan(
            body_fn,
            (state.atari_state, action),
            None,
            length=self.frame_skip,
        )

        latest_obs = jax.tree.map(lambda x: x[-1], obs)
        flat_latest_obs = self._flatten_single_obs(latest_obs)
        obs_stack = jnp.concatenate([state.obs_stack[1:], jnp.expand_dims(flat_latest_obs, axis=0)], axis=0)

        reward = jnp.sum(rewards)
        if self.clip_reward:
            reward = jnp.sign(reward)

        terminated = terminations.any()
        truncated = truncations.any()
        # Autoreset (gym's SAME_STEP mode) -> reset whole stack
        obs_stack, oc_state = jax.lax.cond(
            jnp.logical_or(infos["env_done"].any(), truncated),  # use actual env_done for reset condition, not affected by episodic life
            lambda: self.reset(atari_state.key),  # reset if done, using the current key for proper random state advancement
            lambda: (obs_stack, ObjectCentricState(atari_state, obs_stack)),  # step if not done
        )

        def reduce_info(k, v):
            if k in ["env_reward", "all_rewards"]:
                return jnp.sum(v, axis=0)
            if k == "env_done":
                return jnp.any(v)
            return v[-1]

        info_dict = {k: reduce_info(k, v) for k, v in infos.items()}
        return obs_stack, oc_state, reward, terminated, truncated, info_dict

@struct.dataclass
class YOLOObjectCentricState:
    atari_state: AtariState
    obs_stack: jax.Array
    last_frame: jax.Array
    last_results: Any  # ultralytics results object


class YOLOObjectCentricWrapper(JaxatariWrapper):
    """
    Wrapper for Atari environments that returns stacked object-centric observations
    derived from a YOLO detector run on rendered frames, instead of ground-truth
    game state. NOTE: unlike ObjectCentricWrapper/PixelObsWrapper, this wrapper is
    intentionally NOT jax.jit-compiled — it calls out to OpenCV and a PyTorch/
    ultralytics model per step, neither of which is traceable by JAX. Apply this
    wrapper directly after AtariWrapper.
    """

    def __init__(self, env, yolo_model_path, class_map, num_features,
                 frame_stack_size=4, frame_skip=4, conf_threshold=0.40,
                 imgsz=640, iou_threshold=0.50, display_size=None,
                 clip_reward=True, autoreset=True):
        super().__init__(env)
        assert isinstance(env, AtariWrapper), "YOLOObjectCentricWrapper must be applied directly after AtariWrapper"
        self.frame_stack_size = frame_stack_size
        self.frame_skip = frame_skip
        self.num_features = num_features
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.iou_threshold = iou_threshold
        self.display_size = display_size  # None = keep native 160x210; set a tuple to match your training resolution
        self.class_map = class_map
        self.clip_reward = clip_reward
        self.autoreset = autoreset
        self.device = 0 if torch.cuda.is_available() else "cpu"

        self.model = YOLO(yolo_model_path)
        self.model.to(self.device)

        self.last_frame = None
        self.last_results = None

        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.frame_stack_size, self.num_features),
            dtype=jnp.float32,
        )

    def observation_space(self) -> spaces.Box:
        return self._observation_space

    def _prep_frame(self, env_state):
        frame_rgb = np.array(self._env.render(env_state))
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if self.display_size is not None:
            frame_bgr = cv2.resize(frame_bgr, self.display_size, interpolation=cv2.INTER_NEAREST)
        return frame_bgr

    def _detections_to_flat_obs(self, frame_bgr):
        results = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=self.conf_threshold,
                                      iou=self.iou_threshold, device=self.device, verbose=False)[0]

        flat = np.zeros(self.num_features, dtype=np.float32)
        instance_counts = {cls: 0 for cls in self.class_map}

        boxes = sorted(results.boxes, key=lambda b: float(b.conf.item()), reverse=True)

        for box in boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            if conf < self.conf_threshold or cls not in self.class_map:
                continue

            slot_start, slot_size, max_instances = self.class_map[cls]
            idx = instance_counts[cls]
            if idx >= max_instances:
                continue

            offset = slot_start + idx * slot_size
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x, y = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            flat[offset:offset + min(slot_size, 5)] = [x, y, w, h, 1.0]
            instance_counts[cls] += 1

        return flat, results

    def reset(self, key: chex.PRNGKey):
        _, atari_state = self._env.reset(key)
        last_frame = self._prep_frame(atari_state.env_state)
        flat, last_results = self._detections_to_flat_obs(last_frame)

        obs_stack_np = np.stack([flat] * self.frame_stack_size)
        obs_stack = jnp.array(obs_stack_np)

        state = YOLOObjectCentricState(atari_state, obs_stack, last_frame, last_results)

        self.last_frame = last_frame
        self.last_results = last_results

        return obs_stack, state

    def step(self, state: YOLOObjectCentricState, action: int):
        atari_state = state.atari_state
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self.frame_skip):
            _, atari_state, reward, terminated, truncated, info = self._env.step(atari_state, action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        if self.clip_reward:
            total_reward = float(np.sign(total_reward))

        if self.autoreset and (bool(info.get("env_done", terminated)) or truncated):
            obs_stack, new_state = self.reset(atari_state.key)
            return obs_stack, new_state, total_reward, terminated, truncated, info

        last_frame = self._prep_frame(atari_state.env_state)
        flat, last_results = self._detections_to_flat_obs(last_frame)

        obs_stack_np = np.concatenate([np.array(state.obs_stack)[1:], flat[np.newaxis]], axis=0)
        obs_stack = jnp.array(obs_stack_np)

        new_state = YOLOObjectCentricState(atari_state, obs_stack, last_frame, last_results)

        self.last_frame = last_frame
        self.last_results = last_results

        return obs_stack, new_state, total_reward, terminated, truncated, info

class DinoV2ObjectCentricWrapper:
    """
    Drop-in replacement for ObjectCentricWrapper that uses zero-shot DINOv2
    attention-map detection instead of ground-truth game state or a trained model.

    Detection approach
    ------------------
    DINOv2's CLS-token self-attention maps are used to identify the most salient
    image patches.  The thresholded attention map is segmented with connected-
    component analysis to produce bounding boxes.  The ``max_objects``
    highest-salience regions are assigned to consecutive slots in order of
    salience (slot 0 = most salient).  No labels or prior knowledge about the
    game are required.

    Output format
    -------------
    Identical to ``YOLOObjectCentricWrapper``:
    ``(frame_stack_size, num_features)`` float32 array where each detected object
    occupies a slot of five values ``[x_centre, y_centre, width, height, confidence]``.

    Parameters
    ----------
    env :
        AtariWrapper-wrapped base env.  Wrapping in ``PixelObsWrapper`` first is
        the most convenient setup (pixel frames come back from ``step`` directly),
        but a bare ``AtariWrapper`` env also works via the env's ``render()`` method.
    num_features : int
        Total length of the flat feature vector per frame.
        Must be at least ``max_objects * 5``.
    frame_stack_size : int
        Number of consecutive observations to stack (default 4).
    frame_skip : int
        Number of AtariWrapper steps per ``step()`` call (default 4).
        Only the last frame is used for detection.
    dino_model_name : str
        HuggingFace model identifier for DINOv2
        (default ``"facebook/dinov2-base"``).
    attention_head_fusion : str
        How to combine multi-head attention maps.
        One of ``"mean"`` (default), ``"min"``, ``"max"``.
    attention_threshold : float
        Fraction of the maximum attention value used as the binary foreground
        threshold (default ``0.6``).  Higher → smaller, more precise masks.
    max_objects : int
        Maximum number of salient regions to detect (default 5).
    class_map : dict, optional
        ``{region_index: (slot_start, slot_size)}`` where region 0 is the most
        salient object.  If *None* a default layout is built automatically:
        region ``i`` → slot starting at ``i * 5``, size 5.
    device : str or None
        Torch device string (e.g. ``"cuda"``, ``"cpu"``).  Auto-detected when
        *None* (default).
    clip_reward : bool
        Clip reward to ``{-1, 0, +1}`` via ``sign`` (default ``True``).

    Example
    -------
    ::

        from jaxatari.wrappers import AtariWrapper, DinoV2ObjectCentricWrapper
        import jaxatari, jax

        env = DinoV2ObjectCentricWrapper(
            AtariWrapper(jaxatari.make("pong")),
            num_features=25,   # 5 objects × 5 features
            max_objects=5,
        )
        obs, state = env.reset(jax.random.PRNGKey(0))
        # obs.shape == (4, 25)
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        env,
        num_features: int,
        frame_stack_size: int = 4,
        frame_skip: int = 4,
        dino_model_name: str = "facebook/dinov2-base",
        attention_head_fusion: str = "mean",
        attention_threshold: float = 0.6,
        max_objects: int = 5,
        class_map: Optional[dict] = None,
        device: Optional[str] = None,
        clip_reward: bool = True,
    ):
        self._env = env
        self.num_features = num_features
        self.frame_stack_size = frame_stack_size
        self.frame_skip = frame_skip
        self.attention_head_fusion = attention_head_fusion
        self.attention_threshold = attention_threshold
        self.max_objects = max_objects
        self.clip_reward = clip_reward

        # Resolve torch device
        import torch
        self._device = "cuda" if (device is None and torch.cuda.is_available()) else (device or "cpu")

        # Build default class_map if none provided: region i → slot i*5, size 5
        self.class_map: dict = class_map if class_map is not None else {
            i: (i * 5, 5) for i in range(max_objects)
        }

        self._load_dino(dino_model_name)

        self._obs_shape = (self.frame_stack_size, self.num_features)

    # ------------------------------------------------------------------ #
    #  Model loading                                                       #
    # ------------------------------------------------------------------ #

    def _load_dino(self, model_name: str) -> None:
        """Load DINOv2 from HuggingFace Transformers and register an attention hook."""
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' package is required for DinoV2ObjectCentricWrapper. "
                "Install it with:  pip install transformers"
            ) from exc

        self._dino_processor = AutoImageProcessor.from_pretrained(model_name)
        self._dino_model = AutoModel.from_pretrained(model_name).to(self._device)
        self._dino_model.eval()

        # Patch size from config (14 px for ViT-B/14)
        self._patch_size: int = self._dino_model.config.patch_size

        # Attention weights from the last encoder layer, populated by a forward hook
        self._last_attention: Optional[np.ndarray] = None

        def _hook(module, inputs, output):
            # output shape: (batch, num_heads, seq_len, seq_len)
            self._last_attention = output.detach().cpu().float().numpy()

        self._dino_model.encoder.layer[-1].attention.attention.register_forward_hook(_hook)

    # ------------------------------------------------------------------ #
    #  Detection                                                           #
    # ------------------------------------------------------------------ #

    def _attention_to_boxes(
        self, frame_rgb: np.ndarray
    ) -> list:
        """
        Run DINOv2 on *frame_rgb* (H×W×3 uint8) and return a list of up to
        ``self.max_objects`` detections, each as
        ``(x_centre, y_centre, width, height, confidence)`` in pixel coords,
        sorted by descending confidence (= normalised attention mass).
        """
        import torch
        from PIL import Image
        import cv2  # used for connected components

        img_h, img_w = frame_rgb.shape[:2]
        pil_img = Image.fromarray(frame_rgb)

        inputs = self._dino_processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            # The hook fires inside this call and stores the attention weights
            _ = self._dino_model(**inputs, output_attentions=True)

        attn = self._last_attention  # (1, num_heads, seq_len, seq_len)
        if attn is None:
            return []

        # Number of patches along each spatial axis
        # seq_len = 1 (CLS) + num_patches  →  num_patches = seq_len - 1
        seq_len = attn.shape[-1]
        num_patches = seq_len - 1

        # Infer spatial grid from the preprocessed image size
        # The processor typically resizes to multiples of patch_size
        proc_h = inputs["pixel_values"].shape[-2]
        proc_w = inputs["pixel_values"].shape[-1]
        grid_h = proc_h // self._patch_size
        grid_w = proc_w // self._patch_size

        # Robustly handle cases where the number of patches doesn't perfectly
        # tile (e.g. due to padding).  Clamp to what the attention map has.
        if grid_h * grid_w > num_patches:
            grid_h = int(num_patches ** 0.5)
            grid_w = num_patches // grid_h

        # Extract CLS→patch attention (row 0, columns 1…)
        # Shape: (num_heads, num_patches)
        cls_attn = attn[0, :, 0, 1: grid_h * grid_w + 1]  # (heads, grid_h*grid_w)

        # Fuse across heads
        if self.attention_head_fusion == "mean":
            patch_attn = cls_attn.mean(axis=0)
        elif self.attention_head_fusion == "min":
            patch_attn = cls_attn.min(axis=0)
        else:  # "max"
            patch_attn = cls_attn.max(axis=0)

        # Reshape to spatial grid
        patch_attn = patch_attn[: grid_h * grid_w].reshape(grid_h, grid_w)

        # Normalise to [0, 1] for thresholding
        attn_min, attn_max = patch_attn.min(), patch_attn.max()
        if attn_max - attn_min > 1e-8:
            patch_attn = (patch_attn - attn_min) / (attn_max - attn_min)
        else:
            return []

        # Binary foreground mask
        mask = (patch_attn >= self.attention_threshold).astype(np.uint8)

        # Connected components → one box per connected region
        num_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(
            
        )

        detections = []
        for label in range(1, num_labels):  # 0 is background
            x_patch = stats[label, cv2.CC_STAT_LEFT]
            y_patch = stats[label, cv2.CC_STAT_TOP]
            w_patch = stats[label, cv2.CC_STAT_WIDTH]
            h_patch = stats[label, cv2.CC_STAT_HEIGHT]

            # Convert patch coordinates → pixel coordinates
            scale_x = img_w / grid_w
            scale_y = img_h / grid_h
            x1 = x_patch * scale_x
            y1 = y_patch * scale_y
            x2 = (x_patch + w_patch) * scale_x
            y2 = (y_patch + h_patch) * scale_y

            x_c = (x1 + x2) / 2
            y_c = (y1 + y2) / 2
            w_px = x2 - x1
            h_px = y2 - y1

            # Confidence ≈ normalised total attention mass of the region
            region_mask = (label_map == label)
            conf = float(patch_attn[region_mask].mean())

            detections.append((x_c, y_c, w_px, h_px, conf))

        # Sort by confidence descending, keep top-max_objects
        detections.sort(key=lambda d: d[4], reverse=True)
        return detections[: self.max_objects]

    def _detections_to_flat_obs(self, frame_np: np.ndarray) -> np.ndarray:
        """
        Run DINOv2 attention detection on *frame_np* (H×W×3 uint8) and return a
        float32 vector of length ``self.num_features`` with each detection written
        into its assigned slot as ``[x, y, w, h, confidence]``.
        """
        flat = np.zeros(self.num_features, dtype=np.float32)
        detections = self._attention_to_boxes(frame_np)
        for region_idx, (x, y, w, h, conf) in enumerate(detections):
            if region_idx not in self.class_map:
                continue
            slot_start, slot_size = self.class_map[region_idx]
            values = np.array([x, y, w, h, conf], dtype=np.float32)
            fill_len = min(slot_size, 5)
            if slot_start + fill_len <= self.num_features:
                flat[slot_start: slot_start + fill_len] = values[:fill_len]
        return flat

    # ------------------------------------------------------------------ #
    #  Pixel-frame extraction                                              #
    # ------------------------------------------------------------------ #

    def _get_frame(self, pixel_obs_or_state) -> np.ndarray:
        """
        Extract the latest RGB frame as a numpy uint8 array (H×W×3).

        Works whether the inner env returns ``(frame_stack, H, W, C)`` pixel
        obs directly (when wrapped in ``PixelObsWrapper``) or just plain
        ``AtariState``.
        """
        if isinstance(pixel_obs_or_state, (np.ndarray, jnp.ndarray)):
            # pixel_obs has shape (frame_stack, H, W, C)
            arr = np.array(pixel_obs_or_state)
            if arr.ndim == 4:
                return arr[-1].astype(np.uint8)
            return arr.astype(np.uint8)
        # Fallback: render via the env directly (AtariState)
        from jaxatari.wrappers import AtariWrapper
        if hasattr(self._env, "render"):
            return np.array(self._env.render(pixel_obs_or_state.env_state)).astype(np.uint8)
        raise RuntimeError(
            "Could not extract a pixel frame from the environment output. "
            "Wrap the environment in PixelObsWrapper before DinoV2ObjectCentricWrapper, "
            "or use an env that exposes a render() method."
        )

    # ------------------------------------------------------------------ #
    #  Public env interface                                                #
    # ------------------------------------------------------------------ #

    def observation_space(self) -> spaces.Box:
        """Returns a Box space of shape ``(frame_stack_size, num_features)``."""
        return spaces.Box(
            low=0.0,
            high=np.inf,
            shape=self._obs_shape,
            dtype=np.float32,
        )

    def action_space(self):
        """Delegates to the inner environment."""
        return self._env.action_space()

    # Proxy unknown attributes to the inner env (mirrors JaxatariWrapper)
    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def reset(self, key: chex.PRNGKey) -> Tuple[np.ndarray, Any]:
        """
        Reset the environment and return an initial stacked observation.

        Returns
        -------
        obs_stack : np.ndarray of shape ``(frame_stack_size, num_features)``
        state : env state (passed through from the inner env)
        """
        pixel_obs, state = self._env.reset(key)
        frame = self._get_frame(pixel_obs)
        flat = self._detections_to_flat_obs(frame)
        obs_stack = np.stack([flat] * self.frame_stack_size)  # (stack, features)
        return jnp.array(obs_stack), state

    def step(
        self, state: Any, action: int
    ) -> Tuple[np.ndarray, Any, float, bool, bool, Dict[str, Any]]:
        """
        Step the environment ``frame_skip`` times and return a detection-based
        stacked observation.

        The inner ``step`` is called ``frame_skip`` times; only the pixel frame
        from the *last* sub-step is used for detection.  Rewards are summed and
        optionally clipped to ``{-1, 0, +1}``.

        Returns
        -------
        obs_stack : jnp.ndarray of shape ``(frame_stack_size, num_features)``
        new_state : env state
        reward    : float
        terminated : bool
        truncated  : bool
        info      : dict
        """
        # Execute frame_skip inner steps, collecting rewards & done flags
        pixel_obs = None
        new_state = state
        total_reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        for _ in range(self.frame_skip):
            pixel_obs, new_state, reward, terminated, truncated, info = self._env.step(
                new_state, action
            )
            total_reward += float(reward)
            if terminated or truncated:
                break

        if self.clip_reward:
            total_reward = float(np.sign(total_reward))

        # Detect objects in the last frame
        frame = self._get_frame(pixel_obs)
        flat = self._detections_to_flat_obs(frame)

        # Retrieve or initialise the current obs stack
        if hasattr(state, "_dino_obs_stack"):
            current_stack = state._dino_obs_stack
        else:
            current_stack = np.zeros(self._obs_shape, dtype=np.float32)

        obs_stack = np.concatenate(
            [current_stack[1:], flat[np.newaxis]], axis=0
        )

        # Stash the stack on the state object for the next call.
        # We attach it as a plain Python attribute so it survives across steps
        # without polluting JAX pytree tracing (the inner state is unchanged).
        try:
            new_state._dino_obs_stack = obs_stack
        except AttributeError:
            # The state is a frozen dataclass / struct – wrap it in a thin
            # container that carries the extra field alongside the original.
            new_state = _DinoStateWrapper(new_state, obs_stack)

        return jnp.array(obs_stack), new_state, total_reward, terminated, truncated, info


class _DinoStateWrapper:
    """
    Thin mutable wrapper that carries a DINOv2 obs stack alongside an
    immutable inner environment state (e.g. a Flax ``struct.dataclass``).

    This is only created when the inner state object does not support
    arbitrary attribute assignment.
    """

    def __init__(self, inner_state: Any, obs_stack: np.ndarray):
        self._inner_state = inner_state
        self._dino_obs_stack = obs_stack

    # Proxy all attribute access to the inner state so the rest of the code
    # (e.g. LogWrapper, FlattenObservationWrapper) sees the real env state.
    def __getattr__(self, name: str):
        return getattr(self._inner_state, name)

    def __repr__(self) -> str:
        return f"_DinoStateWrapper({self._inner_state!r})"


@functools.partial(jax.jit, static_argnames=('sigma',))
def _gaussian_blur_2d_nchw(image: chex.Array, sigma: float = 3.0) -> chex.Array:
    """Depthwise separable Gaussian blur for NCHW images (used by preprocess_image)."""
    # image input: [N, C, H, W]
    c = image.shape[1]
    radius = int(sigma * 3)
    size = radius * 2 + 1
    x = jnp.linspace(-radius, radius, size)
    phi_x = jnp.exp(-0.5 * (x / sigma)**2)
    phi_x = (phi_x / phi_x.sum()).astype(image.dtype)

    h_kernel = phi_x[None, None, None, :]
    h_kernel = jnp.tile(h_kernel, (c, 1, 1, 1))

    v_kernel = phi_x[None, None, :, None]
    v_kernel = jnp.tile(v_kernel, (c, 1, 1, 1))

    out = jax.lax.conv_general_dilated(
        image, h_kernel, (1, 1), padding='SAME',
        feature_group_count=c,
        dimension_numbers=('NCHW', 'OIHW', 'NCHW')
    )
    out = jax.lax.conv_general_dilated(
        out, v_kernel, (1, 1), padding='SAME',
        feature_group_count=c,
        dimension_numbers=('NCHW', 'OIHW', 'NCHW')
    )
    return out


@functools.partial(jax.jit, static_argnums=(0,))
def preprocess_image(class_instance: JaxatariWrapper, image: chex.Array) -> chex.Array:
    """Applies resizing and grayscaling to a single image frame."""
    image = image.astype(jnp.float32)

    # Has to use a standard Python `if` since jax.lax.cond would fail due to different shapes. This is possible since do_pixel_resize is a static parameter.
    if class_instance.do_pixel_resize:
        image = jim.resize(image, (class_instance.pixel_resize_shape[0], class_instance.pixel_resize_shape[1], image.shape[-1]), method='bilinear')
    
    # applies grayscale if enabled with the same method as for resize
    if class_instance.grayscale:
        image = jnp.dot(image, jnp.array([0.2989, 0.5870, 0.1140]))[..., jnp.newaxis] # numbers for grayscale transformation as in https://en.wikipedia.org/wiki/Luma_(video)

    # Apply gaussian smoothing to natively downscaled images to get similar effect to actual downscaling
    if class_instance.native_downscaling and class_instance.smooth_image:
        image_gauss = _gaussian_blur_2d_nchw(image[None].transpose(0, 3, 1, 2))
        image = image_gauss.squeeze().reshape(image.shape)
    
    return image.astype(jnp.uint8)

@struct.dataclass 
class PixelState:
    atari_state: AtariState
    image_stack: jax.Array

class PixelObsWrapper(JaxatariWrapper):
    """
    Wrapper for Atari environments that returns the flattened pixel observations.
    Apply this wrapper after the AtariWrapper!
    """
    # TODO: remove do_pixel_resize and resize whenever a different shape / grayscale is given?
    def __init__(self, env, do_pixel_resize: bool = False, pixel_resize_shape: tuple[int, int] = (84, 84), grayscale: bool = False, use_native_downscaling: bool = False, smooth_image: bool = False, frame_stack_size: int = 4, frame_skip: int = 4, max_pooling: bool = True, clip_reward: bool = True):
        super().__init__(env)
        assert isinstance(env, AtariWrapper), "PixelObsWrapper has to be applied after AtariWrapper"
        self.frame_stack_size = frame_stack_size
        self.frame_skip = frame_skip
        self.max_pooling = max_pooling
        self.clip_reward = clip_reward
        self.smooth_image = smooth_image
        self.native_downscaling = False

        # Access the Base Environment
        base_env = self._env._env if isinstance(self._env, AtariWrapper) else self._env

        if do_pixel_resize and use_native_downscaling:
            # call helper from modifications to make sure that applied mods remain applied after native downscaling (lazy import to avoid circular dependency)
            from jaxatari.modification import apply_native_downscaling
            self.do_pixel_resize, self.grayscale = apply_native_downscaling(
                base_env, pixel_resize_shape, grayscale
            )
            self.pixel_resize_shape = pixel_resize_shape
            self.native_downscaling = True
        else:
            self.do_pixel_resize = do_pixel_resize
            self.pixel_resize_shape = pixel_resize_shape
            self.grayscale = grayscale

        # Dynamically calculate the final observation space shape
        # If we hot-swapped, image_space() will now return the correct small size automatically
        final_shape = self._env.image_space().shape
        
        # If we are doing wrapper-side resizing (legacy), we still calculate manually
        if self.do_pixel_resize:
            height, width = self.pixel_resize_shape
            channels = 1 if self.grayscale else final_shape[2]
            final_shape = (height, width, channels)

        # Create the space for a single preprocessed frame
        image_space = spaces.Box(low=0, high=255, shape=final_shape, dtype=jnp.uint8)
        # Stack the single-frame space
        self._observation_space = spaces.stack_space(image_space, self.frame_stack_size)

    def observation_space(self) -> spaces.Space:
        """Returns the stacked image space."""
        return self._observation_space
    

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, PixelState]:
        _, atari_state = self._env.reset(key)
        image = self._env.render(atari_state.env_state)
        processed_image = preprocess_image(self, image)

        image_stack = jnp.stack([processed_image] * self.frame_stack_size)
        return image_stack, PixelState(atari_state, image_stack)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: PixelState,
        action: int,
    ) -> Tuple[chex.Array, PixelState, float, bool, bool, Dict[Any, Any]]:

        def body_fn(carry, _):
            atari_state, action = carry
            _, new_atari_state, reward, terminated, truncated, info = self._env.step(atari_state, action)
            return (new_atari_state, action), (new_atari_state.env_state, reward, terminated, truncated, info)

        (atari_state, _), (env_states, rewards, terminations, truncations, infos) = jax.lax.scan(
            body_fn,
            (state.atari_state, action),
            None,
            length=self.frame_skip,
        )

        last_env_state = jax.tree.map(lambda x: x[-1], env_states)
        if self.max_pooling and self.frame_skip > 1:
            image = self._env.render(last_env_state)
            prev_env_state = jax.tree.map(lambda x: x[-2], env_states)
            prev_image = self._env.render(prev_env_state)
            latest_image = jnp.maximum(image, prev_image)
        else:
            latest_image = self._env.render(last_env_state)
        processed_image = preprocess_image(self, latest_image)
        image_stack = jnp.concatenate([state.image_stack[1:], jnp.expand_dims(processed_image, axis=0)], axis=0)

        reward = jnp.sum(rewards)
        if self.clip_reward:
            reward = jnp.sign(reward)
        terminated = terminations.any()
        truncated = truncations.any()
        # Autoreset (gym's SAME_STEP mode) -> reset whole stack
        image_stack, pixel_state = jax.lax.cond(
            jnp.logical_or(infos["env_done"].any(), truncated),  # use actual env_done for reset condition, not affected by episodic life
            lambda: self.reset(atari_state.key),  
            lambda: (image_stack, PixelState(atari_state, image_stack))
        )

        def reduce_info(k, v):
            if k in ["env_reward", "all_rewards"]:
                return jnp.sum(v, axis=0)
            if k == "env_done":
                return jnp.any(v)
            return v[-1]

        info_dict = {k: reduce_info(k, v) for k, v in infos.items()}
        return image_stack, pixel_state, reward, terminated, truncated, info_dict

@struct.dataclass 
class PixelAndObjectCentricState:
    atari_state: AtariState
    image_stack: jax.Array
    obs_stack: Any

class PixelAndObjectCentricWrapper(JaxatariWrapper):
    """
    Wrapper for Atari environments that returns the flattened pixel observations and object-centric observations.
    Apply this wrapper after the AtariWrapper!
    """
    
    def __init__(self, env, do_pixel_resize: bool = False, pixel_resize_shape: tuple[int, int] = (84, 84), grayscale: bool = False, use_native_downscaling: bool = False, smooth_image: bool = False, frame_stack_size: int = 4, frame_skip: int = 4, max_pooling: bool = True, clip_reward: bool = True):
        super().__init__(env)
        assert isinstance(env, AtariWrapper), "PixelAndObjectCentricWrapper must be applied after AtariWrapper"
        self.frame_stack_size = frame_stack_size
        self.frame_skip = frame_skip
        self.max_pooling = max_pooling
        self.clip_reward = clip_reward
        self.smooth_image = smooth_image
        self.native_downscaling = False
        
        # Access the Base Environment
        base_env = self._env._env if isinstance(self._env, AtariWrapper) else self._env

        if do_pixel_resize and use_native_downscaling:
            # call helper from modifications to make sure that applied mods remain applied after native downscaling (lazy import to avoid circular dependency)
            from jaxatari.modification import apply_native_downscaling
            self.do_pixel_resize, self.grayscale = apply_native_downscaling(
                base_env, pixel_resize_shape, grayscale
            )
            self.pixel_resize_shape = pixel_resize_shape
            self.native_downscaling = True
        else:
            self.do_pixel_resize = do_pixel_resize
            self.pixel_resize_shape = pixel_resize_shape
            self.grayscale = grayscale

        # Part 1: Define the stacked image space.
        # If we hot-swapped, image_space() will now return the correct small size automatically
        final_shape = self._env.image_space().shape
        
        # If we are doing wrapper-side resizing (legacy), we still calculate manually
        if self.do_pixel_resize:
            height, width = self.pixel_resize_shape
            channels = 1 if self.grayscale else final_shape[2]
            final_shape = (height, width, channels)
        
        image_space = spaces.Box(low=0, high=255, shape=final_shape, dtype=jnp.uint8)
        stacked_image_space = spaces.stack_space(image_space, self.frame_stack_size)

        # Part 2: Define the FLATTENED object space with exact bounds.
        single_frame_space = self._env.observation_space()
        lows, highs = [], []

        for leaf_space in jax.tree.leaves(single_frame_space):
            if isinstance(leaf_space, spaces.Box):
                low_arr = np.broadcast_to(leaf_space.low, leaf_space.shape).flatten()
                high_arr = np.broadcast_to(leaf_space.high, leaf_space.shape).flatten()
                lows.append(low_arr)
                highs.append(high_arr)
            elif isinstance(leaf_space, spaces.Discrete):
                lows.append(np.array([0], dtype=leaf_space.dtype))
                highs.append(np.array([leaf_space.n - 1], dtype=leaf_space.dtype))
            else:
                raise TypeError(f"Unsupported space type for flattening: {type(leaf_space)}")
        
        if not lows:
            raise ValueError("The observation space appears to be empty or contain unsupported types.")

        single_frame_lows = np.concatenate(lows)
        single_frame_highs = np.concatenate(highs)

        stacked_object_space_flat = spaces.Box(
            low=single_frame_lows,
            high=single_frame_highs,
            shape=(self.frame_stack_size, int(single_frame_lows.shape[0])),
            dtype=jnp.float32
        )

        # Part 3: Combine them into the final Tuple space.
        self._observation_space = spaces.Tuple((
            stacked_image_space,
            stacked_object_space_flat
        ))
    
    
    def observation_space(self) -> spaces.Tuple:
        """Returns a Tuple space containing stacked image and object spaces."""
        return self._observation_space
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _flatten_single_obs(self, obs):
        """Flatten a single object-centric observation using ravel_pytree."""
        return flatten_util.ravel_pytree(obs)[0].astype(jnp.float32)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey
    ) -> Tuple[Tuple[chex.Array, chex.Array], PixelAndObjectCentricState]:
        obs, atari_state = self._env.reset(key)
        flat_obs = self._flatten_single_obs(obs)
        obs_stack = jnp.stack([flat_obs] * self.frame_stack_size)

        image = self._env.render(atari_state.env_state)
        processed_image = preprocess_image(self, image)
        image_stack = jnp.stack([processed_image] * self.frame_stack_size)

        new_state = PixelAndObjectCentricState(atari_state, image_stack, obs_stack)
        return (image_stack, obs_stack), new_state
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: PixelAndObjectCentricState,
        action: int,
    ) -> Tuple[Tuple[chex.Array, chex.Array], PixelAndObjectCentricState, float, bool, bool, Dict[Any, Any]]:
        def body_fn(carry, _):
            atari_state, action = carry
            obs, new_atari_state, reward, terminated, truncated, info = self._env.step(atari_state, action)
            return (new_atari_state, action), (obs, new_atari_state.env_state, reward, terminated, truncated, info)

        (atari_state, _), (obs, env_states, rewards, terminations, truncations, infos) = jax.lax.scan(
            body_fn,
            (state.atari_state, action),
            None,
            length=self.frame_skip,
        )

        latest_obs = jax.tree.map(lambda x: x[-1], obs)
        flat_latest_obs = self._flatten_single_obs(latest_obs)
        obs_stack = jnp.concatenate([state.obs_stack[1:], jnp.expand_dims(flat_latest_obs, axis=0)], axis=0)

        last_env_state = jax.tree.map(lambda x: x[-1], env_states)
        if self.max_pooling and self.frame_skip > 1:
            image = self._env.render(last_env_state)
            prev_env_state = jax.tree.map(lambda x: x[-2], env_states)
            prev_image = self._env.render(prev_env_state)
            latest_image = jnp.maximum(image, prev_image)
        else:
            latest_image = self._env.render(last_env_state)

        processed_image = preprocess_image(self, latest_image)
        image_stack = jnp.concatenate([state.image_stack[1:], jnp.expand_dims(processed_image, axis=0)], axis=0)

        reward = jnp.sum(rewards)
        if self.clip_reward:
            reward = jnp.sign(reward)
        terminated = terminations.any()
        truncated = truncations.any()
        # Autoreset (gym's SAME_STEP mode) -> reset whole stack
        (image_stack, obs_stack), pixel_oc_state = jax.lax.cond(
            jnp.logical_or(infos["env_done"].any(), truncated),  # use actual env_done for reset condition, not affected by episodic life
            lambda: self.reset(atari_state.key),
            lambda: ((image_stack, obs_stack), PixelAndObjectCentricState(atari_state, image_stack, obs_stack))
        )

        def reduce_info(k, v):
            if k in ["env_reward", "all_rewards"]:
                return jnp.sum(v, axis=0)
            if k == "env_done":
                return jnp.any(v)
            return v[-1]

        info_dict = {k: reduce_info(k, v) for k, v in infos.items()}
        return (image_stack, obs_stack), pixel_oc_state, reward, terminated, truncated, info_dict

class PixelAndObjectObsWrapper(PixelAndObjectCentricWrapper):
    """
    Exactly the same as PixelAndObjectCentricWrapper, but return structured OC-obs instead of flattened array.
    """

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey
    ) -> Tuple[Tuple[chex.Array, Any], PixelAndObjectCentricState]:
        obs, atari_state = self._env.reset(key)
        image = self._env.render(atari_state.env_state)
        processed_image = preprocess_image(self, image)
        image_stack = jnp.stack([processed_image] * self.frame_stack_size)

        obs_stack = jax.tree.map(
            lambda leaf: jnp.stack([leaf] * self.frame_stack_size),
            obs,
        )

        new_state = PixelAndObjectCentricState(atari_state, image_stack, obs_stack)
        return (image_stack, obs_stack), new_state
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: PixelAndObjectCentricState,
        action: int,
    ) -> Tuple[Tuple[chex.Array, chex.Array], PixelAndObjectCentricState, float, bool, bool, Dict[Any, Any]]:
        def body_fn(carry, _):
            atari_state, action = carry
            obs, new_atari_state, reward, terminated, truncated, info = self._env.step(atari_state, action)
            return (new_atari_state, action), (obs, new_atari_state.env_state, reward, terminated, truncated, info)

        (atari_state, _), (obs, env_states, rewards, terminations, truncations, infos) = jax.lax.scan(
            body_fn,
            (state.atari_state, action),
            None,
            length=self.frame_skip,
        )

        latest_obs = jax.tree.map(lambda x: x[-1], obs)
        obs_stack = jax.tree.map(
            lambda stack_leaf, obs_leaf: jnp.concatenate([stack_leaf[1:], jnp.expand_dims(obs_leaf, axis=0)], axis=0),
            state.obs_stack,
            latest_obs,
        )

        last_env_state = jax.tree.map(lambda x: x[-1], env_states)
        if self.max_pooling and self.frame_skip > 1:
            image = self._env.render(last_env_state)
            prev_env_state = jax.tree.map(lambda x: x[-2], env_states)
            prev_image = self._env.render(prev_env_state)
            latest_image = jnp.maximum(image, prev_image)
        else:
            latest_image = self._env.render(last_env_state)

        processed_image = preprocess_image(self, latest_image)
        image_stack = jnp.concatenate([state.image_stack[1:], jnp.expand_dims(processed_image, axis=0)], axis=0)

        reward = jnp.sum(rewards)
        terminated = terminations.any()
        truncated = truncations.any()
        # Autoreset (gym's SAME_STEP mode) -> reset whole stack
        (image_stack, obs_stack), pixel_oc_state = jax.lax.cond(
            jnp.logical_or(infos["env_done"].any(), truncated),  # use actual env_done for reset condition, not affected by episodic life
            lambda: self.reset(atari_state.key),
            lambda: ((image_stack, obs_stack), PixelAndObjectCentricState(atari_state, image_stack, obs_stack)),
        )

        def reduce_info(k, v):
            if k in ["env_reward", "all_rewards"]:
                return jnp.sum(v, axis=0)
            if k == "env_done":
                return jnp.any(v)
            return v[-1]

        info_dict = {k: reduce_info(k, v) for k, v in infos.items()}
        new_state = PixelAndObjectCentricState(atari_state, image_stack, obs_stack)
        return (image_stack, obs_stack), pixel_oc_state, reward, terminated, truncated, info_dict


class FlattenObservationWrapper(JaxatariWrapper):
    """
    A wrapper that flattens each leaf array in an observation Pytree.

    Compatible with all the other wrappers, flattens the observations whilst preserving the overarching structure (i.e. if the observation is a tuple of multiple observations, the flattened observation will be a tuple of flattened observations).
    """

    def __init__(self, env):
        super().__init__(env)

        # build the new (flattened) observation space
        original_space = self._env.observation_space()

        def flatten_space(space: spaces.Box) -> spaces.Box:
            # Create flattened low/high arrays by broadcasting the original bounds
            # and then reshaping. This preserves the bounds for each element.
            flat_low = np.broadcast_to(space.low, space.shape).flatten()
            flat_high = np.broadcast_to(space.high, space.shape).flatten()
            
            return spaces.Box(
                low=jnp.array(flat_low),
                high=jnp.array(flat_high),
                dtype=space.dtype
            )
        
        self._observation_space = jax.tree.map(
            flatten_space,
            original_space,
            is_leaf=lambda x: isinstance(x, spaces.Box)
        )

    def observation_space(self) -> spaces.Space:
        """Returns a space where each leaf array is flattened."""
        return self._observation_space

    def _process_obs(self, obs_tree: chex.ArrayTree) -> chex.ArrayTree:
        """Applies .flatten() to each leaf array in the pytree."""
        def flatten_and_cast(leaf):
            flattened = leaf.flatten()
            # Cast to float32 to match space dtype
            return flattened.astype(jnp.float32) if isinstance(leaf, jnp.ndarray) else flattened
        return jax.tree.map(flatten_and_cast, obs_tree)

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.ArrayTree, Any]:
        obs, state = self._env.reset(key)
        processed_obs = self._process_obs(obs)
        return processed_obs, state # State can be passed through directly

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: Any,
        action: int,
    ) -> Tuple[chex.ArrayTree, Any, float, bool, bool, Dict[str, Any]]:
        obs, next_state, reward, terminated, truncated, info = self._env.step(state, action)
        processed_obs = self._process_obs(obs)
        return processed_obs, next_state, reward, terminated, truncated, info


class NormalizeObservationWrapper(JaxatariWrapper):
    """
    A wrapper that normalizes each leaf in an observation Pytree.
    This wrapper is compatible with any observation structure (Pytrees).
    """

    def __init__(self, env, to_neg_one: bool = False, dtype=jnp.float16):
        super().__init__(env)
        self._to_neg_one = to_neg_one
        self._dtype = dtype

        original_space = self._env.observation_space()

        # Create Pytrees of the same structure as observations, but holding the low/high bounds.
        self._low = jax.tree.map(
            lambda s: jnp.array(s.low, dtype=s.dtype),
            original_space,
            is_leaf=lambda x: isinstance(x, spaces.Box)
        )
        self._high = jax.tree.map(
            lambda s: jnp.array(s.high, dtype=s.dtype),
            original_space,
            is_leaf=lambda x: isinstance(x, spaces.Box)
        )

        # The new observation space will have the same structure, but all leaves
        def _normalize_space(space: spaces.Box) -> spaces.Box:
            low_val = -1.0 if self._to_neg_one else 0.0
            return spaces.Box(
                low=low_val,
                high=1.0,
                shape=space.shape,
                dtype=self._dtype
            )

        self._observation_space = jax.tree.map(
            _normalize_space,
            original_space,
            is_leaf=lambda x: isinstance(x, spaces.Box)
        )

    def observation_space(self) -> spaces.Space:
        """Returns the normalized observation space where leaves are in [0, 1]."""
        return self._observation_space

    def _normalize_leaf(self, obs_leaf, low_leaf, high_leaf):
        """Helper function to normalize a single leaf array."""
        obs_leaf = obs_leaf.astype(self._dtype)
        
        # Calculate the range and scale for normalization
        range_leaf = high_leaf.astype(self._dtype) - low_leaf.astype(self._dtype)
        scale = 1.0 / jnp.where(range_leaf > 1e-8, range_leaf, 1.0)
        
        # Normalize to [0, 1]
        normalized_0_1 = (obs_leaf - low_leaf.astype(self._dtype)) * scale

        # Conditionally shift to [-1, 1]
        final_normalized = jax.lax.cond(
            self._to_neg_one,
            lambda x: 2.0 * x - 1.0,
            lambda x: x,
            normalized_0_1
        )
        
        # Clip to ensure values are within the target range
        clip_low = -1.0 if self._to_neg_one else 0.0
        return jnp.clip(final_normalized, clip_low, 1.0)

    def _normalize_obs(self, obs: chex.ArrayTree) -> chex.ArrayTree:
        """
        Applies normalization to each leaf array in the observation pytree,
        robustly handling structural mismatches between observation and space Pytrees.
        """
        # Get the leaves of all pytrees. Since the number of leaves and their
        # order is guaranteed to be the same, we can work with the flat lists.
        obs_leaves = jax.tree.leaves(obs)
        low_leaves = jax.tree.leaves(self._low)
        high_leaves = jax.tree.leaves(self._high)

        # Apply the normalization to each corresponding leaf triplet.
        normalized_leaves = [
            self._normalize_leaf(o, l, h)
            for o, l, h in zip(obs_leaves, low_leaves, high_leaves)
        ]

        # Reconstruct the output pytree with the same structure as the input 'obs'.
        return jax.tree.unflatten(jax.tree.structure(obs), normalized_leaves)

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.ArrayTree, Any]:
        obs, state = self._env.reset(key)
        normalized_obs = self._normalize_obs(obs)
        return normalized_obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: Any,
        action: int,
    ) -> Tuple[chex.ArrayTree, Any, float, bool, bool, Dict[str, Any]]:
        obs, next_state, reward, terminated, truncated, info = self._env.step(state, action)
        normalized_obs = self._normalize_obs(obs)
        return normalized_obs, next_state, reward, terminated, truncated, info


@struct.dataclass
class LogState:
    atari_state: Any # Can be any of the states from wrappers above
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int

class LogWrapper(JaxatariWrapper):
    """Log episode returns and lengths. An episode ends when the wrapped env returns done=True.
    Uses env_reward from info when present (unclipped); otherwise uses the step reward.
    """

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey
    ) -> Tuple[chex.Array, LogState]:
        obs, atari_state = self._env.reset(key)
        state = LogState(atari_state, 0.0, 0, 0.0, 0)
        return obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: LogState,
        action: int,
    ) -> Tuple[chex.Array, LogState, float, bool, bool, Dict[Any, Any]]:
        obs, atari_state, reward, terminated, truncated, info = self._env.step(state.atari_state, action)
        # use env_reward (unclipped/unchanged) for logging when available
        new_episode_return = state.episode_returns + info.get("env_reward", reward)
        new_episode_length = state.episode_lengths + 1
        # use env_done for logging when available (e.g. to ignore episodic_life)
        done = info.get("env_done", jnp.bool_(terminated))
        done = jnp.logical_or(done, truncated) # truncated episodes are considered done for logging purposes
        state = LogState(
            atari_state=atari_state,
            episode_returns=jnp.where(done, jnp.float32(0), jnp.float32(new_episode_return)),
            episode_lengths=jnp.where(done, jnp.int32(0), jnp.int32(new_episode_length)),
            returned_episode_returns=jnp.where(
                done, jnp.float32(new_episode_return), jnp.float32(state.returned_episode_returns)
            ),
            returned_episode_lengths=jnp.where(
                done, jnp.int32(new_episode_length), jnp.int32(state.returned_episode_lengths)
            ),
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = done
        # Still need to return the actual/wrapped termination signal (affected by episodic life)
        return obs, state, reward, terminated, truncated, info

@struct.dataclass
class MultiRewardLogState:
    atari_state: Any # Can be any of the states from wrappers above
    episode_returns_env: float
    episode_returns: chex.Array
    episode_lengths: int
    returned_episode_returns_env: float
    returned_episode_returns: chex.Array
    returned_episode_lengths: int

class MultiRewardLogWrapper(JaxatariWrapper):
    """Log episode returns and lengths for multiple rewards. An episode ends when the wrapped env returns done=True.
    Apply MultiRewardWrapper to the core env when using this wrapper.
    Final logs: 'returned_episode_returns_0', ... for each reward function; env reward in 'returned_episode_env_returns'.
    """

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey,
    ) -> Tuple[chex.Array, MultiRewardLogState]:
        obs, atari_state = self._env.reset(key)
        # Dummy step to get info structure 
        _, _, _, _, _, dummy_info = self._env.step(atari_state, 0)
        rewards_shape_provider = dummy_info.get("all_rewards", jnp.zeros(1))
        episode_returns_init = jnp.zeros_like(rewards_shape_provider)
        state = MultiRewardLogState(atari_state, 0.0, episode_returns_init, 0, 0.0, episode_returns_init, 0)
        return obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: MultiRewardLogState,
        action: int,
    ) -> Tuple[chex.Array, MultiRewardLogState, float, bool, bool, Dict[Any, Any]]:
        obs, atari_state, reward, terminated, truncated, info = self._env.step(state.atari_state, action)
        new_episode_return_env = state.episode_returns_env + info.get("env_reward", reward)
        all_rewards_step = info.get("all_rewards", jnp.zeros_like(state.episode_returns))
        new_episode_return = state.episode_returns + all_rewards_step
        new_episode_length = state.episode_lengths + 1
        done_ = info.get("env_done", jnp.bool_(terminated))
        state = MultiRewardLogState(
            atari_state=atari_state,
            episode_returns_env=jnp.where(done_, jnp.float32(0), jnp.float32(new_episode_return_env)),
            episode_returns=jnp.where(done_, jnp.zeros_like(state.episode_returns), new_episode_return),
            episode_lengths=jnp.where(done_, jnp.int32(0), jnp.int32(new_episode_length)),
            returned_episode_returns_env=jnp.where(
                done_, jnp.float32(new_episode_return_env), jnp.float32(state.returned_episode_returns_env)
            ),
            returned_episode_returns=jnp.where(
                done_, new_episode_return, state.returned_episode_returns
            ),
            returned_episode_lengths=jnp.where(
                done_, jnp.int32(new_episode_length), jnp.int32(state.returned_episode_lengths)
            ),
        )
        info["returned_episode_env_returns"] = state.returned_episode_returns_env
        for i, r in enumerate(new_episode_return):
            info[f"returned_episode_returns_{i}"] = state.returned_episode_returns[i]
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = done_
        return obs, state, reward, terminated, truncated, info
