from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np

from fusion_simulation.agent.config import AgentConfig, AgentRole
from fusion_simulation.agent.manager import AgentManager
from fusion_simulation.core.kernel import KernelConfig, SimulationKernel
from fusion_simulation.sensor.manager import SensorManager

logger = logging.getLogger(__name__)


class ObservationManager:
    _groups: dict[str, list[str]]
    _noise_scale: dict[str, float]
    _history_len: int
    _history: dict[str, list[np.ndarray]]

    def __init__(self, groups: dict[str, list[str]] | None = None,
                 noise_scale: dict[str, float] | None = None,
                 history_len: int = 1):
        self._groups = groups or {"policy": []}
        self._noise_scale = noise_scale or {}
        self._history_len = history_len
        self._history = {}

    def compute(self, raw_obs: dict[str, Any]) -> dict[str, np.ndarray]:
        result = {}
        for group_name, keys in self._groups.items():
            arrays = []
            for key in keys:
                val = raw_obs.get(key)
                if val is None:
                    continue
                arr = np.asarray(val, dtype=np.float32).flatten()
                scale = self._noise_scale.get(key, 0.0)
                if scale > 0.0:
                    arr = arr + np.random.normal(0, scale, arr.shape).astype(np.float32)
                arrays.append(arr)
            if arrays:
                result[group_name] = np.concatenate(arrays)
            if group_name not in self._history:
                self._history[group_name] = []
            if group_name in result:
                self._history[group_name].append(result[group_name])
                if len(self._history[group_name]) > self._history_len:
                    self._history[group_name] = self._history[group_name][-self._history_len:]
        return result

    def get_history(self, group: str) -> list[np.ndarray]:
        return self._history.get(group, [])

    def reset(self) -> None:
        self._history.clear()


class ActionManager:
    _pending_action: np.ndarray | None
    _action_scale: np.ndarray | None
    _action_lower: np.ndarray | None
    _action_upper: np.ndarray | None

    def __init__(self, action_dim: int,
                 action_scale: list[float] | None = None,
                 action_lower: list[float] | None = None,
                 action_upper: list[float] | None = None):
        self._action_dim = action_dim
        self._pending_action = None
        self._action_scale = np.array(action_scale, dtype=np.float32) if action_scale else None
        self._action_lower = np.array(action_lower, dtype=np.float32) if action_lower else None
        self._action_upper = np.array(action_upper, dtype=np.float32) if action_upper else None

    def process_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32).flatten()
        if len(action) < self._action_dim:
            action = np.pad(action, (0, self._action_dim - len(action)))
        elif len(action) > self._action_dim:
            action = action[:self._action_dim]
        if self._action_scale is not None:
            action = action * self._action_scale
        if self._action_lower is not None:
            action = np.maximum(action, self._action_lower)
        if self._action_upper is not None:
            action = np.minimum(action, self._action_upper)
        self._pending_action = action

    def apply_action(self) -> np.ndarray | None:
        action = self._pending_action
        self._pending_action = None
        return action

    def reset(self) -> None:
        self._pending_action = None


class RewardManager:
    _reward_fns: list[tuple[str, Any]]
    _cumulative: float
    _step_reward: float

    def __init__(self):
        self._reward_fns = []
        self._cumulative = 0.0
        self._step_reward = 0.0

    def add_reward_fn(self, name: str, fn: Any) -> None:
        self._reward_fns.append((name, fn))

    def compute(self, obs: dict, info: dict) -> float:
        total = 0.0
        for name, fn in self._reward_fns:
            try:
                total += float(fn(obs, info))
            except Exception as e:
                logger.warning("Reward fn %s failed: %s", name, e)
        self._step_reward = total
        self._cumulative += total
        return total

    @property
    def cumulative(self) -> float:
        return self._cumulative

    @property
    def step_reward(self) -> float:
        return self._step_reward

    def reset(self) -> None:
        self._cumulative = 0.0
        self._step_reward = 0.0


class TerminationManager:
    _term_fns: list[tuple[str, Any]]
    _timeout_fn: Any | None
    _max_steps: int
    _current_step: int

    def __init__(self, max_steps: int = 1000, timeout_fn: Any | None = None):
        self._term_fns = []
        self._timeout_fn = timeout_fn
        self._max_steps = max_steps
        self._current_step = 0

    def add_termination_fn(self, name: str, fn: Any) -> None:
        self._term_fns.append((name, fn))

    def compute_terminated(self, obs: dict, info: dict) -> bool:
        for name, fn in self._term_fns:
            try:
                if fn(obs, info):
                    logger.debug("Terminated by: %s", name)
                    return True
            except Exception as e:
                logger.warning("Term fn %s failed: %s", name, e)
        return False

    def compute_time_out(self) -> bool:
        self._current_step += 1
        if self._current_step >= self._max_steps:
            return True
        if self._timeout_fn is not None:
            try:
                return bool(self._timeout_fn())
            except Exception:
                logger.debug("Timeout check failed", exc=True)
        return False

    def reset(self) -> None:
        self._current_step = 0


class FusionGymEnv:
    is_vector_env: ClassVar[bool] = True

    def __init__(self, agent_config: AgentConfig | None = None,
                 decimation: int = 4,
                 max_steps: int = 1000,
                 headless: bool = True,
                 physics_dt: float = 0.01):
        self._decimation = decimation
        self._kernel = SimulationKernel(KernelConfig(
            headless=headless, physics_dt=physics_dt, max_steps=max_steps,
        ))
        self._sensor_mgr = SensorManager()
        self._agent_mgr = AgentManager()
        self._agent_config = agent_config or AgentConfig(
            name="agent", role=AgentRole.ROBOT, action_dim=6,
        )
        self._obs_mgr = ObservationManager(
            groups={"policy": self._agent_config.obs_keys} if self._agent_config.obs_keys else {"policy": []},
        )
        self._action_mgr = ActionManager(
            action_dim=self._agent_config.action_dim,
            action_scale=self._agent_config.action_scale,
            action_lower=self._agent_config.action_lower,
            action_upper=self._agent_config.action_upper,
        )
        self._reward_mgr = RewardManager()
        self._term_mgr = TerminationManager(max_steps=max_steps)
        self._agent_name = self._agent_config.name
        self._initialized = False
        self._info: dict[str, Any] = {}

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        self._kernel.init(
            sensor_manager=self._sensor_mgr,
            agent_manager=self._agent_mgr,
        )
        self._agent_mgr.add_agent(self._agent_config)
        self._initialized = True
        logger.info("FusionGymEnv initialized: agent=%s decimation=%d", self._agent_name, self._decimation)

    @property
    def single_observation_space(self) -> dict[str, Any]:
        return {"policy": (self._agent_config.action_dim,)}

    @property
    def single_action_space(self) -> tuple[int, ...]:
        return (self._agent_config.action_dim,)

    @property
    def action_space(self) -> tuple[int, ...]:
        return self.single_action_space

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        self._ensure_init()
        if seed is not None:
            self._kernel = SimulationKernel(KernelConfig(headless=True, seed=seed))
            self._kernel.init(
                sensor_manager=self._sensor_mgr,
                agent_manager=self._agent_mgr,
            )
        self._kernel.reset()
        self._obs_mgr.reset()
        self._action_mgr.reset()
        self._reward_mgr.reset()
        self._term_mgr.reset()
        self._info = {}
        raw_obs = self._sensor_mgr.get_observations()
        obs = self._obs_mgr.compute(raw_obs)
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        self._ensure_init()
        self._action_mgr.process_action(action)
        for _ in range(self._decimation):
            applied = self._action_mgr.apply_action()
            if applied is not None:
                agent = self._agent_mgr.get_agent(self._agent_name)
                if agent is not None:
                    agent.record_step(applied.tolist(), reward=0.0)
            self._kernel.step(num_steps=1)
        raw_obs = self._sensor_mgr.get_observations()
        obs = self._obs_mgr.compute(raw_obs)
        reward = self._reward_mgr.compute(obs, self._info)
        terminated = self._term_mgr.compute_terminated(obs, self._info)
        timed_out = self._term_mgr.compute_time_out()
        self._info["step"] = self._term_mgr._current_step
        self._info["cumulative_reward"] = self._reward_mgr.cumulative
        return obs, reward, terminated, timed_out, self._info

    def render(self) -> None:
        pass

    def close(self) -> None:
        if self._initialized:
            self._kernel.close()
            self._initialized = False

    @property
    def kernel(self) -> SimulationKernel:
        return self._kernel
