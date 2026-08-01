from __future__ import annotations

import logging
from typing import Any

from fusion_simulation.agent.config import AgentConfig
from fusion_simulation.agent.policy import PolicyClient
from fusion_simulation.core.ecs import EntityManager
from fusion_simulation.sensor.manager import SensorManager

logger = logging.getLogger(__name__)


class AgentHandle:
    def __init__(self, config: AgentConfig, policy: PolicyClient) -> None:
        self.config = config
        self.policy = policy
        self._step_count: int = 0
        self._last_action: list[float] = []
        self._cumulative_reward: float = 0.0
        self._done: bool = False

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def entity_id(self) -> str:
        return self.config.entity_id

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def last_action(self) -> list[float]:
        return self._last_action

    @property
    def cumulative_reward(self) -> float:
        return self._cumulative_reward

    @property
    def done(self) -> bool:
        return self._done

    def record_step(self, action: list[float], reward: float = 0.0, done: bool = False) -> None:
        self._step_count += 1
        self._last_action = action
        self._cumulative_reward += reward
        self._done = done

    def reset(self) -> None:
        self._step_count = 0
        self._last_action = []
        self._cumulative_reward = 0.0
        self._done = False


class AgentManager:
    def __init__(self, ecs: EntityManager | None = None, sensor_manager: SensorManager | None = None) -> None:
        self._ecs = ecs
        self._sensor_manager = sensor_manager
        self._agents: dict[str, AgentHandle] = {}

    @property
    def agent_count(self) -> int:
        return len(self._agents)

    def set_ecs(self, ecs: EntityManager) -> None:
        self._ecs = ecs

    def set_sensor_manager(self, sm: SensorManager) -> None:
        self._sensor_manager = sm

    def add_agent(self, config: AgentConfig) -> AgentHandle:
        if config.name in self._agents:
            logger.warning("Agent already exists: %s, replacing", config.name)
            self.remove_agent(config.name)
        policy = PolicyClient(endpoint=config.policy_endpoint, model_name=config.model_name)
        handle = AgentHandle(config=config, policy=policy)
        self._agents[config.name] = handle
        logger.info("Added agent: %s role=%s entity=%s", config.name, config.role.value, config.entity_id)
        return handle

    def remove_agent(self, name: str) -> bool:
        agent = self._agents.pop(name, None)
        if agent is None:
            return False
        agent.policy.close()
        logger.info("Removed agent: %s", name)
        return True

    def get_agent(self, name: str) -> AgentHandle | None:
        return self._agents.get(name)

    def list_agents(self) -> list[str]:
        return list(self._agents.keys())

    def collect_observations(self, agent_name: str) -> dict[str, Any]:
        agent = self._agents.get(agent_name)
        if agent is None:
            return {}
        obs: dict[str, Any] = {}
        if self._sensor_manager is not None:
            all_obs = self._sensor_manager.get_observations()
            if agent.config.obs_keys:
                for key in agent.config.obs_keys:
                    if key in all_obs:
                        obs[key] = all_obs[key]
            else:
                obs = all_obs
        obs["agent_name"] = agent_name
        obs["step_count"] = agent.step_count
        return obs

    def compute_action(self, agent_name: str) -> list[float]:
        agent = self._agents.get(agent_name)
        if agent is None:
            logger.warning("Agent not found: %s", agent_name)
            return []
        obs = self.collect_observations(agent_name)
        action = agent.policy.predict(obs, action_dim=agent.config.action_dim)
        action = self._scale_action(action, agent.config)
        agent.record_step(action)
        return action

    def observe_act_loop(self, agent_name: str) -> list[float]:
        agent = self._agents.get(agent_name)
        if agent is None:
            logger.warning("Agent not found: %s", agent_name)
            return []
        if agent.done:
            return agent.last_action
        if agent.step_count % agent.config.decimation != 0:
            return agent.last_action
        obs = self.collect_observations(agent_name)
        image = self._get_agent_image(agent_name)
        if image is not None:
            action = agent.policy.infer_from_image(
                image=image,
                prompt=f"Step {agent.step_count}: Given this camera view, output {agent.config.action_dim} action values as JSON array.",
                action_dim=agent.config.action_dim,
            )
        else:
            action = agent.policy.predict(obs, action_dim=agent.config.action_dim)
        action = self._scale_action(action, agent.config)
        agent.record_step(action)
        return action

    def _get_agent_image(self, agent_name: str) -> Any:
        if self._sensor_manager is None:
            return None
        agent = self._agents.get(agent_name)
        if agent is None:
            return None
        entity_id = agent.entity_id
        for sensor_name in self._sensor_manager.list_sensors():
            sensor = self._sensor_manager.get_sensor(sensor_name)
            if sensor is None:
                continue
            if sensor.entity_id and sensor.entity_id != entity_id:
                continue
            if not hasattr(sensor, "rgb"):
                continue
            rgb = sensor.rgb
            if rgb is not None:
                return rgb
        return None

    def _scale_action(self, action: list[float], config: AgentConfig) -> list[float]:
        scaled = [v * config.action_scale for v in action]
        if config.action_lower is not None:
            scaled = [max(v, lo) for v, lo in zip(scaled, config.action_lower)]
        if config.action_upper is not None:
            scaled = [min(v, hi) for v, hi in zip(scaled, config.action_upper)]
        return scaled

    def step_all(self) -> dict[str, list[float]]:
        actions = {}
        for name, agent in self._agents.items():
            if agent.done:
                continue
            if agent.step_count % agent.config.decimation != 0:
                actions[name] = agent.last_action
                continue
            actions[name] = self.compute_action(name)
        return actions

    def step_all_with_vision(self) -> dict[str, list[float]]:
        actions = {}
        for name in self._agents:
            actions[name] = self.observe_act_loop(name)
        return actions

    def reset_agent(self, name: str) -> bool:
        agent = self._agents.get(name)
        if agent is None:
            return False
        agent.reset()
        logger.info("Reset agent: %s", name)
        return True

    def reset_all(self) -> None:
        for agent in self._agents.values():
            agent.reset()
        logger.info("All agents reset")

    def close(self) -> None:
        for agent in self._agents.values():
            agent.policy.close()
        self._agents.clear()
        logger.info("AgentManager closed")

    def status(self) -> dict[str, Any]:
        return {
            "agent_count": len(self._agents),
            "agents": {
                name: {
                    "role": a.config.role.value,
                    "entity_id": a.entity_id,
                    "step_count": a.step_count,
                    "cumulative_reward": a.cumulative_reward,
                    "done": a.done,
                    "policy_available": a.policy.is_available,
                }
                for name, a in self._agents.items()
            },
        }
