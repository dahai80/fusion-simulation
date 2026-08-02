from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fusion_simulation.agent.config import AgentConfig, AgentRole

logger = logging.getLogger(__name__)


@dataclass
class TaskTemplate:
    name: str
    description: str
    agent_configs: list[AgentConfig]
    reward_spec: dict[str, Any] = field(default_factory=dict)
    termination_spec: dict[str, Any] = field(default_factory=dict)
    scene_name: str = "default"
    max_steps: int = 1000
    metadata: dict[str, Any] = field(default_factory=dict)
    reward_fn: Callable | None = None
    termination_fn: Callable | None = None
    reset_fn: Callable | None = None


_BUILTIN_TEMPLATES: dict[str, TaskTemplate] = {}


def register_template(template: TaskTemplate) -> None:
    _BUILTIN_TEMPLATES[template.name] = template
    logger.info("Task template registered: %s", template.name)


def get_template(name: str) -> TaskTemplate | None:
    return _BUILTIN_TEMPLATES.get(name)


def list_templates() -> list[dict[str, str]]:
    return [{"name": t.name, "description": t.description, "scene": t.scene_name} for t in _BUILTIN_TEMPLATES.values()]


def _reward_distance_to_goal(obs: dict[str, Any], info: dict[str, Any]) -> float:
    spec = info.get("reward_spec", {})
    weight = spec.get("weight", 1.0)
    goal = spec.get("goal", [0.5, 0.0, 0.5])
    agent_pos = obs.get("position", [0, 0, 0])
    if isinstance(agent_pos, (list, tuple, np.ndarray)):
        pos = np.array(agent_pos[:3], dtype=np.float64)
    else:
        pos = np.zeros(3)
    dist = np.linalg.norm(pos - np.array(goal))
    return float(-dist * weight)


def _termination_goal_reached(obs: dict[str, Any], info: dict[str, Any]) -> bool:
    spec = info.get("termination_spec", {})
    threshold = spec.get("threshold", 0.05)
    goal = spec.get("goal", [0.5, 0.0, 0.5])
    max_steps = info.get("max_steps", 1000)
    step = info.get("step", 0)
    agent_pos = obs.get("position", [0, 0, 0])
    if isinstance(agent_pos, (list, tuple, np.ndarray)):
        pos = np.array(agent_pos[:3], dtype=np.float64)
    else:
        pos = np.zeros(3)
    dist = np.linalg.norm(pos - np.array(goal))
    if dist < threshold:
        logger.debug("Goal reached: dist=%.4f < threshold=%.4f", dist, threshold)
        return True
    if step >= max_steps:
        logger.debug("Max steps reached: %d >= %d", step, max_steps)
        return True
    return False


def _reward_push_distance(obs: dict[str, Any], info: dict[str, Any]) -> float:
    spec = info.get("reward_spec", {})
    weight = spec.get("weight", 1.0)
    target_pos = spec.get("target", [0.8, 0.0, 0.0])
    object_pos = obs.get("object_position", [0, 0, 0])
    if isinstance(object_pos, (list, tuple, np.ndarray)):
        pos = np.array(object_pos[:3], dtype=np.float64)
    else:
        pos = np.zeros(3)
    dist = np.linalg.norm(pos - np.array(target_pos))
    return float(-dist * weight)


def _reward_cooperative(obs: dict[str, Any], info: dict[str, Any]) -> float:
    spec = info.get("reward_spec", {})
    weight = spec.get("weight", 1.0)
    positions = obs.get("agent_positions", [])
    if len(positions) < 2:
        return 0.0
    p0 = np.array(positions[0][:3], dtype=np.float64)
    p1 = np.array(positions[1][:3], dtype=np.float64)
    goal = spec.get("goal", [0.5, 0.0, 0.5])
    g = np.array(goal, dtype=np.float64)
    d0 = np.linalg.norm(p0 - g)
    d1 = np.linalg.norm(p1 - g)
    avg_dist = (d0 + d1) / 2.0
    return float(-avg_dist * weight)


def _termination_both_reached(obs: dict[str, Any], info: dict[str, Any]) -> bool:
    spec = info.get("termination_spec", {})
    threshold = spec.get("threshold", 0.05)
    max_steps = info.get("max_steps", 1000)
    step = info.get("step", 0)
    if step >= max_steps:
        return True
    positions = obs.get("agent_positions", [])
    if len(positions) < 2:
        return False
    goal = spec.get("goal", [0.5, 0.0, 0.5])
    g = np.array(goal, dtype=np.float64)
    for p in positions:
        if np.linalg.norm(np.array(p[:3], dtype=np.float64) - g) > threshold:
            return False
    return True


def _reward_task_completion(obs: dict[str, Any], info: dict[str, Any]) -> float:
    spec = info.get("reward_spec", {})
    weight = spec.get("weight", 1.0)
    task_progress = obs.get("task_progress", 0.0)
    return float(task_progress * weight)


def _termination_task_done(obs: dict[str, Any], info: dict[str, Any]) -> bool:
    spec = info.get("termination_spec", {})
    max_steps = info.get("max_steps", 1000)
    step = info.get("step", 0)
    if step >= max_steps:
        return True
    task_progress = obs.get("task_progress", 0.0)
    threshold = spec.get("threshold", 0.05)
    if task_progress >= (1.0 - threshold):
        return True
    return False


def _init_builtin_templates() -> None:
    register_template(
        TaskTemplate(
            name="single_robot_pick",
            description="Single robot pick-and-place task",
            agent_configs=[
                AgentConfig(name="robot0", role=AgentRole.ROBOT, action_dim=6),
            ],
            reward_spec={"type": "distance_to_goal", "weight": 1.0, "goal": [0.5, 0.0, 0.5]},
            termination_spec={"type": "goal_reached", "threshold": 0.05, "goal": [0.5, 0.0, 0.5]},
            scene_name="pick",
            max_steps=500,
            reward_fn=_reward_distance_to_goal,
            termination_fn=_termination_goal_reached,
        )
    )
    register_template(
        TaskTemplate(
            name="single_robot_push",
            description="Single robot pushing task",
            agent_configs=[
                AgentConfig(name="robot0", role=AgentRole.ROBOT, action_dim=6),
            ],
            reward_spec={"type": "push_distance", "weight": 1.0, "target": [0.8, 0.0, 0.0]},
            termination_spec={"type": "goal_reached", "threshold": 0.1, "goal": [0.8, 0.0, 0.0]},
            scene_name="push",
            max_steps=500,
            reward_fn=_reward_push_distance,
            termination_fn=_termination_goal_reached,
        )
    )
    register_template(
        TaskTemplate(
            name="dual_robot_coop",
            description="Two robots cooperative task",
            agent_configs=[
                AgentConfig(name="robot0", role=AgentRole.ROBOT, action_dim=6),
                AgentConfig(name="robot1", role=AgentRole.ROBOT, action_dim=6),
            ],
            reward_spec={"type": "cooperative_reward", "weight": 1.0, "goal": [0.5, 0.0, 0.5]},
            termination_spec={"type": "both_reached", "threshold": 0.05, "goal": [0.5, 0.0, 0.5]},
            scene_name="default",
            max_steps=1000,
            reward_fn=_reward_cooperative,
            termination_fn=_termination_both_reached,
        )
    )
    register_template(
        TaskTemplate(
            name="robot_with_observer",
            description="Robot with observer monitoring",
            agent_configs=[
                AgentConfig(name="robot0", role=AgentRole.ROBOT, action_dim=6),
                AgentConfig(name="observer0", role=AgentRole.OBSERVER, action_dim=0),
            ],
            reward_spec={"type": "distance_to_goal", "weight": 1.0, "goal": [0.5, 0.0, 0.5]},
            termination_spec={"type": "goal_reached", "threshold": 0.05, "goal": [0.5, 0.0, 0.5]},
            scene_name="pick",
            max_steps=500,
            reward_fn=_reward_distance_to_goal,
            termination_fn=_termination_goal_reached,
        )
    )
    register_template(
        TaskTemplate(
            name="controller_robot",
            description="Controller directing a robot",
            agent_configs=[
                AgentConfig(name="controller0", role=AgentRole.CONTROLLER, action_dim=6),
                AgentConfig(name="robot0", role=AgentRole.ROBOT, action_dim=6),
            ],
            reward_spec={"type": "task_completion", "weight": 1.0},
            termination_spec={"type": "task_done", "threshold": 0.05},
            scene_name="default",
            max_steps=1000,
            reward_fn=_reward_task_completion,
            termination_fn=_termination_task_done,
        )
    )
    logger.info("Builtin task templates initialized: %d templates", len(_BUILTIN_TEMPLATES))


_init_builtin_templates()
