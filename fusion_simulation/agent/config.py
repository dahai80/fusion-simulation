from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    ROBOT = "robot"
    OBSERVER = "observer"
    CONTROLLER = "controller"


@dataclass
class AgentConfig:
    name: str = ""
    role: AgentRole = AgentRole.ROBOT
    entity_id: str = ""
    policy_endpoint: str = "http://localhost:11434/v1/chat/completions"
    model_name: str = "qwen3.5-9b"
    action_dim: int = 0
    obs_keys: list[str] = field(default_factory=list)
    decimation: int = 1
    action_scale: float = 1.0
    action_lower: list[float] | None = None
    action_upper: list[float] | None = None
    params: dict[str, Any] = field(default_factory=dict)
