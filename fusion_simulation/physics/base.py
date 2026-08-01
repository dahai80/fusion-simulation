from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PhysicsConfig:
    gravity: list[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    time_step: float = 0.01
    solver_iterations: int = 50
    num_sub_steps: int = 1
    seed: int = 42


@dataclass
class BodyState:
    body_id: int = -1
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    linear_velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angular_velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    joint_positions: list[float] = field(default_factory=list)
    joint_velocities: list[float] = field(default_factory=list)
    joint_efforts: list[float] = field(default_factory=list)


class PhysicsEngine(ABC):
    @abstractmethod
    def init(self, config: PhysicsConfig | None = None, headless: bool = True) -> None:
        ...

    @abstractmethod
    def step(self) -> None:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @abstractmethod
    def load_urdf(self, urdf_path: str, position: list[float] | None = None,
                  orientation: list[float] | None = None, fixed_base: bool = False,
                  use_fixed_base: bool = False) -> int:
        ...

    @abstractmethod
    def load_plane(self, position: list[float] | None = None) -> int:
        ...

    @abstractmethod
    def remove_body(self, body_id: int) -> None:
        ...

    @abstractmethod
    def get_body_state(self, body_id: int) -> BodyState:
        ...

    @abstractmethod
    def set_body_position(self, body_id: int, position: list[float],
                          orientation: list[float] | None = None) -> None:
        ...

    @abstractmethod
    def apply_force(self, body_id: int, force: list[float], position: list[float] | None = None) -> None:
        ...

    @abstractmethod
    def apply_joint_action(self, body_id: int, joint_indices: list[int],
                           values: list[float], mode: str = "position") -> None:
        ...

    @abstractmethod
    def get_joint_info(self, body_id: int) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def ray_test(self, origin: list[float], direction: list[float], max_dist: float = 100.0) -> dict[str, Any] | None:
        ...

    @abstractmethod
    def get_contact_points(self, body_id: int) -> list[dict[str, Any]]:
        ...

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        ...
