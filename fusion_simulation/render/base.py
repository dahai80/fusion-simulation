from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RenderConfig:
    width: int = 640
    height: int = 480
    fov: float = 60.0
    near: float = 0.1
    far: float = 100.0
    bg_color: list[float] = field(default_factory=lambda: [0.3, 0.3, 0.3, 1.0])


class RenderEngine(ABC):
    @abstractmethod
    def init(self, config: RenderConfig | None = None) -> None: ...

    @abstractmethod
    def render(self) -> None: ...

    @abstractmethod
    def capture_camera(
        self,
        target_position: list[float] | None = None,
        distance: float = 1.5,
        yaw: float = 45.0,
        pitch: float = -30.0,
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def is_initialized(self) -> bool: ...
