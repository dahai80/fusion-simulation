from __future__ import annotations

import logging
from typing import Any

from fusion_simulation.render.base import RenderConfig, RenderEngine

logger = logging.getLogger(__name__)


class PyBulletRender(RenderEngine):
    def __init__(self, physics_engine: Any | None = None) -> None:
        self._physics = physics_engine
        self._config: RenderConfig = RenderConfig()
        self._initialized: bool = False

    def init(self, config: RenderConfig | None = None) -> None:
        self._config = config or RenderConfig()
        self._initialized = True
        logger.info("PyBulletRender initialized: %dx%d", self._config.width, self._config.height)

    def render(self) -> None:
        pass

    def capture_camera(
        self,
        target_position: list[float] | None = None,
        distance: float = 1.5,
        yaw: float = 45.0,
        pitch: float = -30.0,
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, Any]:
        if not self._initialized:
            logger.warning("PyBulletRender not initialized")
            return {"rgb": None, "depth": None, "segmentation": None, "width": 0, "height": 0}
        if self._physics is None:
            logger.warning("No physics engine attached for camera capture")
            return {"rgb": None, "depth": None, "segmentation": None, "width": 0, "height": 0}
        w = width or self._config.width
        h = height or self._config.height
        return self._physics.get_camera_image(
            width=w, height=h,
            target_position=target_position or [0.0, 0.0, 0.0],
            distance=distance, yaw=yaw, pitch=pitch,
            fov=self._config.fov, near=self._config.near, far=self._config.far,
        )

    def close(self) -> None:
        self._initialized = False
        logger.info("PyBulletRender closed")

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def attach_physics(self, physics_engine: Any) -> None:
        self._physics = physics_engine
        logger.debug("Physics engine attached to PyBulletRender")
