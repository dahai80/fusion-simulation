from __future__ import annotations

import logging
from typing import Any

import numpy as np

from fusion_simulation.sensor.base import SensorBase, SensorConfig, SensorType
from fusion_simulation.sensor.manager import register_sensor

logger = logging.getLogger(__name__)


class RgbCameraSensor(SensorBase):
    def __init__(self, config: SensorConfig) -> None:
        super().__init__(config)
        params = config.params or {}
        self._width: int = params.get("width", 320)
        self._height: int = params.get("height", 240)
        self._fov: float = params.get("fov", 60.0)
        self._near: float = params.get("near", 0.01)
        self._far: float = params.get("far", 100.0)
        self._camera_position: list[float] | None = params.get("camera_position")
        self._camera_target: list[float] | None = params.get("camera_target")
        self._camera_up: list[float] | None = params.get("camera_up", [0, 0, 1])
        self._rgb: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._segmentation: np.ndarray | None = None

    @property
    def rgb(self) -> np.ndarray | None:
        return self._rgb

    @property
    def depth(self) -> np.ndarray | None:
        return self._depth

    @property
    def segmentation(self) -> np.ndarray | None:
        return self._segmentation

    def _capture(self, sim_time: float, physics_engine: Any = None) -> dict[str, Any]:
        if physics_engine is None:
            logger.warning("RgbCameraSensor: no physics engine for capture")
            return {"rgb": None, "depth": None, "segmentation": None, "width": self._width, "height": self._height}
        cam_pos = self._camera_position or [0, 0, 1]
        cam_target = self._camera_target or [0, 0, 0]
        cam_up = self._camera_up or [0, 0, 1]
        try:
            img = physics_engine.get_camera_image(
                width=self._width,
                height=self._height,
                fov=self._fov,
                camera_position=cam_pos,
                camera_target=cam_target,
                camera_up=cam_up,
                near=self._near,
                far=self._far,
            )
            self._rgb = img.get("rgb")
            self._depth = img.get("depth")
            self._segmentation = img.get("segmentation")
        except Exception:
            logger.exception("RgbCameraSensor capture failed")
            return {"rgb": None, "depth": None, "segmentation": None, "width": self._width, "height": self._height}
        return {
            "rgb": self._rgb is not None,
            "depth": self._depth is not None,
            "segmentation": self._segmentation is not None,
            "width": self._width,
            "height": self._height,
        }

    def get_observation(self) -> dict[str, Any]:
        obs = super().get_observation()
        obs["shape"] = {"width": self._width, "height": self._height}
        return obs

    def reset(self) -> None:
        super().reset()
        self._rgb = None
        self._depth = None
        self._segmentation = None


register_sensor(SensorType.RGB_CAMERA, RgbCameraSensor)
