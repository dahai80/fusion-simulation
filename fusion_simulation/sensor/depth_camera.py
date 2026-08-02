from __future__ import annotations

import logging
from typing import Any

import numpy as np

from fusion_simulation.sensor.base import SensorBase, SensorConfig, SensorType
from fusion_simulation.sensor.manager import register_sensor

logger = logging.getLogger(__name__)


class DepthCameraSensor(SensorBase):
    def __init__(self, config: SensorConfig) -> None:
        super().__init__(config)
        params = config.params or {}
        self._width: int = params.get("width", 640)
        self._height: int = params.get("height", 480)
        self._fov: float = params.get("fov", 60.0)
        self._near: float = params.get("near", 0.01)
        self._far: float = params.get("far", 100.0)
        self._camera_position: list[float] | None = params.get("camera_position")
        self._camera_target: list[float] | None = params.get("camera_target")
        self._camera_up: list[float] | None = params.get("camera_up", [0, 0, 1])
        self._depth: np.ndarray | None = None
        self._point_cloud: np.ndarray | None = None

    @property
    def depth(self) -> np.ndarray | None:
        return self._depth

    @property
    def point_cloud(self) -> np.ndarray | None:
        return self._point_cloud

    def _capture(self, sim_time: float, physics_engine: Any = None) -> dict[str, Any]:
        if physics_engine is None:
            logger.warning("DepthCameraSensor: no physics engine for capture")
            return {"depth": None, "width": self._width, "height": self._height, "near": self._near, "far": self._far}
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
            depth_raw = img.get("depth")
            if depth_raw is not None:
                self._depth = np.asarray(depth_raw, dtype=np.float32)
                if self._depth.ndim == 3:
                    self._depth = self._depth[:, :, 0]
                self._depth = np.clip(self._depth, self._near, self._far)
                self._point_cloud = self._depth_to_point_cloud(self._depth)
            else:
                self._depth = np.zeros((self._height, self._width), dtype=np.float32)
                self._point_cloud = None
        except Exception:
            logger.exception("DepthCameraSensor capture failed")
            return {"depth": None, "width": self._width, "height": self._height, "near": self._near, "far": self._far}
        return {
            "depth": self._depth is not None,
            "width": self._width,
            "height": self._height,
            "near": self._near,
            "far": self._far,
            "min_depth": float(np.min(self._depth)) if self._depth is not None else 0.0,
            "max_depth": float(np.max(self._depth)) if self._depth is not None else 0.0,
        }

    def _depth_to_point_cloud(self, depth: np.ndarray) -> np.ndarray:
        h, w = depth.shape
        fx = fy = w / (2.0 * np.tan(np.radians(self._fov / 2.0)))
        cx, cy = w / 2.0, h / 2.0
        u = np.arange(w, dtype=np.float32)
        v = np.arange(h, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        x = (uu - cx) * depth / fx
        y = (vv - cy) * depth / fy
        z = depth
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        return points

    def get_observation(self) -> dict[str, Any]:
        obs = super().get_observation()
        obs["shape"] = {"width": self._width, "height": self._height}
        obs["near"] = self._near
        obs["far"] = self._far
        return obs

    def reset(self) -> None:
        super().reset()
        self._depth = None
        self._point_cloud = None


register_sensor(SensorType.DEPTH_CAMERA, DepthCameraSensor)
