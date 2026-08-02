from __future__ import annotations

import logging
from typing import Any

import numpy as np

from fusion_simulation.sensor.base import SensorBase, SensorConfig, SensorType
from fusion_simulation.sensor.manager import register_sensor

logger = logging.getLogger(__name__)


class SemanticCameraSensor(SensorBase):
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
        self._segmentation: np.ndarray | None = None
        self._instance_seg: np.ndarray | None = None
        self._label_map: dict[int, str] = {}

    @property
    def segmentation(self) -> np.ndarray | None:
        return self._segmentation

    @property
    def instance_segmentation(self) -> np.ndarray | None:
        return self._instance_seg

    @property
    def label_map(self) -> dict[int, str]:
        return self._label_map

    def _capture(self, sim_time: float, physics_engine: Any = None) -> dict[str, Any]:
        if physics_engine is None:
            logger.warning("SemanticCameraSensor: no physics engine for capture")
            return {"segmentation": None, "width": self._width, "height": self._height, "num_labels": 0}
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
            seg_raw = img.get("segmentation")
            if seg_raw is not None:
                self._segmentation = np.asarray(seg_raw, dtype=np.int32)
                if self._segmentation.ndim == 3:
                    self._segmentation = self._segmentation[:, :, 0]
                unique_labels = np.unique(self._segmentation)
                for label in unique_labels:
                    label_int = int(label)
                    if label_int not in self._label_map:
                        self._label_map[label_int] = f"object_{label_int}"
                self._instance_seg = self._segmentation.copy()
            else:
                self._segmentation = np.zeros((self._height, self._width), dtype=np.int32)
                self._instance_seg = np.zeros((self._height, self._width), dtype=np.int32)
        except Exception:
            logger.exception("SemanticCameraSensor capture failed")
            return {"segmentation": None, "width": self._width, "height": self._height, "num_labels": 0}
        return {
            "segmentation": self._segmentation is not None,
            "width": self._width,
            "height": self._height,
            "num_labels": len(self._label_map),
            "unique_labels": list(self._label_map.keys()),
        }

    def set_label_map(self, label_map: dict[int, str]) -> None:
        self._label_map = dict(label_map)
        logger.debug("SemanticCamera label map updated: %d labels", len(self._label_map))

    def get_observation(self) -> dict[str, Any]:
        obs = super().get_observation()
        obs["shape"] = {"width": self._width, "height": self._height}
        obs["num_labels"] = len(self._label_map)
        return obs

    def reset(self) -> None:
        super().reset()
        self._segmentation = None
        self._instance_seg = None
        self._label_map.clear()


register_sensor(SensorType.SEGMENTATION_CAMERA, SemanticCameraSensor)
