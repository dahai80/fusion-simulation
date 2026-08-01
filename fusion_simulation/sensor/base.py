from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SensorType(str, Enum):
    RGB_CAMERA = "rgb_camera"
    DEPTH_CAMERA = "depth_camera"
    SEGMENTATION_CAMERA = "segmentation_camera"
    IMU = "imu"
    FORCE_TORQUE = "force_torque"
    JOINT_ENCODER = "joint_encoder"
    CONTACT = "contact"


@dataclass
class SensorConfig:
    sensor_type: SensorType = SensorType.RGB_CAMERA
    entity_id: str = ""
    name: str = ""
    update_rate: float = 30.0
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)


class SensorBase:
    def __init__(self, config: SensorConfig) -> None:
        self._config = config
        self._data: dict[str, Any] = {}
        self._last_update_time: float = -1.0
        self._frame_count: int = 0
        self._enabled: bool = config.enabled

    @property
    def config(self) -> SensorConfig:
        return self._config

    @property
    def sensor_type(self) -> SensorType:
        return self._config.sensor_type

    @property
    def entity_id(self) -> str:
        return self._config.entity_id

    @property
    def name(self) -> str:
        return self._config.name or f"{self._config.sensor_type.value}_{self._config.entity_id}"

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    @property
    def last_update_time(self) -> float:
        return self._last_update_time

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def should_update(self, sim_time: float) -> bool:
        if not self._enabled:
            return False
        if self._config.update_rate <= 0:
            return True
        interval = 1.0 / self._config.update_rate
        return (sim_time - self._last_update_time) >= interval

    def update(self, sim_time: float, physics_engine: Any = None) -> dict[str, Any]:
        if not self.should_update(sim_time):
            return self._data
        self._data = self._capture(sim_time, physics_engine)
        self._last_update_time = sim_time
        self._frame_count += 1
        return self._data

    def _capture(self, sim_time: float, physics_engine: Any = None) -> dict[str, Any]:
        raise NotImplementedError

    def reset(self) -> None:
        self._data = {}
        self._last_update_time = -1.0
        self._frame_count = 0

    def get_observation(self) -> dict[str, Any]:
        return {"type": self.sensor_type.value, "name": self.name, "data": self._data}
