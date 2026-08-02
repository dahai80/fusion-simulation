from __future__ import annotations

import logging
from typing import Any

import numpy as np

from fusion_simulation.sensor.base import SensorBase, SensorConfig, SensorType
from fusion_simulation.sensor.manager import register_sensor

logger = logging.getLogger(__name__)


class ImuSensor(SensorBase):
    def __init__(self, config: SensorConfig) -> None:
        super().__init__(config)
        params = config.params or {}
        self._accel_noise: float = params.get("accelerometer_noise", 0.01)
        self._gyro_noise: float = params.get("gyroscope_noise", 0.001)
        self._linear_acceleration: list[float] = [0.0, 0.0, -9.81]
        self._angular_velocity: list[float] = [0.0, 0.0, 0.0]
        self._orientation_quat: list[float] = [0.0, 0.0, 0.0, 1.0]

    @property
    def linear_acceleration(self) -> list[float]:
        return self._linear_acceleration

    @property
    def angular_velocity(self) -> list[float]:
        return self._angular_velocity

    @property
    def orientation(self) -> list[float]:
        return self._orientation_quat

    def _capture(self, sim_time: float, physics_engine: Any = None) -> dict[str, Any]:
        if physics_engine is None or not physics_engine.is_initialized:
            logger.warning("ImuSensor: no physics engine for capture")
            return {
                "linear_acceleration": self._linear_acceleration,
                "angular_velocity": self._angular_velocity,
            }
        entity_id = self._config.entity_id
        if not entity_id:
            return {
                "linear_acceleration": self._add_noise([0.0, 0.0, -9.81], self._accel_noise),
                "angular_velocity": self._add_noise([0.0, 0.0, 0.0], self._gyro_noise),
            }
        try:
            body_id = self._resolve_body_id(physics_engine, entity_id)
            if body_id is not None:
                state = physics_engine.get_body_state(body_id)
                if state is not None:
                    self._linear_acceleration = self._add_noise(
                        list(state.linear_velocity),
                        self._accel_noise,
                    )
                    self._angular_velocity = self._add_noise(
                        list(state.angular_velocity),
                        self._gyro_noise,
                    )
                    self._orientation_quat = list(state.orientation)
        except Exception:
            logger.exception("ImuSensor capture failed")
        return {
            "linear_acceleration": self._linear_acceleration,
            "angular_velocity": self._angular_velocity,
            "orientation": self._orientation_quat,
        }

    def _add_noise(self, values: list[float], scale: float) -> list[float]:
        if scale <= 0:
            return values
        return [v + float(np.random.normal(0, scale)) for v in values]

    def _resolve_body_id(self, physics_engine: Any, entity_id: str) -> int | None:
        try:
            body_ids = physics_engine.get_all_body_ids()
            for bid in body_ids:
                state = physics_engine.get_body_state(bid)
                if state is not None and str(bid) == entity_id:
                    return bid
        except Exception:
            pass
        return None

    def get_observation(self) -> dict[str, Any]:
        obs = super().get_observation()
        obs["linear_acceleration"] = self._linear_acceleration
        obs["angular_velocity"] = self._angular_velocity
        return obs

    def reset(self) -> None:
        super().reset()
        self._linear_acceleration = [0.0, 0.0, -9.81]
        self._angular_velocity = [0.0, 0.0, 0.0]
        self._orientation_quat = [0.0, 0.0, 0.0, 1.0]


register_sensor(SensorType.IMU, ImuSensor)
