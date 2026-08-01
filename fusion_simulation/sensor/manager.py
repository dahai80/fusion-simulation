from __future__ import annotations

import logging
from typing import Any

from fusion_simulation.sensor.base import SensorBase, SensorConfig, SensorType

logger = logging.getLogger(__name__)

_SENSOR_REGISTRY: dict[SensorType, type[SensorBase]] = {}


def register_sensor(sensor_type: SensorType, cls: type[SensorBase]) -> None:
    _SENSOR_REGISTRY[sensor_type] = cls
    logger.debug("Registered sensor type: %s -> %s", sensor_type.value, cls.__name__)


def create_sensor(config: SensorConfig) -> SensorBase:
    cls = _SENSOR_REGISTRY.get(config.sensor_type)
    if cls is None:
        raise ValueError(f"Unknown sensor type: {config.sensor_type}")
    return cls(config)


class SensorManager:
    def __init__(self, physics_engine: Any = None) -> None:
        self._sensors: dict[str, SensorBase] = {}
        self._physics_engine = physics_engine
        self._sim_time: float = 0.0

    @property
    def sensor_count(self) -> int:
        return len(self._sensors)

    def set_physics_engine(self, engine: Any) -> None:
        self._physics_engine = engine

    def add_sensor(self, config: SensorConfig) -> SensorBase:
        if config.name in self._sensors:
            logger.warning("Sensor already exists: %s, replacing", config.name)
            self.remove_sensor(config.name)
        sensor = create_sensor(config)
        self._sensors[config.name] = sensor
        logger.info("Added sensor: %s type=%s entity=%s", config.name, config.sensor_type.value, config.entity_id)
        return sensor

    def remove_sensor(self, name: str) -> bool:
        if name not in self._sensors:
            return False
        del self._sensors[name]
        logger.info("Removed sensor: %s", name)
        return True

    def get_sensor(self, name: str) -> SensorBase | None:
        return self._sensors.get(name)

    def list_sensors(self) -> list[str]:
        return list(self._sensors.keys())

    def update(self) -> None:
        for name, sensor in self._sensors.items():
            try:
                sensor.update(self._sim_time, self._physics_engine)
            except Exception:
                logger.exception("Error updating sensor: %s", name)

    def set_sim_time(self, sim_time: float) -> None:
        self._sim_time = sim_time

    def get_observations(self) -> dict[str, dict[str, Any]]:
        return {name: s.get_observation() for name, s in self._sensors.items()}

    def get_sensor_data(self, name: str) -> dict[str, Any] | None:
        sensor = self._sensors.get(name)
        return sensor.data if sensor else None

    def reset(self) -> None:
        for sensor in self._sensors.values():
            sensor.reset()
        self._sim_time = 0.0
        logger.info("SensorManager reset")

    def enable_sensor(self, name: str, enabled: bool = True) -> bool:
        sensor = self._sensors.get(name)
        if sensor is None:
            return False
        sensor.enabled = enabled
        return True

    def status(self) -> dict[str, Any]:
        return {
            "sensor_count": len(self._sensors),
            "sensors": {
                name: {
                    "type": s.sensor_type.value,
                    "enabled": s.enabled,
                    "frame_count": s.frame_count,
                    "last_update": s.last_update_time,
                }
                for name, s in self._sensors.items()
            },
        }
