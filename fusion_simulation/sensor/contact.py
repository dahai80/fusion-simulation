from __future__ import annotations

import logging
from typing import Any

from fusion_simulation.sensor.base import SensorBase, SensorConfig, SensorType
from fusion_simulation.sensor.manager import register_sensor

logger = logging.getLogger(__name__)


class ContactSensor(SensorBase):
    def __init__(self, config: SensorConfig) -> None:
        super().__init__(config)
        params = config.params or {}
        self._force_threshold: float = params.get("force_threshold", 0.01)
        self._contacts: list[dict[str, Any]] = []
        self._total_force: float = 0.0
        self._in_contact: bool = False

    @property
    def contacts(self) -> list[dict[str, Any]]:
        return self._contacts

    @property
    def total_force(self) -> float:
        return self._total_force

    @property
    def in_contact(self) -> bool:
        return self._in_contact

    def _capture(self, sim_time: float, physics_engine: Any = None) -> dict[str, Any]:
        if physics_engine is None or not physics_engine.is_initialized:
            logger.warning("ContactSensor: no physics engine for capture")
            return {"contacts": [], "total_force": 0.0, "in_contact": False}
        entity_id = self._config.entity_id
        if not entity_id:
            return {"contacts": [], "total_force": 0.0, "in_contact": False}
        try:
            body_id = self._resolve_body_id(physics_engine, entity_id)
            if body_id is not None:
                raw_contacts = physics_engine.get_contact_points(body_id)
                self._contacts = [
                    {
                        "body_a": c.get("body_a", -1),
                        "body_b": c.get("body_b", -1),
                        "contact_normal": c.get("contact_normal", [0.0, 0.0, 0.0]),
                        "contact_distance": c.get("contact_distance", 0.0),
                        "normal_force": c.get("normal_force", 0.0),
                        "contact_position": c.get("contact_position", [0.0, 0.0, 0.0]),
                        "lateral_friction_force_1": c.get("lateral_friction_force_1", 0.0),
                        "lateral_friction_force_2": c.get("lateral_friction_force_2", 0.0),
                    }
                    for c in raw_contacts
                ]
                self._total_force = sum(c.get("normal_force", 0.0) for c in self._contacts)
                self._in_contact = self._total_force >= self._force_threshold
            else:
                self._contacts = []
                self._total_force = 0.0
                self._in_contact = False
        except Exception:
            logger.exception("ContactSensor capture failed")
            self._contacts = []
            self._total_force = 0.0
            self._in_contact = False
        return {
            "contacts": len(self._contacts),
            "total_force": self._total_force,
            "in_contact": self._in_contact,
        }

    def _resolve_body_id(self, physics_engine: Any, entity_id: str) -> int | None:
        try:
            body_ids = physics_engine.get_all_body_ids()
            for bid in body_ids:
                if str(bid) == entity_id:
                    return bid
        except Exception:
            pass
        return None

    def get_observation(self) -> dict[str, Any]:
        obs = super().get_observation()
        obs["in_contact"] = self._in_contact
        obs["total_force"] = self._total_force
        obs["contact_count"] = len(self._contacts)
        return obs

    def reset(self) -> None:
        super().reset()
        self._contacts = []
        self._total_force = 0.0
        self._in_contact = False


register_sensor(SensorType.CONTACT, ContactSensor)
