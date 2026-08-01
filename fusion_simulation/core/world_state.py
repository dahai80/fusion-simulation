from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EntitySnapshot:
    entity_id: str
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    linear_velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angular_velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    joint_positions: list[float] = field(default_factory=list)
    joint_velocities: list[float] = field(default_factory=list)
    active: bool = True


@dataclass
class WorldState:
    sim_time: float = 0.0
    frame_count: int = 0
    entities: dict[str, EntitySnapshot] = field(default_factory=dict)
    sensor_readings: dict[str, Any] = field(default_factory=dict)

    def get_entity(self, entity_id: str) -> EntitySnapshot | None:
        return self.entities.get(entity_id)

    def set_entity(self, entity_id: str, snapshot: EntitySnapshot) -> None:
        self.entities[entity_id] = snapshot

    def remove_entity(self, entity_id: str) -> bool:
        if entity_id in self.entities:
            del self.entities[entity_id]
            return True
        return False

    def snapshot(self) -> WorldState:
        return WorldState(
            sim_time=self.sim_time,
            frame_count=self.frame_count,
            entities={eid: copy.deepcopy(es) for eid, es in self.entities.items()},
            sensor_readings=copy.deepcopy(self.sensor_readings),
        )

    def restore(self, other: WorldState) -> None:
        self.sim_time = other.sim_time
        self.frame_count = other.frame_count
        self.entities = {eid: copy.deepcopy(es) for eid, es in other.entities.items()}
        self.sensor_readings = copy.deepcopy(other.sensor_readings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sim_time": self.sim_time,
            "frame_count": self.frame_count,
            "entities": {
                eid: {
                    "position": es.position,
                    "orientation": es.orientation,
                    "linear_velocity": es.linear_velocity,
                    "angular_velocity": es.angular_velocity,
                    "joint_positions": es.joint_positions,
                    "joint_velocities": es.joint_velocities,
                    "active": es.active,
                }
                for eid, es in self.entities.items()
            },
        }

    def entity_count(self) -> int:
        return len(self.entities)

    def active_entity_count(self) -> int:
        return sum(1 for es in self.entities.values() if es.active)
