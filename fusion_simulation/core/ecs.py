from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="Component")


@dataclass(frozen=True)
class EntityId:
    value: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    def __str__(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, EntityId):
            return self.value == other.value
        return NotImplemented


@dataclass
class Component:
    entity_id: EntityId = field(default_factory=EntityId)


@dataclass
class Transform(Component):
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])


@dataclass
class RigidBody(Component):
    mass: float = 1.0
    friction: float = 0.5
    restitution: float = 0.0
    linear_velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angular_velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    kinematic: bool = False


@dataclass
class Articulation(Component):
    joint_names: list[str] = field(default_factory=list)
    joint_positions: list[float] = field(default_factory=list)
    joint_velocities: list[float] = field(default_factory=list)
    joint_efforts: list[float] = field(default_factory=list)
    joint_lower_limits: list[float] = field(default_factory=list)
    joint_upper_limits: list[float] = field(default_factory=list)
    urdf_path: str = ""


@dataclass
class AgentBind(Component):
    agent_id: str = ""
    policy_type: str = "rule"
    model_name: str = ""
    sensors: list[str] = field(default_factory=list)
    action_dim: int = 0


@dataclass
class CameraSensor(Component):
    width: int = 640
    height: int = 480
    fov: float = 60.0
    near: float = 0.1
    far: float = 100.0
    target_position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    distance: float = 1.5
    yaw: float = 45.0
    pitch: float = -30.0


@dataclass
class IMUSensor(Component):
    accelerometer_noise: float = 0.01
    gyroscope_noise: float = 0.01


_COMPONENT_REGISTRY: dict[str, type[Component]] = {
    "Transform": Transform,
    "RigidBody": RigidBody,
    "Articulation": Articulation,
    "AgentBind": AgentBind,
    "CameraSensor": CameraSensor,
    "IMUSensor": IMUSensor,
}


class EntityManager:
    def __init__(self) -> None:
        self._entities: dict[EntityId, dict[type[Component], Component]] = {}
        self._next_id: int = 0
        logger.info("EntityManager created")

    def create_entity(self, entity_id: EntityId | None = None) -> EntityId:
        if entity_id is None:
            entity_id = EntityId()
        self._entities[entity_id] = {}
        logger.debug("Entity created: %s", entity_id)
        return entity_id

    def destroy_entity(self, entity_id: EntityId) -> bool:
        if entity_id in self._entities:
            del self._entities[entity_id]
            logger.debug("Entity destroyed: %s", entity_id)
            return True
        return False

    def add_component(self, entity_id: EntityId, component: Component) -> None:
        if entity_id not in self._entities:
            raise KeyError(f"Entity {entity_id} does not exist")
        component.entity_id = entity_id
        comp_type = type(component)
        self._entities[entity_id][comp_type] = component
        logger.debug("Component %s added to entity %s", comp_type.__name__, entity_id)

    def remove_component(self, entity_id: EntityId, comp_type: type[T]) -> bool:
        if entity_id not in self._entities:
            return False
        if comp_type in self._entities[entity_id]:
            del self._entities[entity_id][comp_type]
            logger.debug("Component %s removed from entity %s", comp_type.__name__, entity_id)
            return True
        return False

    def get_component(self, entity_id: EntityId, comp_type: type[T]) -> T | None:
        if entity_id not in self._entities:
            return None
        return self._entities[entity_id].get(comp_type)

    def get_components(self, entity_id: EntityId) -> dict[type[Component], Component]:
        return dict(self._entities.get(entity_id, {}))

    def has_component(self, entity_id: EntityId, comp_type: type[Component]) -> bool:
        return entity_id in self._entities and comp_type in self._entities[entity_id]

    def query(self, *comp_types: type[Component]) -> list[tuple[EntityId, tuple[Component, ...]]]:
        results = []
        for eid, comps in self._entities.items():
            if all(ct in comps for ct in comp_types):
                results.append((eid, tuple(comps[ct] for ct in comp_types)))
        return results

    def list_entities(self) -> list[EntityId]:
        return list(self._entities.keys())

    def entity_count(self) -> int:
        return len(self._entities)

    def clear(self) -> None:
        self._entities.clear()
        logger.info("EntityManager cleared")

    def serialize_entity(self, entity_id: EntityId) -> dict[str, Any] | None:
        if entity_id not in self._entities:
            return None
        comps = self._entities[entity_id]
        return {
            "entity_id": str(entity_id),
            "components": {ct.__name__: _serialize_component(c) for ct, c in comps.items()},
        }

    def serialize_all(self) -> list[dict[str, Any]]:
        return [self.serialize_entity(eid) for eid in self._entities if self.serialize_entity(eid) is not None]


def _serialize_component(comp: Component) -> dict[str, Any]:
    result: dict[str, Any] = {"_type": type(comp).__name__}
    for f in comp.__dataclass_fields__:
        if f == "entity_id":
            continue
        val = getattr(comp, f)
        result[f] = val
    return result


def deserialize_component(data: dict[str, Any]) -> Component | None:
    comp_type_name = data.get("_type", "")
    comp_cls = _COMPONENT_REGISTRY.get(comp_type_name)
    if comp_cls is None:
        logger.warning("Unknown component type: %s", comp_type_name)
        return None
    kwargs = {k: v for k, v in data.items() if k != "_type" and k != "entity_id"}
    try:
        return comp_cls(**kwargs)
    except Exception as e:
        logger.error("Failed to deserialize component %s: %s", comp_type_name, e)
        return None
