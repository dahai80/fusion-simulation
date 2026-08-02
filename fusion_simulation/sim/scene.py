from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fusion_simulation.core.ecs import (
    Articulation,
    EntityId,
    EntityManager,
    RigidBody,
    Transform,
)
from fusion_simulation.physics.base import PhysicsEngine

logger = logging.getLogger(__name__)


@dataclass
class SceneAsset:
    asset_type: str = "urdf"
    path: str = ""
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    fixed_base: bool = False
    name: str = ""


@dataclass
class SceneConfig:
    name: str = "default"
    description: str = ""
    gravity: list[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    time_step: float = 0.01
    assets: list[SceneAsset] = field(default_factory=list)
    ground_plane: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class SceneResourceManager:
    BUILTIN_SCENES: dict[str, dict[str, Any]]

    def __init__(self, ecs: EntityManager, physics: PhysicsEngine) -> None:
        self._ecs = ecs
        self._physics = physics
        self.BUILTIN_SCENES = {
            "default": {
                "name": "default",
                "description": "Empty scene with ground plane",
                "gravity": [0.0, 0.0, -9.81],
                "time_step": 0.01,
                "ground_plane": True,
                "assets": [],
            },
            "pick": {
                "name": "pick",
                "description": "Object picking task",
                "gravity": [0.0, 0.0, -9.81],
                "time_step": 0.01,
                "ground_plane": True,
                "assets": [
                    {"asset_type": "urdf", "path": "table/table.urdf", "position": [0.5, 0.0, 0.0], "name": "table"},
                ],
            },
            "push": {
                "name": "push",
                "description": "Object pushing task",
                "gravity": [0.0, 0.0, -9.81],
                "time_step": 0.01,
                "ground_plane": True,
                "assets": [],
            },
        }
        self._loaded_bodies: dict[str, int] = {}
        self._loaded_entities: dict[str, EntityId] = {}
        logger.info("SceneResourceManager created")

    def load_scene(self, scene_config: SceneConfig) -> dict[str, Any]:
        results: dict[str, Any] = {
            "status": "loaded",
            "scene": scene_config.name,
            "assets": [],
            "errors": [],
        }
        if scene_config.ground_plane:
            try:
                plane_id = self._physics.load_plane()
                self._loaded_bodies["ground_plane"] = plane_id
                logger.info("Ground plane loaded: body_id=%d", plane_id)
            except Exception as e:
                results["errors"].append(f"ground_plane: {e}")
                logger.error("Failed to load ground plane: %s", e)
        for asset in scene_config.assets:
            try:
                body_id = self._load_asset(asset)
                results["assets"].append({"name": asset.name, "body_id": body_id})
            except Exception as e:
                results["errors"].append(f"{asset.name}: {e}")
                logger.error("Failed to load asset %s: %s", asset.name, e)
        logger.info(
            "Scene '%s' loaded: %d assets, %d errors", scene_config.name, len(results["assets"]), len(results["errors"])
        )
        return results

    def load_scene_from_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        config = self._dict_to_scene_config(data)
        return self.load_scene(config)

    def load_scene_from_file(self, path: str) -> dict[str, Any]:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Scene file not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        return self.load_scene_from_dict(data)

    def load_builtin(self, name: str) -> dict[str, Any]:
        if name not in self.BUILTIN_SCENES:
            raise ValueError(f"Unknown builtin scene: {name}. Available: {list(self.BUILTIN_SCENES.keys())}")
        return self.load_scene_from_dict(self.BUILTIN_SCENES[name])

    def unload_all(self) -> None:
        for name, body_id in self._loaded_bodies.items():
            try:
                self._physics.remove_body(body_id)
            except Exception as e:
                logger.warning("Failed to remove body %s (%d): %s", name, body_id, e)
        self._loaded_bodies.clear()
        for name, eid in self._loaded_entities.items():
            self._ecs.destroy_entity(eid)
        self._loaded_entities.clear()
        logger.info("Scene unloaded")

    def list_builtin_scenes(self) -> list[dict[str, str]]:
        return [{"name": k, "description": v.get("description", "")} for k, v in self.BUILTIN_SCENES.items()]

    def save_scene(self, scene_config: SceneConfig, path: str) -> None:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self._scene_config_to_dict(scene_config)
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Scene saved to %s", p)

    def _load_asset(self, asset: SceneAsset) -> int:
        if asset.asset_type == "urdf":
            body_id = self._physics.load_urdf(
                asset.path,
                position=asset.position,
                orientation=asset.orientation,
                fixed_base=asset.fixed_base,
            )
            eid = self._ecs.create_entity()
            transform = Transform(entity_id=eid, position=asset.position, orientation=asset.orientation)
            self._ecs.add_component(eid, transform)
            if asset.fixed_base:
                self._ecs.add_component(eid, Articulation(entity_id=eid, urdf_path=asset.path))
            else:
                self._ecs.add_component(eid, RigidBody(entity_id=eid))
            key = asset.name or f"asset_{body_id}"
            self._loaded_bodies[key] = body_id
            self._loaded_entities[key] = eid
            return body_id
        else:
            raise ValueError(f"Unsupported asset type: {asset.asset_type}")

    def _dict_to_scene_config(self, data: dict[str, Any]) -> SceneConfig:
        assets = []
        for a in data.get("assets", []):
            assets.append(
                SceneAsset(
                    asset_type=a.get("asset_type", "urdf"),
                    path=a.get("path", ""),
                    position=a.get("position", [0.0, 0.0, 0.0]),
                    orientation=a.get("orientation", [0.0, 0.0, 0.0, 1.0]),
                    fixed_base=a.get("fixed_base", False),
                    name=a.get("name", ""),
                )
            )
        return SceneConfig(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            gravity=data.get("gravity", [0.0, 0.0, -9.81]),
            time_step=data.get("time_step", 0.01),
            assets=assets,
            ground_plane=data.get("ground_plane", True),
            metadata=data.get("metadata", {}),
        )

    def _scene_config_to_dict(self, config: SceneConfig) -> dict[str, Any]:
        return {
            "name": config.name,
            "description": config.description,
            "gravity": config.gravity,
            "time_step": config.time_step,
            "ground_plane": config.ground_plane,
            "assets": [
                {
                    "asset_type": a.asset_type,
                    "path": a.path,
                    "position": a.position,
                    "orientation": a.orientation,
                    "fixed_base": a.fixed_base,
                    "name": a.name,
                }
                for a in config.assets
            ],
            "metadata": config.metadata,
        }
