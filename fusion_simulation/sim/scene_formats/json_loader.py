from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fusion_simulation.sim.scene import SceneAsset, SceneConfig

logger = logging.getLogger(__name__)


class JsonSceneLoader:
    @staticmethod
    def load(path: str) -> SceneConfig:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Scene file not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        return JsonSceneLoader.from_dict(data)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> SceneConfig:
        assets = []
        for a in data.get("assets", []):
            assets.append(SceneAsset(
                asset_type=a.get("asset_type", "urdf"),
                path=a.get("path", ""),
                position=a.get("position", [0.0, 0.0, 0.0]),
                orientation=a.get("orientation", [0.0, 0.0, 0.0, 1.0]),
                fixed_base=a.get("fixed_base", False),
                name=a.get("name", ""),
            ))
        config = SceneConfig(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            gravity=data.get("gravity", [0.0, 0.0, -9.81]),
            time_step=data.get("time_step", 0.01),
            assets=assets,
            ground_plane=data.get("ground_plane", True),
            metadata=data.get("metadata", {}),
        )
        logger.info("JSON scene loaded: %s (%d assets)", config.name, len(config.assets))
        return config

    @staticmethod
    def save(config: SceneConfig, path: str) -> None:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
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
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Scene saved to %s", p)

    @staticmethod
    def validate(data: dict[str, Any]) -> list[str]:
        errors = []
        if "name" not in data:
            errors.append("Missing required field: name")
        for i, asset in enumerate(data.get("assets", [])):
            if not asset.get("path"):
                errors.append(f"Asset {i}: missing path")
        return errors
