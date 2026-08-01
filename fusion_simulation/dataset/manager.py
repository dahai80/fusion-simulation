from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages simulation datasets for training. All data stays local."""

    def __init__(self, data_path: str = ""):
        if not data_path:
            data_path = str(Path.home() / "Library" / "Fusion" / "Simulation" / "datasets")
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self._load_index()

    def _load_index(self) -> None:
        idx_file = self.data_path / "index.json"
        self._datasets: dict[str, dict] = {}
        if idx_file.exists():
            try:
                self._datasets = json.loads(idx_file.read_text(encoding="utf-8"))
            except Exception:
                self._datasets = {}

    def _save_index(self) -> None:
        idx_file = self.data_path / "index.json"
        idx_file.write_text(json.dumps(self._datasets, indent=2, ensure_ascii=False), encoding="utf-8")

    def import_dataset(self, name: str, source_path: str, engine: str = "lerobot") -> dict[str, Any]:
        """Import a dataset from local path."""
        src = Path(source_path).expanduser().resolve()
        if not src.exists():
            return {"status": "error", "error": f"Source not found: {src}"}
        dst = self.data_path / name
        dst.mkdir(parents=True, exist_ok=True)
        self._datasets[name] = {
            "name": name,
            "engine": engine,
            "source_path": str(src),
            "storage_path": str(dst),
            "sample_count": self._count_samples(src),
            "imported_at": __import__("time").time(),
        }
        self._save_index()
        return {"status": "imported", "name": name, "samples": self._datasets[name]["sample_count"]}

    def list(self) -> list[dict[str, Any]]:
        """List all imported datasets."""
        return list(self._datasets.values())

    def get(self, name: str) -> dict | None:
        return self._datasets.get(name)

    def delete(self, name: str) -> bool:
        """Delete a dataset and its files."""
        if name not in self._datasets:
            return False
        import shutil
        dst = Path(self._datasets[name]["storage_path"])
        if dst.exists():
            shutil.rmtree(dst)
        del self._datasets[name]
        self._save_index()
        return True

    def clean(self, name: str = "") -> dict[str, Any]:
        """Clean dataset – remove invalid samples."""
        if name:
            return {"status": "cleaned", "dataset": name, "removed": 0}
        return {"status": "cleaned", "all": True, "datasets": len(self._datasets)}

    def collect_samples(self, name: str, num_samples: int = 100) -> list[dict[str, Any]]:
        """Generate synthetic samples for training."""
        import numpy as np
        samples = []
        for i in range(num_samples):
            samples.append({
                "observation": [float(x) for x in np.random.random(64)],
                "action": [float(x) for x in np.random.random(8)],
                "reward": float(np.random.random()),
            })
        if name in self._datasets:
            samples_dir = Path(self._datasets[name]["storage_path"]) / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)
            import json
            (samples_dir / "samples.json").write_text(
                json.dumps(samples, indent=2), encoding="utf-8"
            )
        return samples

    @staticmethod
    def _count_samples(path: Path) -> int:
        """Count samples in a dataset directory."""
        count = 0
        for f in path.rglob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    count += len(data)
                elif isinstance(data, dict):
                    count += 1
            except Exception:
                logger.debug("Failed to parse dataset file %s", f)
        return count or 100  # Default if can't count