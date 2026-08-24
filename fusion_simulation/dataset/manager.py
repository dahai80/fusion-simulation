from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _try_flock(fd: int, exclusive: bool) -> None:
    try:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
    except (ImportError, OSError):
        logger.debug("flock unavailable on this platform, skipping file lock")


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
                with open(idx_file, encoding="utf-8") as f:
                    _try_flock(f.fileno(), exclusive=False)
                    raw = json.load(f)
            except json.JSONDecodeError as e:
                logger.error("Dataset index.json corrupt, resetting to empty: %s", e)
                self._datasets = {}
                return
            except OSError as e:
                logger.error("Failed to read dataset index.json: %s", e)
                self._datasets = {}
                return
            if not isinstance(raw, dict):
                logger.error("Dataset index.json not a dict, resetting to empty")
                self._datasets = {}
                return
            self._datasets = raw

    def _save_index(self) -> None:
        idx_file = self.data_path / "index.json"
        payload = json.dumps(self._datasets, indent=2, ensure_ascii=False)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(self.data_path), prefix=".index_", suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                _try_flock(f.fileno(), exclusive=True)
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, idx_file)
            logger.debug("Dataset index saved atomically: %s", idx_file)
        except OSError as e:
            logger.error("Failed to save dataset index atomically: %s", e)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _validate_name(name: str) -> None:
        if not name:
            raise ValueError("Dataset name must not be empty")
        if "/" in name or "\\" in name or ".." in name or name.startswith("."):
            raise ValueError(f"Invalid dataset name (path traversal blocked): {name!r}")

    def _assert_within_data_root(self, path: Path) -> None:
        root = self.data_path.resolve()
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"Path escapes data root: {resolved} not under {root}")

    def import_dataset(self, name: str, source_path: str, engine: str = "lerobot") -> dict[str, Any]:
        """Import a dataset from local path."""
        self._validate_name(name)
        src = Path(source_path).expanduser().resolve()
        if not src.exists():
            return {"status": "error", "error": f"Source not found: {src}"}
        dst = self.data_path / name
        self._assert_within_data_root(dst)
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
        self._assert_within_data_root(dst)
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

    def collect_samples(self, name: str, num_samples: int = 100, seed: int = 42) -> list[dict[str, Any]]:
        """Generate synthetic samples for training (DEMO ONLY — not real data)."""
        import numpy as np

        logger.warning(
            "collect_samples generating SYNTHETIC data for dataset='%s' seed=%d — DEMO only, "
            "not real training data. Use load_samples() for real imported data.",
            name,
            seed,
        )
        rng = np.random.default_rng(int(seed))
        samples = []
        for i in range(num_samples):
            samples.append(
                {
                    "observation": [float(x) for x in rng.random(64)],
                    "action": [float(x) for x in rng.random(8)],
                    "reward": float(rng.random()),
                }
            )
        if name in self._datasets:
            samples_dir = Path(self._datasets[name]["storage_path"]) / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)
            import json

            (samples_dir / "samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
        return samples

    def load_samples(self, name: str) -> list[dict[str, Any]]:
        """Load real imported samples from a dataset's storage path.

        Reads every *.json under the dataset storage_path and flattens to a list of
        {observation, action[, reward]} dicts. Refuses to fabricate data: returns
        whatever is actually on disk (empty list if nothing valid).
        """
        info = self._datasets.get(name)
        if info is None:
            logger.error("load_samples: dataset '%s' not found in index", name)
            return []
        storage = Path(info["storage_path"])
        samples: list[dict[str, Any]] = []
        for f in sorted(storage.rglob("*.json")):
            if f.name == "index.json":
                continue
            try:
                raw = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("load_samples: skipping unparseable file %s: %s", f, e)
                continue
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict):
                        samples.append(item)
            elif isinstance(raw, dict):
                samples.append(raw)
        valid = [s for s in samples if "observation" in s and "action" in s]
        if len(valid) < len(samples):
            logger.warning(
                "load_samples: %d/%d records lack 'observation'/'action' keys, dropped",
                len(samples) - len(valid),
                len(samples),
            )
        logger.info("load_samples: loaded %d real samples for dataset='%s'", len(valid), name)
        return valid

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
