from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    # Seed Python + NumPy global RNG so identical config produces identical
    # rollouts across nodes. PyBullet solver determinism relies on identical
    # scene load order + initial state (its randomSeed kwarg is build-dependent).
    if seed is None:
        logger.warning("seed_everything got None — skipping, determinism NOT guaranteed")
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import pybullet as p

        p.setPhysicsEngineParameter(randomSeed=int(seed))
    except Exception:
        logger.debug("pybullet randomSeed not set (pybullet unavailable or build rejects kwarg)")
    logger.info("Global RNG seeded: python.random + numpy seed=%d", int(seed))


def make_rng(seed: int | None) -> np.random.Generator:
    # Local Generator — does NOT touch global np.random, isolates per-episode noise.
    if seed is None:
        logger.debug("make_rng got None seed — returning non-deterministic Generator")
        return np.random.default_rng()
    return np.random.default_rng(int(seed))


def derive_seeds(base_seed: int, count: int, offset: int = 0) -> list[int]:
    # Deterministic child-seed derivation: spawn N independent episode seeds
    # from one base_seed. Same base + offset -> same child seeds across nodes.
    rng = np.random.default_rng(int(base_seed) + int(offset))
    high = 2**31 - 1
    seeds = rng.integers(0, high, size=count)
    return [int(s) for s in seeds]


def numpy_to_native(obj: Any) -> Any:
    # Recursively convert numpy scalars/arrays to JSON-serializable natives.
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: numpy_to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [numpy_to_native(v) for v in obj]
    return obj
