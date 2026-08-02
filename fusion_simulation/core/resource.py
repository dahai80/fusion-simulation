from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ResourceQuota:
    max_agents: int = 16
    max_sensors: int = 32
    max_bodies: int = 256
    max_recording_frames: int = 10000
    max_memory_mb: float = 4096.0
    max_cpu_percent: float = 80.0
    max_step_time_ms: float = 50.0


@dataclass
class ResourceUsage:
    num_agents: int = 0
    num_sensors: int = 0
    num_bodies: int = 0
    recording_frames: int = 0
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    last_step_time_ms: float = 0.0


class ResourceQuotaManager:
    def __init__(self, quota: ResourceQuota | None = None) -> None:
        self._quota = quota or ResourceQuota()
        self._usage = ResourceUsage()
        self._step_start: float = 0.0
        logger.info(
            "ResourceQuotaManager created: agents=%d sensors=%d bodies=%d",
            self._quota.max_agents,
            self._quota.max_sensors,
            self._quota.max_bodies,
        )

    @property
    def quota(self) -> ResourceQuota:
        return self._quota

    @property
    def usage(self) -> ResourceUsage:
        return self._usage

    def set_quota(self, quota: ResourceQuota) -> None:
        self._quota = quota
        logger.info("Resource quota updated")

    def check_agent_limit(self, current: int) -> bool:
        self._usage.num_agents = current
        ok = current < self._quota.max_agents
        if not ok:
            logger.warning("Agent limit reached: %d/%d", current, self._quota.max_agents)
        return ok

    def check_sensor_limit(self, current: int) -> bool:
        self._usage.num_sensors = current
        ok = current < self._quota.max_sensors
        if not ok:
            logger.warning("Sensor limit reached: %d/%d", current, self._quota.max_sensors)
        return ok

    def check_body_limit(self, current: int) -> bool:
        self._usage.num_bodies = current
        ok = current < self._quota.max_bodies
        if not ok:
            logger.warning("Body limit reached: %d/%d", current, self._quota.max_bodies)
        return ok

    def check_recording_frames(self, current: int) -> bool:
        self._usage.recording_frames = current
        ok = current < self._quota.max_recording_frames
        if not ok:
            logger.warning("Recording frame limit reached: %d/%d", current, self._quota.max_recording_frames)
        return ok

    def step_begin(self) -> None:
        self._step_start = time.monotonic()

    def step_end(self) -> bool:
        if self._step_start <= 0:
            return True
        elapsed_ms = (time.monotonic() - self._step_start) * 1000.0
        self._usage.last_step_time_ms = elapsed_ms
        self._step_start = 0.0
        ok = elapsed_ms <= self._quota.max_step_time_ms
        if not ok:
            logger.warning("Step time exceeded quota: %.1fms > %.1fms", elapsed_ms, self._quota.max_step_time_ms)
        return ok

    def get_status(self) -> dict[str, Any]:
        return {
            "quota": {
                "max_agents": self._quota.max_agents,
                "max_sensors": self._quota.max_sensors,
                "max_bodies": self._quota.max_bodies,
                "max_recording_frames": self._quota.max_recording_frames,
                "max_memory_mb": self._quota.max_memory_mb,
                "max_step_time_ms": self._quota.max_step_time_ms,
            },
            "usage": {
                "num_agents": self._usage.num_agents,
                "num_sensors": self._usage.num_sensors,
                "num_bodies": self._usage.num_bodies,
                "recording_frames": self._usage.recording_frames,
                "last_step_time_ms": self._usage.last_step_time_ms,
            },
        }

    def reset(self) -> None:
        self._usage = ResourceUsage()
        self._step_start = 0.0
        logger.info("ResourceQuotaManager reset")
