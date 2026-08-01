from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from fusion_simulation.agent.manager import AgentManager
from fusion_simulation.sensor.manager import SensorManager

logger = logging.getLogger(__name__)


@dataclass
class ScheduleEntry:
    agent_name: str
    priority: int = 0
    period: float = 0.0
    last_run: float = 0.0
    enabled: bool = True


@dataclass
class ScheduleResult:
    agent_name: str
    action: list[float]
    latency_ms: float
    success: bool
    used_vision: bool


class PromptScheduler:
    def __init__(
        self,
        agent_manager: AgentManager,
        sensor_manager: SensorManager | None = None,
        max_concurrent: int = 4,
    ) -> None:
        self._agent_manager = agent_manager
        self._sensor_manager = sensor_manager
        self._max_concurrent = max_concurrent
        self._schedule: dict[str, ScheduleEntry] = {}
        self._history: list[ScheduleResult] = []
        self._max_history: int = 1000

    def add_schedule(self, agent_name: str, priority: int = 0, period: float = 0.0) -> None:
        if agent_name not in self._agent_manager.list_agents():
            logger.warning("Cannot schedule unknown agent: %s", agent_name)
            return
        self._schedule[agent_name] = ScheduleEntry(
            agent_name=agent_name,
            priority=priority,
            period=period,
        )
        logger.info("Scheduled agent %s: priority=%d period=%.3fs", agent_name, priority, period)

    def remove_schedule(self, agent_name: str) -> None:
        self._schedule.pop(agent_name, None)
        logger.info("Unscheduled agent: %s", agent_name)

    def enable_agent(self, agent_name: str, enabled: bool = True) -> None:
        entry = self._schedule.get(agent_name)
        if entry is not None:
            entry.enabled = enabled

    def tick(self, sim_time: float = 0.0) -> dict[str, list[float]]:
        actions: dict[str, list[float]] = {}
        ready = self._get_ready_agents(sim_time)
        ready.sort(key=lambda e: e.priority, reverse=True)
        for entry in ready[:self._max_concurrent]:
            t0 = time.monotonic()
            try:
                action = self._agent_manager.observe_act_loop(entry.agent_name)
                latency_ms = (time.monotonic() - t0) * 1000.0
                used_vision = self._has_vision(entry.agent_name)
                actions[entry.agent_name] = action
                self._record(ScheduleResult(
                    agent_name=entry.agent_name,
                    action=action,
                    latency_ms=latency_ms,
                    success=True,
                    used_vision=used_vision,
                ))
                entry.last_run = sim_time
            except Exception:
                latency_ms = (time.monotonic() - t0) * 1000.0
                self._record(ScheduleResult(
                    agent_name=entry.agent_name,
                    action=[],
                    latency_ms=latency_ms,
                    success=False,
                    used_vision=False,
                ))
                logger.exception("PromptScheduler tick failed for agent %s", entry.agent_name)
        return actions

    def _get_ready_agents(self, sim_time: float) -> list[ScheduleEntry]:
        ready = []
        for entry in self._schedule.values():
            if not entry.enabled:
                continue
            agent = self._agent_manager.get_agent(entry.agent_name)
            if agent is None or agent.done:
                continue
            if entry.period > 0 and (sim_time - entry.last_run) < entry.period:
                continue
            ready.append(entry)
        return ready

    def _has_vision(self, agent_name: str) -> bool:
        if self._sensor_manager is None:
            return False
        agent = self._agent_manager.get_agent(agent_name)
        if agent is None:
            return False
        for sname in self._sensor_manager.list_sensors():
            sensor = self._sensor_manager.get_sensor(sname)
            if sensor is None:
                continue
            if sensor.entity_id and sensor.entity_id == agent.entity_id and hasattr(sensor, "rgb") and sensor.rgb is not None:
                return True
        return False

    def _record(self, result: ScheduleResult) -> None:
        self._history.append(result)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def get_history(self, agent_name: str = "", last_n: int = 50) -> list[ScheduleResult]:
        hist = self._history
        if agent_name:
            hist = [h for h in hist if h.agent_name == agent_name]
        return hist[-last_n:]

    def get_stats(self) -> dict[str, Any]:
        if not self._history:
            return {"total_scheduled": len(self._schedule), "total_ticks": 0}
        success_count = sum(1 for h in self._history if h.success)
        avg_latency = sum(h.latency_ms for h in self._history) / len(self._history)
        vision_count = sum(1 for h in self._history if h.used_vision)
        return {
            "total_scheduled": len(self._schedule),
            "total_ticks": len(self._history),
            "success_rate": success_count / len(self._history),
            "avg_latency_ms": avg_latency,
            "vision_inference_count": vision_count,
        }

    def reset(self) -> None:
        for entry in self._schedule.values():
            entry.last_run = 0.0
        self._history.clear()
        logger.info("PromptScheduler reset")
