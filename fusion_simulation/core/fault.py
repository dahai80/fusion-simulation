from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class FaultLevel(Enum):
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class FaultRecord:
    component_type: str
    component_id: str
    level: FaultLevel
    message: str
    recovered: bool = False


class FaultIsolationManager:
    def __init__(self) -> None:
        self._fault_counts: dict[str, int] = {}
        self._isolated: set[str] = set()
        self._fault_log: list[FaultRecord] = []
        self._max_faults_per_component: int = 5
        self._recovery_fns: dict[str, Any] = {}
        logger.info("FaultIsolationManager created")

    def set_max_faults(self, limit: int) -> None:
        self._max_faults_per_component = limit

    def register_recovery(self, component_type: str, component_id: str, fn: Any) -> None:
        key = f"{component_type}:{component_id}"
        self._recovery_fns[key] = fn
        logger.info("Recovery fn registered for %s", key)

    def report_fault(self, component_type: str, component_id: str, level: FaultLevel, message: str) -> bool:
        key = f"{component_type}:{component_id}"
        record = FaultRecord(
            component_type=component_type,
            component_id=component_id,
            level=level,
            message=message,
        )
        self._fault_log.append(record)
        self._fault_counts[key] = self._fault_counts.get(key, 0) + 1
        logger.warning("Fault reported: %s level=%s msg=%s", key, level.value, message)
        if level == FaultLevel.CRITICAL:
            self._isolate(component_type, component_id)
            return False
        if self._fault_counts[key] >= self._max_faults_per_component:
            self._isolate(component_type, component_id)
            return False
        if key in self._recovery_fns:
            try:
                self._recovery_fns[key]()
                record.recovered = True
                logger.info("Fault recovered: %s", key)
            except Exception as e:
                logger.error("Recovery failed for %s: %s", key, e)
        return True

    def _isolate(self, component_type: str, component_id: str) -> None:
        key = f"{component_type}:{component_id}"
        self._isolated.add(key)
        logger.error("Component ISOLATED: %s — too many faults or critical error", key)

    def is_isolated(self, component_type: str, component_id: str) -> bool:
        return f"{component_type}:{component_id}" in self._isolated

    def reset_component(self, component_type: str, component_id: str) -> bool:
        key = f"{component_type}:{component_id}"
        if key not in self._isolated:
            return False
        self._isolated.discard(key)
        self._fault_counts[key] = 0
        logger.info("Component reset (un-isolated): %s", key)
        return True

    def get_fault_log(self, limit: int = 100) -> list[dict[str, Any]]:
        records = self._fault_log[-limit:]
        return [
            {
                "component_type": r.component_type,
                "component_id": r.component_id,
                "level": r.level.value,
                "message": r.message,
                "recovered": r.recovered,
            }
            for r in records
        ]

    def get_status(self) -> dict[str, Any]:
        return {
            "isolated_components": list(self._isolated),
            "fault_counts": dict(self._fault_counts),
            "total_faults": len(self._fault_log),
        }

    def reset(self) -> None:
        self._fault_counts.clear()
        self._isolated.clear()
        self._fault_log.clear()
        logger.info("FaultIsolationManager reset")
