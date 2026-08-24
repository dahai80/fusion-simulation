from __future__ import annotations

import logging
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class EventKind(Enum):
    PHYSICS_PRE_STEP = auto()
    PHYSICS_POST_STEP = auto()
    RENDER_PRE_FRAME = auto()
    RENDER_POST_FRAME = auto()
    SIM_STARTED = auto()
    SIM_STOPPED = auto()
    SIM_PAUSED = auto()
    SIM_RESUMED = auto()
    SIM_RESET = auto()
    ENTITY_CREATED = auto()
    ENTITY_DESTROYED = auto()
    AGENT_SPAWNED = auto()
    AGENT_DESTROYED = auto()
    SENSOR_DATA_READY = auto()
    ACTION_APPLIED = auto()
    SCENE_LOADED = auto()
    SNAPSHOT_SAVED = auto()
    SNAPSHOT_RESTORED = auto()


@dataclass
class Event:
    kind: EventKind
    timestamp: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)


class EventBus:
    def __init__(self, max_log_size: int = 1000) -> None:
        self._handlers: dict[EventKind, list[Callable[[Event], None]]] = defaultdict(list)
        self._global_handlers: list[Callable[[Event], None]] = []
        self._event_log: deque[Event] = deque(maxlen=max_log_size)
        self._max_log_size: int = max_log_size
        self._raise_on_error: bool = False
        logger.info("EventBus created")

    def set_raise_on_error(self, enabled: bool) -> None:
        self._raise_on_error = enabled

    def subscribe(self, kind: EventKind, handler: Callable[[Event], None]) -> None:
        self._handlers[kind].append(handler)
        logger.debug("Handler subscribed to %s", kind.name)

    def subscribe_all(self, handler: Callable[[Event], None]) -> None:
        self._global_handlers.append(handler)
        logger.debug("Global handler subscribed")

    def unsubscribe(self, kind: EventKind, handler: Callable[[Event], None]) -> bool:
        handlers = self._handlers.get(kind, [])
        count = 0
        while handler in handlers:
            handlers.remove(handler)
            count += 1
        if count:
            logger.debug("Unsubscribed %d handler(s) from %s", count, kind.name)
        return count > 0

    def emit(self, kind: EventKind, data: dict[str, Any] | None = None, timestamp: float = 0.0) -> None:
        event = Event(kind=kind, timestamp=timestamp, data=data or {})
        self._event_log.append(event)
        if len(self._event_log) > self._max_log_size:
            self._event_log = deque(self._event_log, maxlen=self._max_log_size)
        for handler in list(self._handlers.get(kind, [])):
            try:
                handler(event)
            except Exception as e:
                logger.error("Event handler error for %s: %s", kind.name, e)
                if self._raise_on_error:
                    raise
        for handler in list(self._global_handlers):
            try:
                handler(event)
            except Exception as e:
                logger.error("Global event handler error for %s: %s", kind.name, e)
                if self._raise_on_error:
                    raise

    def get_log(self, kind: EventKind | None = None, limit: int = 100) -> list[Event]:
        if kind is not None:
            filtered = [e for e in self._event_log if e.kind == kind]
        else:
            filtered = list(self._event_log)
        return filtered[-limit:]

    def clear(self) -> None:
        self._event_log.clear()
        logger.debug("EventBus log cleared")

    def handler_count(self, kind: EventKind | None = None) -> int:
        if kind is None:
            return sum(len(h) for h in self._handlers.values()) + len(self._global_handlers)
        return len(self._handlers.get(kind, []))
