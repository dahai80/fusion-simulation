from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimTime:
    sim_time: float = 0.0
    frame_count: int = 0
    physics_dt: float = 0.01
    render_dt: float = 1.0 / 30.0


class SimClock:
    def __init__(
        self,
        physics_dt: float = 0.01,
        render_dt: float = 1.0 / 30.0,
        max_step: int = 0,
    ):
        self._physics_dt = physics_dt
        self._render_dt = render_dt
        self._max_step = max_step
        self._sim_time: float = 0.0
        self._frame_count: int = 0
        self._wall_start: float = 0.0
        self._wall_elapsed: float = 0.0
        self._paused: bool = False
        logger.info(
            "SimClock created: physics_dt=%.4f render_dt=%.4f max_step=%d",
            physics_dt,
            render_dt,
            max_step,
        )

    @property
    def physics_dt(self) -> float:
        return self._physics_dt

    @property
    def render_dt(self) -> float:
        return self._render_dt

    @property
    def sim_time(self) -> float:
        return self._sim_time

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def wall_elapsed(self) -> float:
        return self._wall_elapsed

    @property
    def paused(self) -> bool:
        return self._paused

    def tick(self) -> SimTime:
        if self._paused:
            return self.snapshot()
        if self._max_step > 0 and self._frame_count >= self._max_step:
            logger.debug("SimClock hit max_step=%d", self._max_step)
            return self.snapshot()
        self._sim_time += self._physics_dt
        self._frame_count += 1
        self._wall_elapsed = time.monotonic() - self._wall_start if self._wall_start else 0.0
        return self.snapshot()

    def tick_render(self) -> SimTime:
        return self.snapshot()

    def snapshot(self) -> SimTime:
        return SimTime(
            sim_time=self._sim_time,
            frame_count=self._frame_count,
            physics_dt=self._physics_dt,
            render_dt=self._render_dt,
        )

    def reset(self) -> None:
        self._sim_time = 0.0
        self._frame_count = 0
        self._wall_start = time.monotonic()
        self._wall_elapsed = 0.0
        self._paused = False
        logger.info("SimClock reset")

    def start(self) -> None:
        self._wall_start = time.monotonic()
        self._paused = False
        logger.info("SimClock started")

    def pause(self) -> None:
        self._paused = True
        logger.info("SimClock paused at sim_time=%.4f", self._sim_time)

    def resume(self) -> None:
        self._paused = False
        logger.info("SimClock resumed at sim_time=%.4f", self._sim_time)

    def real_time_factor(self) -> float:
        if self._wall_elapsed <= 0 or self._sim_time <= 0:
            return 0.0
        return self._sim_time / self._wall_elapsed

    def should_render(self) -> bool:
        if self._render_dt <= 0:
            return True
        physics_per_render = max(1, round(self._render_dt / self._physics_dt))
        return self._frame_count % physics_per_render == 0
