from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from fusion_simulation.agent.manager import AgentManager
from fusion_simulation.core.kernel import KernelConfig, SimulationKernel
from fusion_simulation.sensor.manager import SensorManager

logger = logging.getLogger(__name__)


class EngineType(str, Enum):
    LEROBOT = "lerobot"
    XLEROBOT = "xlerobot"


@dataclass
class EnvConfig:
    engine: EngineType = EngineType.LEROBOT
    scene: str = "default"
    render_fps: int = 30
    headless: bool = False
    gravity: float = -9.81
    max_steps: int = 1000
    seed: int = 42


@dataclass
class SimulationState:
    step: int = 0
    robot_joint_positions: list[float] = field(default_factory=list)
    robot_joint_velocities: list[float] = field(default_factory=list)
    camera_image: bytes = b""
    task_completed: bool = False
    error: str = ""


class SimulationEnv:
    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self._kernel: SimulationKernel | None = None
        self._state = SimulationState()

    def init(self) -> dict[str, Any]:
        try:
            kernel_config = KernelConfig(
                physics_dt=1.0 / self.config.render_fps,
                render_dt=1.0 / self.config.render_fps,
                gravity=[0.0, 0.0, self.config.gravity],
                headless=self.config.headless,
                max_steps=self.config.max_steps,
                seed=self.config.seed,
            )
            sm = SensorManager()
            am = AgentManager()
            self._kernel = SimulationKernel(kernel_config)
            self._kernel.init(sensor_manager=sm, agent_manager=am)
            logger.info(
                "SimulationEnv initialized: engine=%s scene=%s fps=%d",
                self.config.engine.value,
                self.config.scene,
                self.config.render_fps,
            )
            return {"status": "initialized", "engine": self.config.engine.value}
        except Exception as e:
            logger.error("Failed to initialize simulation: %s", e)
            return {"status": "error", "error": str(e)}

    def step(self) -> SimulationState:
        if self._kernel is None:
            self._state.error = "Kernel not initialized"
            return self._state
        try:
            self._kernel.step(num_steps=1)
            self._state.step = self._kernel.clock.frame_count
            return self._state
        except Exception as e:
            self._state.error = str(e)
            return self._state

    def reset(self) -> None:
        if self._kernel is not None:
            self._kernel.reset()
        self._state = SimulationState()

    def close(self) -> None:
        if self._kernel is not None:
            self._kernel.close()

    def capture_camera(self) -> bytes:
        return b""

    @property
    def kernel(self) -> SimulationKernel | None:
        return self._kernel

    @staticmethod
    def list_scenes() -> list[dict[str, str]]:
        return [
            {"name": "pick", "description": "Object picking task", "engine": "lerobot"},
            {"name": "place", "description": "Object placing task", "engine": "lerobot"},
            {"name": "push", "description": "Object pushing task", "engine": "lerobot"},
            {"name": "bimanual_reach", "description": "Dual-arm reaching", "engine": "xlerobot"},
            {"name": "bimanual_lift", "description": "Dual-arm lifting", "engine": "xlerobot"},
        ]
