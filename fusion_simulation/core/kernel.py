from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from fusion_simulation.core.clock import SimClock, SimTime
from fusion_simulation.core.ecs import (
    Articulation,
    EntityId,
    EntityManager,
    RigidBody,
    Transform,
)
from fusion_simulation.core.event_bus import EventBus, EventKind
from fusion_simulation.core.world_state import EntitySnapshot, WorldState
from fusion_simulation.physics.base import PhysicsConfig, PhysicsEngine
from fusion_simulation.physics.pybullet_engine import PyBulletEngine
from fusion_simulation.render.base import RenderConfig, RenderEngine
from fusion_simulation.render.pybullet_render import PyBulletRender
from fusion_simulation.sim.scene import SceneConfig, SceneResourceManager

logger = logging.getLogger(__name__)


class KernelState(Enum):
    UNINITIALIZED = auto()
    INITIALIZED = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()


@dataclass
class KernelConfig:
    physics_dt: float = 0.01
    render_dt: float = 1.0 / 30.0
    gravity: list[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    headless: bool = True
    max_steps: int = 0
    seed: int = 42
    time_scale: float = 1.0
    agent_decimation: int = 1
    use_scheduler: bool = False


@dataclass
class FrameResult:
    sim_time: float = 0.0
    frame_count: int = 0
    physics_step_ms: float = 0.0
    sensor_collect_ms: float = 0.0
    agent_decide_ms: float = 0.0
    render_ms: float = 0.0
    total_ms: float = 0.0


class SimulationKernel:
    def __init__(self, config: KernelConfig | None = None) -> None:
        self._config = config or KernelConfig()
        self._clock = SimClock(
            physics_dt=self._config.physics_dt,
            render_dt=self._config.render_dt,
            max_step=self._config.max_steps,
        )
        self._ecs = EntityManager()
        self._events = EventBus()
        self._world = WorldState()
        self._physics: PhysicsEngine | None = None
        self._render: RenderEngine | None = None
        self._scene: SceneResourceManager | None = None
        self._sensor_manager: Any = None
        self._agent_manager: Any = None
        self._state = KernelState.UNINITIALIZED
        self._snapshots: dict[str, WorldState] = {}
        self._frame_result: FrameResult = FrameResult()
        self._body_entity_map: dict[int, EntityId] = {}
        self._run_task: asyncio.Task | None = None
        self._scheduler: Any = None
        logger.info(
            "SimulationKernel created: dt=%.4f headless=%s",
            self._config.physics_dt, self._config.headless,
        )

    @property
    def clock(self) -> SimClock:
        return self._clock

    @property
    def ecs(self) -> EntityManager:
        return self._ecs

    @property
    def events(self) -> EventBus:
        return self._events

    @property
    def world(self) -> WorldState:
        return self._world

    @property
    def physics(self) -> PhysicsEngine | None:
        return self._physics

    @property
    def render_engine(self) -> RenderEngine | None:
        return self._render

    @property
    def scene(self) -> SceneResourceManager | None:
        return self._scene

    @property
    def sensor_manager(self) -> Any:
        return self._sensor_manager

    @property
    def agent_manager(self) -> Any:
        return self._agent_manager

    @property
    def is_running(self) -> bool:
        return self._state == KernelState.RUNNING

    @property
    def is_initialized(self) -> bool:
        return self._state != KernelState.UNINITIALIZED

    @property
    def kernel_state(self) -> KernelState:
        return self._state

    @property
    def frame_result(self) -> FrameResult:
        return self._frame_result

    @property
    def scheduler(self) -> Any:
        return self._scheduler

    def init(
        self,
        physics: PhysicsEngine | None = None,
        render: RenderEngine | None = None,
        sensor_manager: Any = None,
        agent_manager: Any = None,
    ) -> None:
        if self._state != KernelState.UNINITIALIZED:
            logger.warning("SimulationKernel already initialized (state=%s)", self._state.name)
            return
        phys_config = PhysicsConfig(
            gravity=self._config.gravity,
            time_step=self._config.physics_dt,
            seed=self._config.seed,
        )
        self._physics = physics or PyBulletEngine()
        self._physics.init(config=phys_config, headless=self._config.headless)
        self._render = render or PyBulletRender(physics_engine=self._physics)
        self._render.init(RenderConfig())
        self._scene = SceneResourceManager(self._ecs, self._physics)
        if sensor_manager is not None:
            self._sensor_manager = sensor_manager
            self._sensor_manager.set_physics_engine(self._physics)
        else:
            from fusion_simulation.sensor.manager import SensorManager
            self._sensor_manager = SensorManager(physics_engine=self._physics)
        if agent_manager is not None:
            self._agent_manager = agent_manager
        else:
            from fusion_simulation.agent.manager import AgentManager
            self._agent_manager = AgentManager(
                ecs=self._ecs, sensor_manager=self._sensor_manager,
            )
        self._clock.start()
        if self._config.use_scheduler:
            from fusion_simulation.agent.scheduler import PromptScheduler
            self._scheduler = PromptScheduler(
                agent_manager=self._agent_manager,
                sensor_manager=self._sensor_manager,
            )
        self._state = KernelState.INITIALIZED
        self._events.emit(EventKind.SIM_STARTED, {"sim_time": 0.0})
        logger.info("SimulationKernel initialized")

    def step(self, num_steps: int = 1) -> SimTime:
        if self._state == KernelState.UNINITIALIZED:
            raise RuntimeError("SimulationKernel not initialized")
        for _ in range(num_steps):
            self._step_once()
        return self._clock.snapshot()

    def step_once(self) -> FrameResult:
        if self._state == KernelState.UNINITIALIZED:
            raise RuntimeError("SimulationKernel not initialized")
        return self._step_once()

    def _step_once(self) -> FrameResult:
        t0 = time.monotonic()
        self._events.emit(
            EventKind.PHYSICS_PRE_STEP,
            {"sim_time": self._clock.sim_time, "frame": self._clock.frame_count},
        )
        t_phys = time.monotonic()
        self._physics.step()
        physics_ms = (time.monotonic() - t_phys) * 1000.0

        sim_time_obj = self._clock.tick()

        self._sync_ecs_from_physics()
        self._sync_world_state()

        self._events.emit(
            EventKind.PHYSICS_POST_STEP,
            {"sim_time": sim_time_obj.sim_time, "frame": sim_time_obj.frame_count},
        )

        t_sensor = time.monotonic()
        if self._sensor_manager is not None:
            self._sensor_manager.set_sim_time(sim_time_obj.sim_time)
            self._sensor_manager.update()
        sensor_ms = (time.monotonic() - t_sensor) * 1000.0
        self._events.emit(EventKind.SENSOR_DATA_READY, {"sim_time": sim_time_obj.sim_time})

        t_agent = time.monotonic()
        if self._agent_manager is not None:
            if self._scheduler is not None:
                actions = self._scheduler.tick(sim_time_obj.sim_time)
            else:
                actions = self._agent_manager.step_all()
            self._apply_actions(actions)
            if actions:
                self._events.emit(EventKind.ACTION_APPLIED, {"actions": list(actions.keys())})
        agent_ms = (time.monotonic() - t_agent) * 1000.0

        t_render = time.monotonic()
        if self._clock.should_render():
            self._events.emit(EventKind.RENDER_PRE_FRAME, {"sim_time": sim_time_obj.sim_time})
            self._render.render()
            self._events.emit(EventKind.RENDER_POST_FRAME, {"sim_time": sim_time_obj.sim_time})
        render_ms = (time.monotonic() - t_render) * 1000.0

        total_ms = (time.monotonic() - t0) * 1000.0
        self._frame_result = FrameResult(
            sim_time=sim_time_obj.sim_time,
            frame_count=sim_time_obj.frame_count,
            physics_step_ms=physics_ms,
            sensor_collect_ms=sensor_ms,
            agent_decide_ms=agent_ms,
            render_ms=render_ms,
            total_ms=total_ms,
        )
        return self._frame_result

    async def run(self) -> None:
        if self._state == KernelState.UNINITIALIZED:
            raise RuntimeError("SimulationKernel not initialized. Call init() first.")
        self._state = KernelState.RUNNING
        logger.info("SimulationKernel async run started")
        try:
            while self._state == KernelState.RUNNING:
                if self._clock.paused:
                    await asyncio.sleep(0.01)
                    continue
                if self._config.max_steps > 0 and self._clock.frame_count >= self._config.max_steps:
                    logger.info("SimulationKernel reached max_steps=%d", self._config.max_steps)
                    break
                frame_start = time.monotonic()
                self._step_once()
                frame_elapsed = time.monotonic() - frame_start
                target_dt = self._config.physics_dt * self._config.time_scale
                sleep_time = target_dt - frame_elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            logger.info("SimulationKernel async run cancelled")
        except Exception:
            logger.exception("SimulationKernel async run error")
        finally:
            self._state = KernelState.STOPPED
            self._events.emit(EventKind.SIM_STOPPED)
            logger.info("SimulationKernel async run stopped at frame=%d", self._clock.frame_count)

    async def run_async(self) -> asyncio.Task:
        self._run_task = asyncio.create_task(self.run())
        return self._run_task

    def stop_run(self) -> None:
        if self._run_task is not None and not self._run_task.done():
            self._run_task.cancel()
            self._run_task = None
        self._state = KernelState.STOPPED
        self._clock.pause()

    def _sync_ecs_from_physics(self) -> None:
        if self._physics is None or not self._physics.is_initialized:
            return
        for body_id, eid in self._body_entity_map.items():
            transform = self._ecs.get_component(eid, Transform)
            if transform is None:
                continue
            body_state = self._physics.get_body_state(body_id)
            if body_state is None or body_state.body_id < 0:
                continue
            transform.position = list(body_state.position)
            transform.orientation = list(body_state.orientation)
            rigid_body = self._ecs.get_component(eid, RigidBody)
            if rigid_body is not None:
                rigid_body.linear_velocity = list(body_state.linear_velocity)
                rigid_body.angular_velocity = list(body_state.angular_velocity)
            articulation = self._ecs.get_component(eid, Articulation)
            if articulation is not None:
                if body_state.joint_positions:
                    articulation.joint_positions = list(body_state.joint_positions)
                if body_state.joint_velocities:
                    articulation.joint_velocities = list(body_state.joint_velocities)

    def reset(self) -> None:
        if self._state == KernelState.UNINITIALIZED:
            return
        self._physics.reset()
        self._ecs.clear()
        self._world = WorldState()
        self._clock.reset()
        self._body_entity_map.clear()
        if self._sensor_manager is not None:
            self._sensor_manager.reset()
        if self._agent_manager is not None:
            self._agent_manager.reset_all()
        self._state = KernelState.INITIALIZED
        self._events.emit(EventKind.SIM_RESET)
        logger.info("SimulationKernel reset")

    def close(self) -> None:
        if self._state == KernelState.UNINITIALIZED:
            return
        self._state = KernelState.STOPPED
        self._events.emit(EventKind.SIM_STOPPED)
        if self._agent_manager is not None:
            self._agent_manager.close()
        self._render.close()
        self._physics.close()
        self._state = KernelState.UNINITIALIZED
        self._body_entity_map.clear()
        logger.info("SimulationKernel closed")

    def start(self) -> None:
        if self._state == KernelState.UNINITIALIZED:
            raise RuntimeError("SimulationKernel not initialized")
        self._state = KernelState.RUNNING
        self._clock.start()
        logger.info("SimulationKernel started")

    def stop(self) -> None:
        self._state = KernelState.STOPPED
        self._clock.pause()
        logger.info("SimulationKernel stopped")

    def pause(self) -> None:
        self._clock.pause()
        if self._state == KernelState.RUNNING:
            self._state = KernelState.PAUSED
        self._events.emit(EventKind.SIM_PAUSED)
        logger.info("SimulationKernel paused")

    def resume(self) -> None:
        self._clock.resume()
        if self._state == KernelState.PAUSED:
            self._state = KernelState.RUNNING
        self._events.emit(EventKind.SIM_RESUMED)
        logger.info("SimulationKernel resumed")

    def load_scene(self, scene_config: SceneConfig) -> dict[str, Any]:
        if self._scene is None:
            raise RuntimeError("SceneResourceManager not available. Call init() first.")
        result = self._scene.load_scene(scene_config)
        self._sync_scene_entities()
        self._events.emit(EventKind.SCENE_LOADED, {"scene": scene_config.name})
        return result

    def load_builtin_scene(self, name: str) -> dict[str, Any]:
        if self._scene is None:
            raise RuntimeError("SceneResourceManager not available. Call init() first.")
        result = self._scene.load_builtin(name)
        self._sync_scene_entities()
        self._events.emit(EventKind.SCENE_LOADED, {"scene": name})
        return result

    def load_scene_from_file(self, path: str) -> dict[str, Any]:
        if self._scene is None:
            raise RuntimeError("SceneResourceManager not available. Call init() first.")
        result = self._scene.load_scene_from_file(path)
        self._sync_scene_entities()
        self._events.emit(EventKind.SCENE_LOADED, {"scene": path})
        return result

    def _sync_scene_entities(self) -> None:
        if self._scene is None:
            return
        loaded_entities = getattr(self._scene, '_loaded_entities', {})
        loaded_bodies = getattr(self._scene, '_loaded_bodies', {})
        for name, eid in loaded_entities.items():
            for body_name, body_id in loaded_bodies.items():
                if body_name == name:
                    self._body_entity_map[body_id] = eid
                    break
        logger.debug(
            "Synced scene entities: %d entities, %d body mappings",
            len(loaded_entities), len(self._body_entity_map),
        )

    def save_snapshot(self, name: str = "") -> str:
        snap = self._world.snapshot()
        snap_id = name or f"snapshot_{self._clock.frame_count}"
        self._snapshots[snap_id] = snap
        self._events.emit(EventKind.SNAPSHOT_SAVED, {"snapshot_id": snap_id})
        logger.info("Snapshot saved: %s", snap_id)
        return snap_id

    def restore_snapshot(self, snap_id: str) -> bool:
        if snap_id not in self._snapshots:
            logger.warning("Snapshot not found: %s", snap_id)
            return False
        self._world.restore(self._snapshots[snap_id])
        self._clock._sim_time = self._world.sim_time
        self._clock._frame_count = self._world.frame_count
        self._events.emit(EventKind.SNAPSHOT_RESTORED, {"snapshot_id": snap_id})
        logger.info("Snapshot restored: %s", snap_id)
        return True

    def get_world_state(self) -> WorldState:
        return self._world.snapshot()

    def _sync_world_state(self) -> None:
        self._world.sim_time = self._clock.sim_time
        self._world.frame_count = self._clock.frame_count
        for eid in self._ecs.list_entities():
            transform = self._ecs.get_component(eid, Transform)
            rigid_body = self._ecs.get_component(eid, RigidBody)
            articulation = self._ecs.get_component(eid, Articulation)
            if transform is None:
                continue
            eid_str = str(eid)
            if eid_str not in self._world.entities:
                self._world.entities[eid_str] = EntitySnapshot(entity_id=eid_str)
            snap = self._world.entities[eid_str]
            snap.position = list(transform.position)
            snap.orientation = list(transform.orientation)
            if rigid_body is not None:
                snap.linear_velocity = list(rigid_body.linear_velocity)
                snap.angular_velocity = list(rigid_body.angular_velocity)
            if articulation is not None:
                snap.joint_positions = list(articulation.joint_positions)
                snap.joint_velocities = list(articulation.joint_velocities)

    def status(self) -> dict[str, Any]:
        return {
            "initialized": self._state != KernelState.UNINITIALIZED,
            "running": self._state == KernelState.RUNNING,
            "state": self._state.name,
            "sim_time": self._clock.sim_time,
            "frame_count": self._clock.frame_count,
            "entity_count": self._ecs.entity_count(),
            "real_time_factor": self._clock.real_time_factor(),
            "paused": self._clock.paused,
        }

    def _apply_actions(self, actions: dict[str, list[float]]) -> None:
        if self._physics is None or self._agent_manager is None:
            return
        for agent_name, action in actions.items():
            if not action:
                continue
            agent = self._agent_manager.get_agent(agent_name)
            if agent is None:
                continue
            eid_str = agent.entity_id
            for body_id, mapped_eid in self._body_entity_map.items():
                if str(mapped_eid) == eid_str:
                    self._physics.apply_joint_action(
                        body_id,
                        joint_indices=list(range(len(action))),
                        values=action,
                        mode=agent.config.params.get("control_mode", "position"),
                    )
                    break
