from __future__ import annotations

import time
from unittest.mock import patch

from fusion_simulation.agent.config import AgentConfig, AgentRole
from fusion_simulation.agent.manager import AgentManager
from fusion_simulation.core.ecs import EntityManager, RigidBody, Transform
from fusion_simulation.core.event_bus import EventBus, EventKind
from fusion_simulation.core.kernel import KernelConfig, KernelState, SimulationKernel
from fusion_simulation.sensor.base import SensorConfig, SensorType
from fusion_simulation.sensor.manager import SensorManager


class TestEndToEndKernelLifecycle:
    def test_init_step_close_lifecycle(self):
        sm = SensorManager()
        am = AgentManager()
        kernel = SimulationKernel(KernelConfig(headless=True))
        assert kernel.kernel_state == KernelState.UNINITIALIZED
        kernel.init(sensor_manager=sm, agent_manager=am)
        assert kernel.kernel_state == KernelState.INITIALIZED
        result = kernel.step(num_steps=10)
        assert result.frame_count >= 10
        assert result.sim_time > 0
        kernel.close()
        # After close, state returns to UNINITIALIZED
        assert kernel.kernel_state == KernelState.UNINITIALIZED

    def test_load_scene_and_step(self):
        sm = SensorManager()
        am = AgentManager()
        kernel = SimulationKernel(KernelConfig(headless=True))
        kernel.init(sensor_manager=sm, agent_manager=am)
        result = kernel.load_builtin_scene("default")
        assert result is not None
        state = kernel.step(num_steps=5)
        assert state.frame_count >= 5
        kernel.close()

    def test_pause_resume_lifecycle(self):
        sm = SensorManager()
        am = AgentManager()
        kernel = SimulationKernel(KernelConfig(headless=True))
        kernel.init(sensor_manager=sm, agent_manager=am)
        kernel.start()
        assert kernel.kernel_state == KernelState.RUNNING
        kernel.pause()
        assert kernel.kernel_state == KernelState.PAUSED
        kernel.resume()
        assert kernel.kernel_state == KernelState.RUNNING
        kernel.stop_run()
        assert kernel.kernel_state == KernelState.STOPPED
        kernel.close()


class TestEndToEndSensorAgentPipeline:
    def test_sensor_add_observe(self):
        sm = SensorManager()
        sm.add_sensor(SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="cam0"))
        observations = sm.get_observations()
        assert "cam0" in observations

    def test_agent_observe_act_loop(self):
        am = AgentManager()
        cfg = AgentConfig(name="robot0", role=AgentRole.ROBOT, action_dim=6)
        am.add_agent(cfg)
        actions = am.step_all()
        assert "robot0" in actions

    def test_kernel_with_sensor_and_agent(self):
        sm = SensorManager()
        sm.add_sensor(SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="cam0"))
        am = AgentManager()
        am.add_agent(AgentConfig(name="bot", role=AgentRole.ROBOT, action_dim=6))
        kernel = SimulationKernel(KernelConfig(headless=True))
        kernel.init(sensor_manager=sm, agent_manager=am)
        result = kernel.step(num_steps=3)
        assert result.frame_count >= 3
        kernel.close()


class TestEndToEndEventBus:
    def test_step_events_fire(self):
        bus = EventBus()
        fired = []
        bus.subscribe(EventKind.PHYSICS_PRE_STEP, lambda e: fired.append("pre"))
        bus.subscribe(EventKind.PHYSICS_POST_STEP, lambda e: fired.append("post"))
        bus.emit(EventKind.PHYSICS_PRE_STEP, {})
        bus.emit(EventKind.PHYSICS_POST_STEP, {})
        assert "pre" in fired
        assert "post" in fired


class TestEndToEndECSSync:
    def test_ecs_create_and_transform(self):
        mgr = EntityManager()
        eid = mgr.create_entity()
        mgr.add_component(eid, Transform(position=[1.0, 2.0, 3.0]))
        mgr.add_component(eid, RigidBody(mass=5.0))
        t = mgr.get_component(eid, Transform)
        assert t is not None
        assert list(t.position) == [1.0, 2.0, 3.0]


class TestEndToEndServiceHealth:
    def test_server_health_via_metrics(self):
        from fusion_simulation.service.metrics_server import MetricsConfig
        from fusion_simulation.service.server import SimulationServer

        with (
            patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"),
            patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"),
        ):
            srv = SimulationServer(metrics_config=MetricsConfig(port=0))
            srv._start_metrics()
            srv._running = True
            health = srv.get_health()
            assert health.status in ("running", "stopped", "healthy")
            assert health.kernel_state is not None
            srv._stop_metrics()


class TestEndToEndGatewayRegistration:
    def test_gateway_register_deregister(self):
        from fusion_simulation.service.gateway_client import (
            GatewayClient,
            GatewayConfig,
        )

        cfg = GatewayConfig(enabled=False)
        client = GatewayClient(cfg)
        assert client.register()
        assert client._registered
        client.deregister()
        assert not client._registered
        client.close()


class TestPerformanceBenchmarks:
    def test_kernel_step_latency(self):
        sm = SensorManager()
        am = AgentManager()
        kernel = SimulationKernel(KernelConfig(headless=True))
        kernel.init(sensor_manager=sm, agent_manager=am)
        start = time.perf_counter()
        kernel.step(num_steps=100)
        elapsed = time.perf_counter() - start
        per_step_ms = (elapsed / 100) * 1000
        print(f"\n[PERF] Kernel step latency: {per_step_ms:.3f} ms/step (100 steps, {elapsed:.3f}s total)")
        assert per_step_ms < 100, f"Step latency too high: {per_step_ms:.3f} ms/step"
        kernel.close()

    def test_sensor_observation_throughput(self):
        sm = SensorManager()
        for i in range(5):
            sm.add_sensor(SensorConfig(sensor_type=SensorType.RGB_CAMERA, name=f"cam{i}"))
        start = time.perf_counter()
        for _ in range(1000):
            sm.get_observations()
        elapsed = time.perf_counter() - start
        per_obs_us = (elapsed / 1000) * 1_000_000
        print(f"\n[PERF] Sensor observation: {per_obs_us:.1f} µs/obs (1000 obs, {elapsed:.3f}s total)")
        assert per_obs_us < 1000, f"Observation throughput too low: {per_obs_us:.1f} µs/obs"

    def test_agent_decision_latency(self):
        am = AgentManager()
        am.add_agent(AgentConfig(name="bot", role=AgentRole.ROBOT, action_dim=6))
        start = time.perf_counter()
        for _ in range(100):
            am.step_all()
        elapsed = time.perf_counter() - start
        per_decision_ms = (elapsed / 100) * 1000
        print(f"\n[PERF] Agent decision: {per_decision_ms:.3f} ms/decision (100 decisions, {elapsed:.3f}s total)")
        assert per_decision_ms < 500, f"Decision latency too high: {per_decision_ms:.3f} ms"

    def test_ecs_entity_creation_throughput(self):
        mgr = EntityManager()
        start = time.perf_counter()
        for i in range(1000):
            eid = mgr.create_entity()
            mgr.add_component(eid, Transform(position=[float(i), 0.0, 0.0]))
        elapsed = time.perf_counter() - start
        per_entity_us = (elapsed / 1000) * 1_000_000
        print(f"\n[PERF] ECS entity creation: {per_entity_us:.1f} µs/entity (1000 entities, {elapsed:.3f}s total)")
        assert per_entity_us < 500, f"ECS creation throughput too low: {per_entity_us:.1f} µs/entity"

    def test_event_bus_throughput(self):
        bus = EventBus()
        counter = [0]
        bus.subscribe(EventKind.PHYSICS_POST_STEP, lambda e: counter.__setitem__(0, counter[0] + 1))
        start = time.perf_counter()
        for _ in range(10000):
            bus.emit(EventKind.PHYSICS_POST_STEP, {})
        elapsed = time.perf_counter() - start
        per_event_us = (elapsed / 10000) * 1_000_000
        print(f"\n[PERF] EventBus emit: {per_event_us:.2f} µs/event (10000 events, {elapsed:.3f}s total)")
        assert counter[0] == 10000
        assert per_event_us < 100, f"EventBus throughput too low: {per_event_us:.2f} µs/event"

    def test_metrics_collector_throughput(self):
        from fusion_simulation.service.metrics_server import MetricsCollector

        mc = MetricsCollector()
        start = time.perf_counter()
        for i in range(10000):
            mc.inc_counter("test_counter")
            mc.set_gauge("test_gauge", float(i))
        elapsed = time.perf_counter() - start
        per_op_us = (elapsed / 20000) * 1_000_000
        print(f"\n[PERF] MetricsCollector: {per_op_us:.2f} µs/op (20000 ops, {elapsed:.3f}s total)")
        assert per_op_us < 50, f"MetricsCollector throughput too low: {per_op_us:.2f} µs/op"
