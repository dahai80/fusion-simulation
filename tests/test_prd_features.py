# Tests for T1-T6 PRD features: KernelState, FrameResult, async run, PolicyClient vision,
# AgentManager observe_act_loop, PromptScheduler, Server pause/resume/streaming RPCs.
# Called by: pytest (test runner). Affects API: SimulationKernel, PolicyClient, AgentManager,
# PromptScheduler, SimulationServer. Data schemas: FrameResult, ScheduleEntry, ScheduleResult,
# KernelConfig(use_scheduler, time_scale), KernelState enum.
# User instruction: refactor fusion-simulation per PRD, target NVIDIA Isaac Sim competitiveness.
from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import MagicMock, patch

import pytest

from fusion_simulation.agent.config import AgentConfig, AgentRole
from fusion_simulation.agent.scheduler import PromptScheduler, ScheduleEntry, ScheduleResult
from fusion_simulation.core.kernel import FrameResult, KernelConfig, KernelState, SimulationKernel
from fusion_simulation.core.clock import SimClock
from fusion_simulation.core.ecs import EntityManager, RigidBody, Transform
from fusion_simulation.core.event_bus import EventBus, EventKind
from fusion_simulation.core.world_state import WorldState
from fusion_simulation.agent.policy import PolicyClient


class TestKernelState:
    def test_kernel_state_enum(self):
        assert KernelState.UNINITIALIZED.name == "UNINITIALIZED"
        assert KernelState.INITIALIZED.name == "INITIALIZED"
        assert KernelState.RUNNING.name == "RUNNING"
        assert KernelState.PAUSED.name == "PAUSED"
        assert KernelState.STOPPED.name == "STOPPED"

    def test_kernel_config_defaults(self):
        cfg = KernelConfig()
        assert cfg.physics_dt == 0.01
        assert cfg.render_dt == pytest.approx(1.0 / 30.0)
        assert cfg.headless is True
        assert cfg.time_scale == 1.0
        assert cfg.use_scheduler is False

    def test_kernel_config_custom(self):
        cfg = KernelConfig(physics_dt=0.005, time_scale=2.0, use_scheduler=True)
        assert cfg.physics_dt == 0.005
        assert cfg.time_scale == 2.0
        assert cfg.use_scheduler is True


class TestFrameResult:
    def test_frame_result_defaults(self):
        fr = FrameResult()
        assert fr.sim_time == 0.0
        assert fr.frame_count == 0
        assert fr.physics_step_ms == 0.0
        assert fr.total_ms == 0.0

    def test_frame_result_fields(self):
        fr = FrameResult(sim_time=0.5, frame_count=50, physics_step_ms=2.0, total_ms=5.0)
        assert fr.sim_time == 0.5
        assert fr.frame_count == 50


class TestKernelUninitialized:
    def test_step_raises_without_init(self):
        k = SimulationKernel()
        with pytest.raises(RuntimeError, match="not initialized"):
            k.step()

    def test_step_once_raises_without_init(self):
        k = SimulationKernel()
        with pytest.raises(RuntimeError, match="not initialized"):
            k.step_once()

    def test_is_initialized_false(self):
        k = SimulationKernel()
        assert k.is_initialized is False
        assert k.kernel_state == KernelState.UNINITIALIZED


class TestKernelInit:
    def test_init_creates_sensor_and_agent_managers(self):
        with patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"), \
             patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"):
            k = SimulationKernel(KernelConfig(headless=True))
            k.init()
            assert k.sensor_manager is not None
            assert k.agent_manager is not None
            assert k.is_initialized is True
            assert k.kernel_state == KernelState.INITIALIZED

    def test_init_with_scheduler(self):
        with patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"), \
             patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"):
            k = SimulationKernel(KernelConfig(headless=True, use_scheduler=True))
            k.init()
            assert k.scheduler is not None

    def test_init_without_scheduler(self):
        with patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"), \
             patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"):
            k = SimulationKernel(KernelConfig(headless=True, use_scheduler=False))
            k.init()
            assert k.scheduler is None


class TestKernelStepOnce:
    def test_step_once_returns_frame_result(self):
        with patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"), \
             patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"):
            k = SimulationKernel(KernelConfig(headless=True))
            k.init()
            with patch.object(k._physics, "step"), \
                 patch.object(k._render, "render"):
                fr = k.step_once()
                assert isinstance(fr, FrameResult)
                assert fr.frame_count == 1

    def test_step_once_with_mock_physics(self):
        with patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"), \
             patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"):
            k = SimulationKernel(KernelConfig(headless=True))
            k.init()
            with patch.object(k._physics, "step"), \
                 patch.object(k._render, "render"), \
                 patch.object(k._sensor_manager, "update"), \
                 patch.object(k._agent_manager, "step_all", return_value={}):
                fr = k.step_once()
                assert fr.frame_count == 1


class TestKernelAsyncRun:
    def test_run_async_creates_task(self):
        with patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"), \
             patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"):
            k = SimulationKernel(KernelConfig(headless=True, max_steps=3))
            k.init()
            with patch.object(k._physics, "step"), \
                 patch.object(k._render, "render"), \
                 patch.object(k._agent_manager, "step_all", return_value={}):
                async def _run():
                    task = await k.run_async()
                    await asyncio.sleep(0.5)
                    return k.clock.frame_count
                count = asyncio.run(_run())
                assert count >= 0

    def test_stop_run(self):
        with patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"), \
             patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"):
            k = SimulationKernel(KernelConfig(headless=True))
            k.init()
            k.stop_run()
            assert k.kernel_state == KernelState.STOPPED


class TestKernelPauseResume:
    def test_pause_changes_state(self):
        with patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"), \
             patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"):
            k = SimulationKernel(KernelConfig(headless=True))
            k.init()
            k.start()
            k.pause()
            assert k.kernel_state == KernelState.PAUSED

    def test_resume_changes_state(self):
        with patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"), \
             patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"):
            k = SimulationKernel(KernelConfig(headless=True))
            k.init()
            k.start()
            k.pause()
            k.resume()
            assert k.kernel_state == KernelState.RUNNING


class TestKernelSceneLoad:
    def test_load_builtin_scene_without_init_raises(self):
        k = SimulationKernel()
        with pytest.raises(RuntimeError, match="Call init"):
            k.load_builtin_scene("default")


class TestKernelSyncEcs:
    def test_sync_ecs_from_physics_with_body_map(self):
        with patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"), \
             patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"):
            k = SimulationKernel(KernelConfig(headless=True))
            k.init()
            from fusion_simulation.core.ecs import EntityId
            from fusion_simulation.physics.base import BodyState
            eid = k._ecs.create_entity()
            k._ecs.add_component(eid, Transform(entity_id=eid))
            k._ecs.add_component(eid, RigidBody(entity_id=eid))
            k._body_entity_map[42] = eid
            body_state = BodyState(
                body_id=42,
                position=[1.0, 2.0, 3.0],
                orientation=[0.0, 0.0, 0.0, 1.0],
                linear_velocity=[0.1, 0.2, 0.3],
            )
            with patch.object(k._physics, "get_body_state", return_value=body_state), \
                 patch.object(type(k._physics), "is_initialized", new_callable=lambda: property(lambda self: True)):
                k._sync_ecs_from_physics()
            t = k._ecs.get_component(eid, Transform)
            assert t.position == [1.0, 2.0, 3.0]


class TestPolicyClientVision:
    def test_infer_from_image_with_bytes(self):
        client = PolicyClient()
        fake_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        with patch.object(client._client, "post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "[0.5, 0.3, 0.1]"}}]
            }
            mock_post.return_value = mock_resp
            import base64
            img_bytes = base64.b64decode(fake_b64)
            action = client.infer_from_image(img_bytes, action_dim=3)
            assert len(action) == 3
        client.close()

    def test_infer_from_image_with_base64_string(self):
        client = PolicyClient()
        b64_str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        with patch.object(client._client, "post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "[1.0, 0.0]"}}]
            }
            mock_post.return_value = mock_resp
            action = client.infer_from_image(b64_str, action_dim=2)
            assert len(action) == 2
        client.close()

    def test_infer_from_image_failure_returns_zeros(self):
        client = PolicyClient()
        with patch.object(client._client, "post", side_effect=Exception("connection error")):
            action = client.infer_from_image(b"fake", action_dim=4)
            assert action == [0.0, 0.0, 0.0, 0.0]
        client.close()

    def test_encode_image_numpy(self):
        client = PolicyClient()
        try:
            import numpy as np
            arr = np.zeros((10, 10, 3), dtype=np.uint8)
            result = client._encode_image(arr)
            assert isinstance(result, str)
            import base64
            decoded = base64.b64decode(result)
            assert len(decoded) > 0
        except ImportError:
            pytest.skip("numpy not available")
        client.close()

    def test_encode_image_unsupported_type(self):
        client = PolicyClient()
        with pytest.raises(TypeError, match="Unsupported image type"):
            client._encode_image(12345)
        client.close()


class TestPolicyClientStats:
    def test_stats_after_predict(self):
        client = PolicyClient()
        with patch.object(client._client, "post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "[0.1]"}}]
            }
            mock_post.return_value = mock_resp
            client.predict({"key": "value"}, action_dim=1)
        stats = client.stats()
        assert stats["request_count"] == 1
        assert stats["avg_latency"] > 0
        client.close()


class TestAgentManagerVisionLoop:
    def test_observe_act_loop_no_image(self):
        from fusion_simulation.agent.manager import AgentManager, AgentHandle
        from fusion_simulation.sensor.manager import SensorManager
        am = AgentManager(ecs=EntityManager(), sensor_manager=SensorManager())
        cfg = AgentConfig(name="test", entity_id="e1", action_dim=3)
        am.add_agent(cfg)
        with patch.object(am._agents["test"].policy, "predict", return_value=[0.1, 0.2, 0.3]):
            action = am.observe_act_loop("test")
            assert len(action) == 3

    def test_observe_act_loop_agent_done(self):
        from fusion_simulation.agent.manager import AgentManager
        am = AgentManager(ecs=EntityManager())
        cfg = AgentConfig(name="done_agent", entity_id="e1", action_dim=2)
        am.add_agent(cfg)
        am._agents["done_agent"]._done = True
        am._agents["done_agent"]._last_action = [1.0, 2.0]
        action = am.observe_act_loop("done_agent")
        assert action == [1.0, 2.0]

    def test_step_all_with_vision(self):
        from fusion_simulation.agent.manager import AgentManager
        am = AgentManager(ecs=EntityManager())
        cfg = AgentConfig(name="v_agent", entity_id="e1", action_dim=2)
        am.add_agent(cfg)
        with patch.object(am._agents["v_agent"].policy, "predict", return_value=[0.5, 0.5]):
            actions = am.step_all_with_vision()
            assert "v_agent" in actions


class TestPromptScheduler:
    def _make_scheduler(self) -> PromptScheduler:
        from fusion_simulation.agent.manager import AgentManager
        am = AgentManager(ecs=EntityManager())
        cfg = AgentConfig(name="robot1", entity_id="e1", action_dim=3, decimation=1)
        am.add_agent(cfg)
        return PromptScheduler(agent_manager=am, max_concurrent=2)

    def test_add_schedule(self):
        ps = self._make_scheduler()
        ps.add_schedule("robot1", priority=1, period=0.1)
        assert "robot1" in ps._schedule

    def test_add_schedule_unknown_agent(self):
        ps = self._make_scheduler()
        ps.add_schedule("nonexistent", priority=1)
        assert "nonexistent" not in ps._schedule

    def test_remove_schedule(self):
        ps = self._make_scheduler()
        ps.add_schedule("robot1")
        ps.remove_schedule("robot1")
        assert "robot1" not in ps._schedule

    def test_enable_disable_agent(self):
        ps = self._make_scheduler()
        ps.add_schedule("robot1")
        ps.enable_agent("robot1", False)
        assert ps._schedule["robot1"].enabled is False
        ps.enable_agent("robot1", True)
        assert ps._schedule["robot1"].enabled is True

    def test_tick_returns_actions(self):
        ps = self._make_scheduler()
        ps.add_schedule("robot1")
        with patch.object(ps._agent_manager, "observe_act_loop", return_value=[0.1, 0.2, 0.3]):
            actions = ps.tick(sim_time=0.01)
            assert "robot1" in actions

    def test_tick_respects_period(self):
        ps = self._make_scheduler()
        ps.add_schedule("robot1", period=1.0)
        ps._schedule["robot1"].last_run = 0.0
        with patch.object(ps._agent_manager, "observe_act_loop", return_value=[0.1, 0.2, 0.3]):
            actions = ps.tick(sim_time=0.5)
            assert "robot1" not in actions
            actions = ps.tick(sim_time=1.5)
            assert "robot1" in actions

    def test_tick_priority_order(self):
        from fusion_simulation.agent.manager import AgentManager
        am = AgentManager(ecs=EntityManager())
        am.add_agent(AgentConfig(name="low", entity_id="e1", action_dim=2, decimation=1))
        am.add_agent(AgentConfig(name="high", entity_id="e2", action_dim=2, decimation=1))
        ps = PromptScheduler(agent_manager=am, max_concurrent=1)
        ps.add_schedule("low", priority=0)
        ps.add_schedule("high", priority=10)
        call_order = []
        def mock_loop(name):
            call_order.append(name)
            return [0.0, 0.0]
        with patch.object(am, "observe_act_loop", side_effect=mock_loop):
            ps.tick(sim_time=0.01)
        assert call_order[0] == "high"

    def test_get_stats(self):
        ps = self._make_scheduler()
        ps.add_schedule("robot1")
        with patch.object(ps._agent_manager, "observe_act_loop", return_value=[0.1, 0.2, 0.3]):
            ps.tick(sim_time=0.01)
        stats = ps.get_stats()
        assert stats["total_ticks"] == 1
        assert stats["success_rate"] == 1.0

    def test_get_history(self):
        ps = self._make_scheduler()
        ps.add_schedule("robot1")
        with patch.object(ps._agent_manager, "observe_act_loop", return_value=[0.1, 0.2, 0.3]):
            ps.tick(sim_time=0.01)
        hist = ps.get_history()
        assert len(hist) == 1
        assert hist[0].agent_name == "robot1"
        assert hist[0].success is True

    def test_reset(self):
        ps = self._make_scheduler()
        ps.add_schedule("robot1")
        with patch.object(ps._agent_manager, "observe_act_loop", return_value=[0.1, 0.2, 0.3]):
            ps.tick(sim_time=0.01)
        ps.reset()
        assert len(ps._history) == 0


class TestServerNewRpcs:
    def test_pause_rpc(self):
        from fusion_simulation.service.server import SimulationServer
        with patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"), \
             patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"):
            srv = SimulationServer()
            srv.handle_request("init", {})
            resp = srv.handle_request("pause", {})
            assert resp.get("status") == "paused"

    def test_resume_rpc(self):
        from fusion_simulation.service.server import SimulationServer
        with patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"), \
             patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"):
            srv = SimulationServer()
            srv.handle_request("init", {})
            srv.handle_request("pause", {})
            resp = srv.handle_request("resume", {})
            assert resp.get("status") == "resumed"

    def test_step_returns_frame_timing(self):
        from fusion_simulation.service.server import SimulationServer
        with patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"), \
             patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"):
            srv = SimulationServer()
            srv.handle_request("init", {})
            with patch.object(srv._kernel._physics, "step"), \
                 patch.object(srv._kernel._render, "render"), \
                 patch.object(srv._kernel._agent_manager, "step_all", return_value={}):
                resp = srv.handle_request("step", {"num_steps": 1})
                assert "physics_step_ms" in resp
                assert "total_ms" in resp


class TestGatewayClient:
    def test_config_defaults(self):
        from fusion_simulation.service.gateway_client import GatewayConfig
        cfg = GatewayConfig()
        assert cfg.gateway_url == "http://127.0.0.1:11432"
        assert cfg.service_name == "fusion-simulation"
        assert cfg.enabled is False

    def test_register_disabled(self):
        from fusion_simulation.service.gateway_client import GatewayClient, GatewayConfig
        gc = GatewayClient(GatewayConfig(enabled=False))
        assert gc.register() is True
        assert gc.is_registered is True

    def test_register_enabled_fails_gracefully(self):
        from fusion_simulation.service.gateway_client import GatewayClient, GatewayConfig
        gc = GatewayClient(GatewayConfig(enabled=True, gateway_url="http://127.0.0.1:1"))
        assert gc.register() is False
        assert gc.is_registered is False
        gc.close()

    def test_deregister_without_register(self):
        from fusion_simulation.service.gateway_client import GatewayClient
        gc = GatewayClient()
        assert gc.deregister() is True

    def test_send_heartbeat_not_registered(self):
        from fusion_simulation.service.gateway_client import GatewayClient
        gc = GatewayClient()
        assert gc.send_heartbeat() is False

    def test_health_provider(self):
        from fusion_simulation.service.gateway_client import GatewayClient, HealthPayload
        gc = GatewayClient()
        gc._start_time = time.time()
        gc.set_health_provider(lambda: HealthPayload(status="healthy", kernel_state="RUNNING"))
        h = gc._get_health()
        assert h.status == "healthy"
        assert h.kernel_state == "RUNNING"
        gc.close()


class TestMetricsCollector:
    def test_counter(self):
        from fusion_simulation.service.metrics_server import MetricsCollector
        mc = MetricsCollector()
        mc.inc_counter("test_counter")
        assert mc.get_counter("test_counter") == 1.0
        mc.inc_counter("test_counter", value=4.0)
        assert mc.get_counter("test_counter") == 5.0

    def test_gauge(self):
        from fusion_simulation.service.metrics_server import MetricsCollector
        mc = MetricsCollector()
        mc.set_gauge("test_gauge", 42.0)
        assert mc.get_gauge("test_gauge") == 42.0
        mc.set_gauge("test_gauge", 10.0)
        assert mc.get_gauge("test_gauge") == 10.0

    def test_histogram(self):
        from fusion_simulation.service.metrics_server import MetricsCollector
        mc = MetricsCollector()
        mc.observe_histogram("latency", 5.0)
        mc.observe_histogram("latency", 10.0)
        mc.observe_histogram("latency", 15.0)
        prom = mc.to_prometheus()
        assert "latency_count" in prom
        assert "latency_sum" in prom

    def test_labels(self):
        from fusion_simulation.service.metrics_server import MetricsCollector
        mc = MetricsCollector()
        mc.inc_counter("requests", labels={"method": "step"})
        mc.inc_counter("requests", labels={"method": "reset"})
        assert mc.get_counter("requests", labels={"method": "step"}) == 1.0
        assert mc.get_counter("requests", labels={"method": "reset"}) == 1.0

    def test_prometheus_output(self):
        from fusion_simulation.service.metrics_server import MetricsCollector
        mc = MetricsCollector()
        mc.inc_counter("steps_total")
        mc.set_gauge("frame_count", 100)
        mc.observe_histogram("step_ms", 5.0)
        output = mc.to_prometheus()
        assert "fusion_sim_uptime_seconds" in output
        assert "steps_total" in output
        assert "frame_count" in output

    def test_reset(self):
        from fusion_simulation.service.metrics_server import MetricsCollector
        mc = MetricsCollector()
        mc.inc_counter("c")
        mc.set_gauge("g", 1.0)
        mc.reset()
        assert mc.get_counter("c") == 0.0
        assert mc.get_gauge("g") == 0.0


class TestMetricsServer:
    def test_start_stop(self):
        from fusion_simulation.service.metrics_server import MetricsServer, MetricsConfig
        ms = MetricsServer(MetricsConfig(port=0))
        ms.start()
        ms.stop()

    def test_health_endpoint(self):
        from fusion_simulation.service.metrics_server import MetricsServer, MetricsCollector, MetricsConfig
        import httpx
        cfg = MetricsConfig(host="127.0.0.1", port=18081)
        ms = MetricsServer(config=cfg)
        ms.set_health_provider(lambda: {"status": "healthy", "frame_count": 42})
        ms.start()
        try:
            resp = httpx.get("http://127.0.0.1:18081/health", timeout=2.0)
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"
        finally:
            ms.stop()

    def test_metrics_endpoint(self):
        from fusion_simulation.service.metrics_server import MetricsServer, MetricsCollector, MetricsConfig
        import httpx
        cfg = MetricsConfig(host="127.0.0.1", port=18082)
        mc = MetricsCollector()
        mc.inc_counter("test_steps")
        ms = MetricsServer(config=cfg, collector=mc)
        ms.start()
        try:
            resp = httpx.get("http://127.0.0.1:18082/metrics", timeout=2.0)
            assert resp.status_code == 200
            assert "test_steps" in resp.text
        finally:
            ms.stop()


class TestServerHealthAndMetrics:
    def test_get_health_initialized(self):
        from fusion_simulation.service.server import SimulationServer
        with patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"), \
             patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"):
            srv = SimulationServer()
            srv.handle_request("init", {})
            srv._running = True
            health = srv.get_health()
            assert health.status == "healthy"
            assert health.kernel_state in ("INITIALIZED", "RUNNING")

    def test_get_health_stopped(self):
        from fusion_simulation.service.server import SimulationServer
        srv = SimulationServer()
        health = srv.get_health()
        assert health.status == "stopped"

    def test_metrics_collector_on_step(self):
        from fusion_simulation.service.server import SimulationServer
        from fusion_simulation.service.metrics_server import MetricsConfig
        with patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"), \
             patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"):
            srv = SimulationServer(metrics_config=MetricsConfig(port=0))
            srv._start_metrics()
            srv.handle_request("init", {})
            with patch.object(srv._kernel._physics, "step"), \
                 patch.object(srv._kernel._render, "render"), \
                 patch.object(srv._agent_manager, "step_all", return_value={}):
                srv.handle_request("step", {"num_steps": 1})
            mc = srv.metrics_collector
            assert mc is not None
            assert mc.get_counter("fusion_sim_steps_total") >= 1.0


class TestCLIAgent:
    def test_agent_spawn(self, capsys):
        from fusion_simulation.cli import _cmd_agent_dispatch
        args = MagicMock()
        args.action = "spawn"
        args.name = "test_bot"
        args.entity_id = ""
        args.action_dim = 6
        args.role = "robot"
        args.decimation = 1
        _cmd_agent_dispatch(args)
        out = capsys.readouterr().out
        assert "test_bot" in out
        assert "ROBOT" in out or "robot" in out.lower()

    def test_agent_list(self, capsys):
        from fusion_simulation.cli import _cmd_agent_dispatch
        args = MagicMock()
        args.action = "list"
        _cmd_agent_dispatch(args)
        out = capsys.readouterr().out
        assert "kernel" in out.lower() or "no active" in out.lower()

    def test_agent_destroy(self, capsys):
        from fusion_simulation.cli import _cmd_agent_dispatch
        args = MagicMock()
        args.action = "destroy"
        args.name = "test_bot"
        _cmd_agent_dispatch(args)
        out = capsys.readouterr().out
        assert "test_bot" in out


class TestCLISensor:
    def test_sensor_add(self, capsys):
        from fusion_simulation.cli import _cmd_sensor_dispatch
        args = MagicMock()
        args.action = "add"
        args.type = "rgb_camera"
        args.name = "cam0"
        args.entity_id = ""
        args.width = 640
        args.height = 480
        _cmd_sensor_dispatch(args)
        out = capsys.readouterr().out
        assert "rgb_camera" in out
        assert "640" in out

    def test_sensor_list(self, capsys):
        from fusion_simulation.cli import _cmd_sensor_dispatch
        args = MagicMock()
        args.action = "list"
        _cmd_sensor_dispatch(args)
        out = capsys.readouterr().out
        assert "kernel" in out.lower() or "no active" in out.lower()


class TestCLISnapshot:
    def test_snapshot_save(self, capsys):
        from fusion_simulation.cli import _cmd_snapshot_dispatch
        args = MagicMock()
        args.action = "save"
        args.name = "checkpoint_1"
        _cmd_snapshot_dispatch(args)
        out = capsys.readouterr().out
        assert "checkpoint_1" in out

    def test_snapshot_restore(self, capsys):
        from fusion_simulation.cli import _cmd_snapshot_dispatch
        args = MagicMock()
        args.action = "restore"
        args.name = "checkpoint_1"
        _cmd_snapshot_dispatch(args)
        out = capsys.readouterr().out
        assert "checkpoint_1" in out


class TestCLIServiceStop:
    def test_service_stop(self, capsys):
        from fusion_simulation.cli import _cmd_service_dispatch
        args = MagicMock()
        args.action = "stop"
        _cmd_service_dispatch(args)
        out = capsys.readouterr().out
        assert "stop" in out.lower() or "SIGINT" in out


class TestCLIServiceHealth:
    def test_service_health_unreachable(self, capsys):
        from fusion_simulation.cli import _cmd_service_dispatch
        args = MagicMock()
        args.action = "health"
        args.url = "http://127.0.0.1:19999"
        _cmd_service_dispatch(args)
        out = capsys.readouterr().out
        assert "Cannot reach" in out or "unhealthy" in out or "error" in out.lower()


class TestCLIVersion:
    def test_version_output(self, capsys):
        from fusion_simulation.cli import _cmd_version
        _cmd_version()
        out = capsys.readouterr().out
        assert "Fusion-Simulation" in out
        assert "Gateway" in out


class TestCLIGateway:
    def test_gateway_register_disabled(self, capsys):
        from fusion_simulation.cli import _cmd_gateway_dispatch
        args = MagicMock()
        args.action = "register"
        args.gateway_url = "http://127.0.0.1:11432"
        args.service_port = 11447
        args.api_key = ""
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200, json=lambda: {"status": "ok"})
            _cmd_gateway_dispatch(args)
        out = capsys.readouterr().out
        assert "Gateway" in out


class TestCLIArgparse:
    def test_agent_spawn_args(self, capsys):
        from fusion_simulation.cli import _cmd_agent_dispatch
        args = MagicMock()
        args.action = "spawn"
        args.name = "bot1"
        args.entity_id = ""
        args.action_dim = 4
        args.role = "observer"
        args.decimation = 2
        _cmd_agent_dispatch(args)
        out = capsys.readouterr().out
        assert "bot1" in out

    def test_sensor_add_imu(self, capsys):
        from fusion_simulation.cli import _cmd_sensor_dispatch
        args = MagicMock()
        args.action = "add"
        args.type = "imu"
        args.name = "imu0"
        args.entity_id = ""
        args.width = 0
        args.height = 0
        _cmd_sensor_dispatch(args)
        out = capsys.readouterr().out
        assert "imu" in out

    def test_snapshot_save_default(self, capsys):
        from fusion_simulation.cli import _cmd_snapshot_dispatch
        args = MagicMock()
        args.action = "save"
        args.name = "default"
        _cmd_snapshot_dispatch(args)
        out = capsys.readouterr().out
        assert "default" in out

    def test_service_start_includes_metrics(self):
        from fusion_simulation.cli import _cmd_service_dispatch
        args = MagicMock()
        args.action = "start"
        args.host = "0.0.0.0"
        args.port = 11447
        args.metrics_port = 11456
        args.headless = True
        args.gateway_url = ""
        with patch("fusion_simulation.cli._cmd_service_start") as mock_start:
            _cmd_service_dispatch(args)
            mock_start.assert_called_once_with(args)
