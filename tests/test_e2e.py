from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np

from fusion_simulation.agent.config import AgentConfig, AgentRole
from fusion_simulation.agent.manager import AgentManager
from fusion_simulation.core.ecs import EntityManager, RigidBody, Transform
from fusion_simulation.core.event_bus import EventBus, EventKind
from fusion_simulation.core.kernel import KernelConfig, KernelState, SimulationKernel
from fusion_simulation.sensor.base import SensorConfig, SensorType
from fusion_simulation.sensor.manager import SensorManager
from fusion_simulation.sim.env import EnvConfig, SimulationEnv
from fusion_simulation.train.gym_env import (
    ActionManager,
    FusionGymEnv,
    ObservationManager,
    RewardManager,
    TerminationManager,
)

logger_patch = __import__("logging").getLogger(__name__)


# ── Kernel Lifecycle E2E ──────────────────────────────────────────────


class TestKernelLifecycleE2E:
    def test_full_lifecycle_init_step_pause_resume_stop_close(self):
        sm = SensorManager()
        am = AgentManager()
        kernel = SimulationKernel(KernelConfig(headless=True))
        assert kernel.kernel_state == KernelState.UNINITIALIZED

        kernel.init(sensor_manager=sm, agent_manager=am)
        assert kernel.kernel_state == KernelState.INITIALIZED

        result = kernel.step(num_steps=5)
        assert result.frame_count >= 5
        assert result.sim_time > 0

        kernel.start()
        assert kernel.kernel_state == KernelState.RUNNING
        result2 = kernel.step(num_steps=3)
        assert result2.frame_count >= 8

        kernel.pause()
        assert kernel.kernel_state == KernelState.PAUSED

        kernel.resume()
        assert kernel.kernel_state == KernelState.RUNNING

        kernel.stop_run()
        assert kernel.kernel_state == KernelState.STOPPED

        kernel.close()
        assert kernel.kernel_state == KernelState.UNINITIALIZED

    def test_init_close_no_step(self):
        sm = SensorManager()
        am = AgentManager()
        kernel = SimulationKernel(KernelConfig(headless=True))
        kernel.init(sensor_manager=sm, agent_manager=am)
        assert kernel.kernel_state == KernelState.INITIALIZED
        kernel.close()
        assert kernel.kernel_state == KernelState.UNINITIALIZED

    def test_repeated_init_close_cycles(self):
        for i in range(5):
            sm = SensorManager()
            am = AgentManager()
            kernel = SimulationKernel(KernelConfig(headless=True))
            kernel.init(sensor_manager=sm, agent_manager=am)
            assert kernel.kernel_state == KernelState.INITIALIZED
            kernel.step(num_steps=2)
            kernel.close()
            assert kernel.kernel_state == KernelState.UNINITIALIZED
            logger_patch.info("Cycle %d completed", i)

    def test_step_after_reset(self):
        sm = SensorManager()
        am = AgentManager()
        kernel = SimulationKernel(KernelConfig(headless=True))
        kernel.init(sensor_manager=sm, agent_manager=am)
        kernel.step(num_steps=10)
        kernel.reset()
        result = kernel.step(num_steps=5)
        assert result.frame_count >= 5
        kernel.close()

    def test_snapshot_save_restore(self):
        sm = SensorManager()
        am = AgentManager()
        kernel = SimulationKernel(KernelConfig(headless=True))
        kernel.init(sensor_manager=sm, agent_manager=am)
        kernel.step(num_steps=10)
        snap_id = kernel.save_snapshot("test_snap")
        assert snap_id != ""
        result1 = kernel.step(num_steps=5)
        fc_before = result1.frame_count
        restored = kernel.restore_snapshot(snap_id)
        assert restored is True
        kernel.close()

    def test_load_builtin_scene(self):
        sm = SensorManager()
        am = AgentManager()
        kernel = SimulationKernel(KernelConfig(headless=True))
        kernel.init(sensor_manager=sm, agent_manager=am)
        result = kernel.load_builtin_scene("default")
        assert result is not None
        kernel.close()

    def test_kernel_status_returns_valid_dict(self):
        sm = SensorManager()
        am = AgentManager()
        kernel = SimulationKernel(KernelConfig(headless=True))
        kernel.init(sensor_manager=sm, agent_manager=am)
        status = kernel.status()
        assert "state" in status
        assert "frame_count" in status
        kernel.close()

    def test_step_with_sensor_and_agent(self):
        sm = SensorManager()
        sm.add_sensor(SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="cam0"))
        am = AgentManager()
        am.add_agent(AgentConfig(name="bot", role=AgentRole.ROBOT, action_dim=6))
        kernel = SimulationKernel(KernelConfig(headless=True))
        kernel.init(sensor_manager=sm, agent_manager=am)
        result = kernel.step(num_steps=10)
        assert result.frame_count >= 10
        fr = kernel.frame_result
        assert fr.sensor_collect_ms >= 0
        assert fr.agent_decide_ms >= 0
        kernel.close()


# ── Service Layer E2E ─────────────────────────────────────────────────


class TestSimulationServerRPCE2E:
    def test_rpc_init_step_status_close(self):
        from fusion_simulation.service.server import SimulationServer

        with (
            patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"),
            patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"),
        ):
            srv = SimulationServer()
            resp = srv.handle_request("init", {})
            assert resp.get("status") == "initialized"

            resp = srv.handle_request("step", {"num_steps": 3})
            assert resp.get("frame_count", 0) >= 3
            assert resp.get("sim_time", 0.0) >= 0.0

            resp = srv.handle_request("status", {})
            assert "state" in resp or "initialized" in resp

            resp = srv.handle_request("close", {})
            assert resp.get("status") == "closed"

    def test_rpc_add_sensor_and_get_observations(self):
        from fusion_simulation.service.server import SimulationServer

        with (
            patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"),
            patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"),
        ):
            srv = SimulationServer()
            srv.handle_request("init", {})
            resp = srv.handle_request(
                "add_sensor",
                {
                    "type": "rgb_camera",
                    "name": "cam0",
                },
            )
            assert resp.get("status") == "added"

            resp = srv.handle_request("get_observations", {})
            assert "cam0" in resp

            srv.handle_request("close", {})

    def test_rpc_add_agent(self):
        from fusion_simulation.service.server import SimulationServer

        with (
            patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"),
            patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"),
        ):
            srv = SimulationServer()
            srv.handle_request("init", {})
            resp = srv.handle_request(
                "add_agent",
                {
                    "name": "robot0",
                    "role": "robot",
                    "action_dim": 6,
                },
            )
            assert resp.get("status") == "added"

            resp = srv.handle_request("step", {"num_steps": 1})
            assert resp.get("frame_count", 0) >= 1

            srv.handle_request("close", {})

    def test_rpc_load_scene(self):
        from fusion_simulation.service.server import SimulationServer

        with (
            patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"),
            patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"),
        ):
            srv = SimulationServer()
            srv.handle_request("init", {})
            resp = srv.handle_request("load_scene", {"name": "default"})
            assert resp.get("status") == "loaded"
            srv.handle_request("close", {})

    def test_rpc_snapshot_save_restore(self):
        from fusion_simulation.service.server import SimulationServer

        with (
            patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"),
            patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"),
        ):
            srv = SimulationServer()
            srv.handle_request("init", {})
            srv.handle_request("step", {"num_steps": 5})
            resp = srv.handle_request("save_snapshot", {"name": "snap1"})
            assert resp.get("snapshot_id") != ""

            resp = srv.handle_request(
                "restore_snapshot",
                {
                    "snapshot_id": resp.get("snapshot_id"),
                },
            )
            assert resp.get("restored") is True
            srv.handle_request("close", {})

    def test_rpc_pause_resume(self):
        from fusion_simulation.service.server import SimulationServer

        with (
            patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"),
            patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"),
        ):
            srv = SimulationServer()
            srv.handle_request("init", {})
            srv._ensure_kernel().start()

            resp = srv.handle_request("pause", {})
            assert resp.get("status") == "paused"
            assert srv.kernel.kernel_state == KernelState.PAUSED

            resp = srv.handle_request("resume", {})
            assert resp.get("status") == "resumed"
            assert srv.kernel.kernel_state == KernelState.RUNNING

            srv._ensure_kernel().stop_run()
            srv.handle_request("close", {})

    def test_rpc_unknown_method(self):
        from fusion_simulation.service.server import SimulationServer

        with (
            patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"),
            patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"),
        ):
            srv = SimulationServer()
            resp = srv.handle_request("nonexistent", {})
            assert "error" in resp

    def test_rpc_reset(self):
        from fusion_simulation.service.server import SimulationServer

        with (
            patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"),
            patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"),
        ):
            srv = SimulationServer()
            srv.handle_request("init", {})
            srv.handle_request("step", {"num_steps": 10})
            resp = srv.handle_request("reset", {})
            assert resp.get("status") == "reset"
            srv.handle_request("close", {})

    def test_server_health_without_kernel(self):
        from fusion_simulation.service.server import SimulationServer

        srv = SimulationServer()
        health = srv.get_health()
        assert health.status == "stopped"
        assert health.kernel_state == ""

    def test_server_health_with_kernel(self):
        from fusion_simulation.service.server import SimulationServer

        with (
            patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"),
            patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"),
        ):
            srv = SimulationServer()
            srv._ensure_kernel()
            srv._running = True
            health = srv.get_health()
            assert health.status == "healthy"
            assert health.kernel_state != ""


class TestMetricsServerHTTPE2E:
    def test_metrics_server_start_stop(self):
        from fusion_simulation.service.metrics_server import (
            MetricsCollector,
            MetricsConfig,
            MetricsServer,
        )

        collector = MetricsCollector()
        server = MetricsServer(
            config=MetricsConfig(port=0),
            collector=collector,
        )
        server.start()
        assert server._server is not None
        server.stop()
        assert server._server is None

    def test_metrics_collector_counters_and_gauges(self):
        from fusion_simulation.service.metrics_server import MetricsCollector

        mc = MetricsCollector()
        mc.inc_counter("steps", value=10)
        mc.inc_counter("steps", value=5)
        assert mc.get_counter("steps") == 15

        mc.set_gauge("frame_count", 42)
        assert mc.get_gauge("frame_count") == 42

    def test_metrics_collector_prometheus_output(self):
        from fusion_simulation.service.metrics_server import MetricsCollector

        mc = MetricsCollector()
        mc.inc_counter("fusion_sim_steps_total", value=100)
        mc.set_gauge("fusion_sim_frame_count", 500)
        output = mc.to_prometheus()
        assert "fusion_sim_steps_total" in output
        assert "fusion_sim_frame_count" in output
        assert "counter" in output
        assert "gauge" in output

    def test_metrics_collector_with_labels(self):
        from fusion_simulation.service.metrics_server import MetricsCollector

        mc = MetricsCollector()
        mc.inc_counter("requests", labels={"method": "step"})
        mc.inc_counter("requests", labels={"method": "init"})
        output = mc.to_prometheus()
        assert 'method="step"' in output
        assert 'method="init"' in output

    def test_metrics_collector_histogram(self):
        from fusion_simulation.service.metrics_server import MetricsCollector

        mc = MetricsCollector()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            mc.observe_histogram("latency_ms", v)
        output = mc.to_prometheus()
        assert "latency_ms_count" in output
        assert "latency_ms_sum" in output
        assert "quantile" in output

    def test_metrics_collector_reset(self):
        from fusion_simulation.service.metrics_server import MetricsCollector

        mc = MetricsCollector()
        mc.inc_counter("steps", value=10)
        mc.set_gauge("frames", 100)
        mc.reset()
        assert mc.get_counter("steps") == 0.0
        assert mc.get_gauge("frames") == 0.0

    def test_metrics_collector_thread_safety(self):
        from fusion_simulation.service.metrics_server import MetricsCollector

        mc = MetricsCollector()
        errors = []

        def writer(thread_id):
            try:
                for i in range(500):
                    mc.inc_counter("shared_counter", value=1)
                    mc.set_gauge("shared_gauge", float(thread_id * 1000 + i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert mc.get_counter("shared_counter") == 4000.0

    def test_health_endpoint_via_real_http(self):
        import httpx

        from fusion_simulation.service.gateway_client import HealthPayload
        from fusion_simulation.service.metrics_server import (
            MetricsCollector,
            MetricsConfig,
            MetricsServer,
        )

        collector = MetricsCollector()
        config = MetricsConfig(host="127.0.0.1", port=18081)
        server = MetricsServer(config=config, collector=collector)
        health_payload = HealthPayload(
            status="healthy",
            kernel_state="RUNNING",
            frame_count=42,
            sim_time=1.23,
        )
        server.set_health_provider(lambda: health_payload)
        server.start()
        time.sleep(0.3)
        try:
            resp = httpx.get("http://127.0.0.1:18081/health", timeout=5.0)
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"
            assert data["kernel_state"] == "RUNNING"
            assert data["frame_count"] == 42
        finally:
            server.stop()

    def test_prometheus_endpoint_via_real_http(self):
        import httpx

        from fusion_simulation.service.metrics_server import (
            MetricsCollector,
            MetricsConfig,
            MetricsServer,
        )

        collector = MetricsCollector()
        collector.inc_counter("test_counter", value=7)
        config = MetricsConfig(host="127.0.0.1", port=18082)
        server = MetricsServer(config=config, collector=collector)
        server.start()
        time.sleep(0.3)
        try:
            resp = httpx.get("http://127.0.0.1:18082/metrics", timeout=5.0)
            assert resp.status_code == 200
            assert "test_counter" in resp.text
            assert "counter" in resp.text
        finally:
            server.stop()


class TestGatewayClientE2E:
    def test_register_with_disabled_gateway(self):
        from fusion_simulation.service.gateway_client import (
            GatewayClient,
            GatewayConfig,
        )

        cfg = GatewayConfig(enabled=False)
        client = GatewayClient(cfg)
        assert client.register()
        assert client.is_registered
        client.deregister()
        assert not client.is_registered
        client.close()

    def test_register_with_unreachable_gateway(self):
        from fusion_simulation.service.gateway_client import (
            GatewayClient,
            GatewayConfig,
        )

        cfg = GatewayConfig(
            gateway_url="http://127.0.0.1:19999",
            enabled=True,
            heartbeat_interval=1.0,
        )
        client = GatewayClient(cfg)
        result = client.register()
        assert not result
        assert not client.is_registered
        client.close()

    def test_health_provider_integration(self):
        from fusion_simulation.service.gateway_client import (
            GatewayClient,
            GatewayConfig,
            HealthPayload,
        )

        cfg = GatewayConfig(enabled=False)
        client = GatewayClient(cfg)
        payload = HealthPayload(status="healthy", kernel_state="RUNNING")
        client.set_health_provider(lambda: payload)
        health = client._get_health()
        assert health.status == "healthy"
        assert health.kernel_state == "RUNNING"
        client.close()

    def test_close_deregisters(self):
        from fusion_simulation.service.gateway_client import (
            GatewayClient,
            GatewayConfig,
        )

        cfg = GatewayConfig(enabled=False)
        client = GatewayClient(cfg)
        client.register()
        assert client.is_registered
        client.close()
        assert not client.is_registered


# ── Full Pipeline E2E ─────────────────────────────────────────────────


class TestSimulationEnvE2E:
    def test_env_init_step_reset_close(self):
        with (
            patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"),
            patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"),
        ):
            env = SimulationEnv(EnvConfig(headless=True))
            result = env.init()
            assert result.get("status") == "initialized"

            state = env.step()
            assert state.step >= 0

            env.reset()
            state2 = env.step()
            assert state2.step >= 0

            env.close()

    def test_env_repeated_init_close(self):
        with (
            patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"),
            patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"),
        ):
            for i in range(3):
                env = SimulationEnv(EnvConfig(headless=True))
                env.init()
                env.step()
                env.close()
                logger_patch.info("Env cycle %d completed", i)

    def test_env_list_scenes(self):
        scenes = SimulationEnv.list_scenes()
        assert len(scenes) > 0
        names = [s["name"] for s in scenes]
        assert "pick" in names
        assert "push" in names

    def test_env_step_without_init(self):
        env = SimulationEnv(EnvConfig(headless=True))
        state = env.step()
        assert state.error != ""

    def test_env_capture_camera(self):
        with (
            patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"),
            patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"),
        ):
            env = SimulationEnv(EnvConfig(headless=True))
            env.init()
            data = env.capture_camera()
            assert isinstance(data, bytes)
            env.close()


class TestFusionGymEnvE2E:
    def test_gym_reset_step_close(self):
        env = FusionGymEnv(
            agent_config=AgentConfig(name="gym_bot", role=AgentRole.ROBOT, action_dim=4),
            decimation=2,
            max_steps=100,
            headless=True,
        )
        obs, info = env.reset()
        assert isinstance(obs, dict)

        action = np.zeros(4, dtype=np.float32)
        obs2, reward, terminated, truncated, info2 = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

        env.close()

    def test_gym_observation_space(self):
        env = FusionGymEnv(
            agent_config=AgentConfig(name="bot", role=AgentRole.ROBOT, action_dim=6),
            headless=True,
        )
        obs_space = env.single_observation_space
        assert "policy" in obs_space
        act_space = env.action_space
        assert act_space == (6,)
        env.close()

    def test_gym_timeout_truncation(self):
        env = FusionGymEnv(
            agent_config=AgentConfig(name="bot", role=AgentRole.ROBOT, action_dim=4),
            max_steps=5,
            decimation=1,
            headless=True,
        )
        env.reset()
        action = np.zeros(4, dtype=np.float32)
        truncated = False
        for _ in range(10):
            _, _, _, truncated, _ = env.step(action)
            if truncated:
                break
        assert truncated is True
        env.close()

    def test_gym_reward_accumulation(self):
        env = FusionGymEnv(
            agent_config=AgentConfig(name="bot", role=AgentRole.ROBOT, action_dim=4),
            max_steps=50,
            decimation=1,
            headless=True,
        )
        env._reward_mgr.add_reward_fn("constant", lambda obs, info: 1.0)
        env.reset()
        action = np.ones(4, dtype=np.float32)
        total_reward = 0.0
        for _ in range(5):
            _, reward, _, _, _ = env.step(action)
            total_reward += reward
        assert total_reward > 0
        env.close()

    def test_gym_custom_termination(self):
        env = FusionGymEnv(
            agent_config=AgentConfig(name="bot", role=AgentRole.ROBOT, action_dim=4),
            max_steps=1000,
            decimation=1,
            headless=True,
        )
        env._term_mgr.add_termination_fn(
            "always_terminate",
            lambda obs, info: True,
        )
        env.reset()
        action = np.zeros(4, dtype=np.float32)
        _, _, terminated, _, _ = env.step(action)
        assert terminated is True
        env.close()

    def test_gym_repeated_reset(self):
        env = FusionGymEnv(
            agent_config=AgentConfig(name="bot", role=AgentRole.ROBOT, action_dim=4),
            max_steps=100,
            decimation=1,
            headless=True,
        )
        for _ in range(3):
            obs, _ = env.reset()
            assert isinstance(obs, dict)
            action = np.zeros(4, dtype=np.float32)
            env.step(action)
        env.close()


# ── Sensor + Agent Pipeline E2E ───────────────────────────────────────


class TestSensorAgentPipelineE2E:
    def test_multiple_sensors_observe(self):
        from fusion_simulation.sensor.base import SensorBase

        sm = SensorManager()
        sm.add_sensor(SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="cam0"))
        mock_depth = MagicMock(spec=SensorBase)
        mock_depth.get_observation.return_value = {"data": "depth"}
        mock_depth.enabled = True
        sm._sensors["depth0"] = mock_depth
        mock_imu = MagicMock(spec=SensorBase)
        mock_imu.get_observation.return_value = {"data": "imu"}
        mock_imu.enabled = True
        sm._sensors["imu0"] = mock_imu
        obs = sm.get_observations()
        assert "cam0" in obs
        assert "depth0" in obs
        assert "imu0" in obs

    def test_multiple_agents_observe_act(self):
        am = AgentManager()
        am.add_agent(AgentConfig(name="robot0", role=AgentRole.ROBOT, action_dim=6))
        am.add_agent(AgentConfig(name="robot1", role=AgentRole.ROBOT, action_dim=4))
        am.add_agent(AgentConfig(name="observer0", role=AgentRole.OBSERVER, action_dim=0))
        actions = am.step_all()
        assert "robot0" in actions
        assert "robot1" in actions

    def test_sensor_enable_disable(self):
        sm = SensorManager()
        sm.add_sensor(SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="cam0"))
        assert sm.enable_sensor("cam0", enabled=False) is True
        sm.update()
        obs = sm.get_observations()
        assert "cam0" not in obs or obs.get("cam0") is not None

    def test_agent_cumulative_reward_tracking(self):
        am = AgentManager()
        am.add_agent(AgentConfig(name="bot", role=AgentRole.ROBOT, action_dim=4))
        for _ in range(10):
            am.step_all()
        agent = am.get_agent("bot")
        assert agent is not None
        assert agent.step_count == 10

    def test_kernel_full_pipeline_with_sensors_agents_events(self):
        from fusion_simulation.sensor.base import SensorBase

        sm = SensorManager()
        sm.add_sensor(SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="cam0"))
        mock_imu = MagicMock(spec=SensorBase)
        mock_imu.get_observation.return_value = {"accel": [0.0, 0.0, -9.81]}
        mock_imu.enabled = True
        sm._sensors["imu0"] = mock_imu
        am = AgentManager()
        am.add_agent(AgentConfig(name="robot", role=AgentRole.ROBOT, action_dim=6))
        am.add_agent(AgentConfig(name="obs", role=AgentRole.OBSERVER, action_dim=0))
        kernel = SimulationKernel(KernelConfig(headless=True))
        kernel.init(sensor_manager=sm, agent_manager=am)

        event_log = []
        kernel.events.subscribe(EventKind.PHYSICS_POST_STEP, lambda e: event_log.append("post"))
        kernel.events.subscribe(EventKind.SENSOR_DATA_READY, lambda e: event_log.append("sensor"))

        result = kernel.step(num_steps=10)
        assert result.frame_count >= 10
        assert len(event_log) >= 10

        obs = sm.get_observations()
        assert "cam0" in obs
        assert "imu0" in obs

        actions = am.step_all()
        assert "robot" in actions

        kernel.close()


# ── Stability / Stress E2E ────────────────────────────────────────────


class TestStabilityE2E:
    def test_repeated_kernel_cycles_10x(self):
        for i in range(10):
            sm = SensorManager()
            am = AgentManager()
            kernel = SimulationKernel(KernelConfig(headless=True))
            kernel.init(sensor_manager=sm, agent_manager=am)
            kernel.step(num_steps=5)
            kernel.close()

    def test_long_simulation_500_steps(self):
        sm = SensorManager()
        am = AgentManager()
        kernel = SimulationKernel(KernelConfig(headless=True))
        kernel.init(sensor_manager=sm, agent_manager=am)
        result = kernel.step(num_steps=500)
        assert result.frame_count >= 500
        kernel.close()

    def test_concurrent_rpc_requests(self):
        from fusion_simulation.service.server import SimulationServer

        with (
            patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"),
            patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"),
        ):
            srv = SimulationServer()
            srv.handle_request("init", {})
            errors = []
            results = []

            def rpc_step(thread_id):
                try:
                    resp = srv.handle_request("step", {"num_steps": 1})
                    results.append(resp)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=rpc_step, args=(i,)) for i in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            assert len(errors) == 0, f"Concurrent RPC errors: {errors}"
            assert len(results) == 20
            for r in results:
                assert r.get("frame_count", 0) >= 1

            srv.handle_request("close", {})

    def test_sensor_manager_stress_100_sensors(self):
        from fusion_simulation.sensor.base import SensorBase

        sm = SensorManager()
        sm.add_sensor(
            SensorConfig(
                sensor_type=SensorType.RGB_CAMERA,
                name="cam_000",
            )
        )
        for i in range(1, 100):
            mock_s = MagicMock(spec=SensorBase)
            mock_s.get_observation.return_value = {"data": i}
            mock_s.enabled = True
            sm._sensors[f"cam_{i:03d}"] = mock_s
        obs = sm.get_observations()
        assert len(obs) == 100

    def test_agent_manager_stress_50_agents(self):
        am = AgentManager()
        for i in range(50):
            am.add_agent(
                AgentConfig(
                    name=f"bot_{i:03d}",
                    role=AgentRole.ROBOT,
                    action_dim=6,
                )
            )
        actions = am.step_all()
        assert len(actions) == 50

    def test_ecs_stress_1000_entities(self):
        mgr = EntityManager()
        for i in range(1000):
            eid = mgr.create_entity()
            mgr.add_component(eid, Transform(position=[float(i), 0.0, 0.0]))
            mgr.add_component(eid, RigidBody(mass=1.0))
        entities = mgr.list_entities()
        assert len(entities) >= 1000

    def test_event_bus_stress_10000_events(self):
        bus = EventBus()
        counter = [0]
        bus.subscribe(EventKind.PHYSICS_POST_STEP, lambda e: counter.__setitem__(0, counter[0] + 1))
        for _ in range(10000):
            bus.emit(EventKind.PHYSICS_POST_STEP, {})
        assert counter[0] == 10000


# ── GUI / Rendering Stability E2E ─────────────────────────────────────


class TestGUIStabilityE2E:
    def test_metrics_health_endpoint_structure(self):
        import httpx

        from fusion_simulation.service.gateway_client import HealthPayload
        from fusion_simulation.service.metrics_server import (
            MetricsCollector,
            MetricsConfig,
            MetricsServer,
        )

        collector = MetricsCollector()
        collector.inc_counter("fusion_sim_steps_total", value=50)
        collector.set_gauge("fusion_sim_frame_count", 100)
        config = MetricsConfig(host="127.0.0.1", port=18083)
        server = MetricsServer(config=config, collector=collector)
        server.set_health_provider(
            lambda: HealthPayload(
                status="healthy",
                kernel_state="RUNNING",
                frame_count=100,
                sim_time=3.14,
                sensor_count=2,
                agent_count=3,
                uptime_seconds=60.0,
            )
        )
        server.start()
        time.sleep(0.3)
        try:
            resp = httpx.get("http://127.0.0.1:18083/health", timeout=5.0)
            assert resp.status_code == 200
            data = resp.json()
            required_fields = ["status", "kernel_state", "frame_count", "sim_time", "sensor_count", "agent_count"]
            for f in required_fields:
                assert f in data, f"Missing field: {f}"
        finally:
            server.stop()

    def test_metrics_prometheus_format(self):
        import httpx

        from fusion_simulation.service.metrics_server import (
            MetricsCollector,
            MetricsConfig,
            MetricsServer,
        )

        collector = MetricsCollector()
        for i in range(100):
            collector.inc_counter("fusion_sim_steps_total")
            collector.observe_histogram("fusion_sim_step_duration_ms", float(i))
        config = MetricsConfig(host="127.0.0.1", port=18084)
        server = MetricsServer(config=config, collector=collector)
        server.start()
        time.sleep(0.3)
        try:
            resp = httpx.get("http://127.0.0.1:18084/metrics", timeout=5.0)
            assert resp.status_code == 200
            text = resp.text
            assert "fusion_sim_steps_total" in text
            assert "fusion_sim_step_duration_ms_count" in text
            assert "fusion_sim_step_duration_ms_sum" in text
            assert "quantile" in text
        finally:
            server.stop()

    def test_metrics_404_for_unknown_path(self):
        import httpx

        from fusion_simulation.service.metrics_server import (
            MetricsCollector,
            MetricsConfig,
            MetricsServer,
        )

        collector = MetricsCollector()
        config = MetricsConfig(host="127.0.0.1", port=18085)
        server = MetricsServer(config=config, collector=collector)
        server.start()
        time.sleep(0.3)
        try:
            resp = httpx.get("http://127.0.0.1:18085/unknown_path", timeout=5.0)
            assert resp.status_code == 404
        finally:
            server.stop()

    def test_server_event_streaming(self):
        from fusion_simulation.service.server import SimulationServer

        with (
            patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"),
            patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"),
        ):
            srv = SimulationServer()
            srv._running = True
            q = srv.subscribe_events()
            assert q is not None

            srv._ensure_kernel()
            srv.handle_request("step", {"num_steps": 1})

            srv.unsubscribe_events(q)
            assert q not in srv._event_subscribers
            srv.handle_request("close", {})

    def test_server_start_stop_lifecycle(self):
        from fusion_simulation.service.server import SimulationServer

        with (
            patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"),
            patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"),
            patch("grpc.server") as mock_grpc_server,
        ):
            mock_srv = MagicMock()
            mock_grpc_server.return_value = mock_srv
            srv = SimulationServer()
            assert not srv.is_running

            srv.start()
            assert srv.is_running

            srv.stop()
            assert not srv.is_running

    def test_server_repeated_start_stop(self):
        from fusion_simulation.service.server import SimulationServer

        with (
            patch("fusion_simulation.physics.pybullet_engine.PyBulletEngine.init"),
            patch("fusion_simulation.render.pybullet_render.PyBulletRender.init"),
            patch("grpc.server") as mock_grpc_server,
        ):
            mock_srv = MagicMock()
            mock_grpc_server.return_value = mock_srv
            for _ in range(3):
                srv = SimulationServer()
                srv.start()
                assert srv.is_running
                srv.stop()
                assert not srv.is_running


# ── GymEnv Managers E2E ──────────────────────────────────────────────


class TestGymManagersE2E:
    def test_observation_manager_compute_and_history(self):
        om = ObservationManager(
            groups={"policy": ["joint_pos", "joint_vel"]},
            noise_scale={"joint_pos": 0.01},
            history_len=5,
        )
        raw_obs = {
            "joint_pos": [0.1, 0.2, 0.3],
            "joint_vel": [0.01, 0.02, 0.03],
        }
        result = om.compute(raw_obs)
        assert "policy" in result
        assert result["policy"].shape[0] == 6

        for _ in range(4):
            om.compute(raw_obs)
        history = om.get_history("policy")
        assert len(history) == 5

    def test_observation_manager_reset(self):
        om = ObservationManager(groups={"policy": ["x"]}, history_len=10)
        om.compute({"x": [1.0]})
        assert len(om.get_history("policy")) == 1
        om.reset()
        assert len(om.get_history("policy")) == 0

    def test_action_manager_clip_and_scale(self):
        am = ActionManager(
            action_dim=3,
            action_scale=[2.0, 2.0, 2.0],
            action_lower=[-1.0, -1.0, -1.0],
            action_upper=[1.0, 1.0, 1.0],
        )
        am.process_action(np.array([5.0, -5.0, 0.5]))
        applied = am.apply_action()
        assert applied is not None
        # scale first (5*2=10), then clip to [-1,1]: [1.0, -1.0, 1.0]
        np.testing.assert_allclose(applied, [1.0, -1.0, 1.0])

    def test_action_manager_pad_action(self):
        am = ActionManager(action_dim=6)
        am.process_action(np.array([1.0, 2.0]))
        applied = am.apply_action()
        assert applied is not None
        assert applied.shape[0] == 6
        assert applied[0] == 1.0
        assert applied[2] == 0.0

    def test_reward_manager_multiple_fns(self):
        rm = RewardManager()
        rm.add_reward_fn("r1", lambda obs, info: 1.0)
        rm.add_reward_fn("r2", lambda obs, info: 2.0)
        reward = rm.compute({}, {})
        assert reward == 3.0
        assert rm.cumulative == 3.0
        rm.reset()
        assert rm.cumulative == 0.0

    def test_reward_manager_error_handling(self):
        rm = RewardManager()
        rm.add_reward_fn("bad", lambda obs, info: 1 / 0)
        rm.add_reward_fn("good", lambda obs, info: 5.0)
        reward = rm.compute({}, {})
        assert reward == 5.0

    def test_termination_manager_timeout(self):
        tm = TerminationManager(max_steps=3)
        assert not tm.compute_time_out()
        assert not tm.compute_time_out()
        assert tm.compute_time_out()
        tm.reset()
        assert not tm.compute_time_out()

    def test_termination_manager_custom_fn(self):
        tm = TerminationManager(max_steps=1000)
        tm.add_termination_fn("goal", lambda obs, info: obs.get("done", False))
        assert not tm.compute_terminated({}, {})
        assert tm.compute_terminated({"done": True}, {})


# ── CLI E2E ───────────────────────────────────────────────────────────


class TestCLIE2E:
    def test_version_command(self, capsys):
        import sys

        from fusion_simulation.cli import main

        old_argv = sys.argv
        sys.argv = ["fusion-simulation", "version"]
        try:
            main()
        finally:
            sys.argv = old_argv
        captured = capsys.readouterr()
        assert "Fusion-Simulation" in captured.out

    def test_scene_list_command(self, capsys):
        import sys

        from fusion_simulation.cli import main

        old_argv = sys.argv
        sys.argv = ["fusion-simulation", "scene", "list"]
        try:
            main()
        finally:
            sys.argv = old_argv
        captured = capsys.readouterr()
        assert "pick" in captured.out

    def test_agent_spawn_command(self, capsys):
        import sys

        from fusion_simulation.cli import main

        old_argv = sys.argv
        sys.argv = ["fusion-simulation", "agent", "spawn", "--name=robot0", "--action-dim=6"]
        try:
            main()
        finally:
            sys.argv = old_argv
        captured = capsys.readouterr()
        assert "robot0" in captured.out

    def test_sensor_add_command(self, capsys):
        import sys

        from fusion_simulation.cli import main

        old_argv = sys.argv
        sys.argv = ["fusion-simulation", "sensor", "add", "--type=rgb_camera", "--name=cam0"]
        try:
            main()
        finally:
            sys.argv = old_argv
        captured = capsys.readouterr()
        assert "rgb_camera" in captured.out
