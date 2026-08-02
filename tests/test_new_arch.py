from __future__ import annotations

import numpy as np
import pytest

from fusion_simulation.agent.config import AgentConfig, AgentRole
from fusion_simulation.agent.manager import AgentHandle, AgentManager
from fusion_simulation.agent.policy import PolicyClient
from fusion_simulation.core.clock import SimClock
from fusion_simulation.core.ecs import (
    EntityId,
    EntityManager,
    RigidBody,
    Transform,
)
from fusion_simulation.core.event_bus import EventBus, EventKind
from fusion_simulation.core.kernel import KernelConfig, SimulationKernel
from fusion_simulation.core.world_state import WorldState
from fusion_simulation.sensor.base import SensorConfig, SensorType
from fusion_simulation.sensor.manager import (
    SensorManager,
    create_sensor,
)
from fusion_simulation.sensor.rgb_camera import RgbCameraSensor
from fusion_simulation.service.server import SimulationServer
from fusion_simulation.train.gym_env import (
    ActionManager,
    FusionGymEnv,
    ObservationManager,
    RewardManager,
    TerminationManager,
)


class TestSimClock:
    def test_create(self):
        c = SimClock()
        assert c.sim_time == 0.0
        assert c.frame_count == 0

    def test_tick(self):
        c = SimClock(physics_dt=0.01, render_dt=1.0 / 30.0)
        c.start()
        t = c.tick()
        assert t.sim_time == 0.01
        assert t.frame_count == 1

    def test_should_render(self):
        c = SimClock(physics_dt=0.01, render_dt=0.03)
        c.start()
        c.tick()
        c.tick()
        assert not c.should_render()
        c.tick()
        assert c.should_render()

    def test_pause_resume(self):
        c = SimClock()
        c.start()
        c.tick()
        c.pause()
        assert c.paused
        c.resume()
        assert not c.paused

    def test_reset(self):
        c = SimClock()
        c.start()
        c.tick()
        c.tick()
        c.reset()
        assert c.sim_time == 0.0
        assert c.frame_count == 0

    def test_real_time_factor(self):
        c = SimClock()
        c.start()
        c.tick()
        assert c.real_time_factor() >= 0.0


class TestECS:
    def test_entity_id_unique(self):
        id1 = EntityId()
        id2 = EntityId()
        assert id1 != id2

    def test_entity_id_str_roundtrip(self):
        eid = EntityId()
        s = str(eid)
        assert len(s) == 16

    def test_create_entity(self):
        mgr = EntityManager()
        eid = mgr.create_entity()
        assert mgr.entity_count() == 1

    def test_destroy_entity(self):
        mgr = EntityManager()
        eid = mgr.create_entity()
        mgr.destroy_entity(eid)
        assert mgr.entity_count() == 0

    def test_add_get_component(self):
        mgr = EntityManager()
        eid = mgr.create_entity()
        t = Transform(position=[1.0, 2.0, 3.0])
        mgr.add_component(eid, t)
        got = mgr.get_component(eid, Transform)
        assert got is not None
        assert list(got.position) == [1.0, 2.0, 3.0]

    def test_remove_component(self):
        mgr = EntityManager()
        eid = mgr.create_entity()
        mgr.add_component(eid, Transform())
        mgr.remove_component(eid, Transform)
        assert mgr.get_component(eid, Transform) is None

    def test_query(self):
        mgr = EntityManager()
        e1 = mgr.create_entity()
        mgr.add_component(e1, Transform())
        mgr.add_component(e1, RigidBody())
        e2 = mgr.create_entity()
        mgr.add_component(e2, Transform())
        results = mgr.query(Transform, RigidBody)
        assert len(results) == 1

    def test_list_entities(self):
        mgr = EntityManager()
        mgr.create_entity()
        mgr.create_entity()
        assert len(mgr.list_entities()) == 2

    def test_clear(self):
        mgr = EntityManager()
        mgr.create_entity()
        mgr.create_entity()
        mgr.clear()
        assert mgr.entity_count() == 0

    def test_serialize_entity(self):
        mgr = EntityManager()
        eid = mgr.create_entity()
        mgr.add_component(eid, Transform(position=[1.0, 2.0, 3.0]))
        data = mgr.serialize_entity(eid)
        assert "entity_id" in data
        assert "components" in data

    def test_serialize_all(self):
        mgr = EntityManager()
        eid = mgr.create_entity()
        mgr.add_component(eid, Transform(position=[1.0, 2.0, 3.0]))
        all_data = mgr.serialize_all()
        assert len(all_data) == 1


class TestEventBus:
    def test_subscribe_emit(self):
        bus = EventBus()
        received = []
        bus.subscribe(EventKind.PHYSICS_PRE_STEP, lambda e: received.append(e))
        bus.emit(EventKind.PHYSICS_PRE_STEP, {"sim_time": 0.1})
        assert len(received) == 1

    def test_unsubscribe(self):
        bus = EventBus()
        received = []
        handler = lambda e: received.append(e)
        bus.subscribe(EventKind.SIM_RESET, handler)
        bus.unsubscribe(EventKind.SIM_RESET, handler)
        bus.emit(EventKind.SIM_RESET)
        assert len(received) == 0

    def test_subscribe_all(self):
        bus = EventBus()
        received = []
        bus.subscribe_all(lambda e: received.append(e.kind))
        bus.emit(EventKind.SIM_STARTED)
        bus.emit(EventKind.SIM_STOPPED)
        assert len(received) == 2

    def test_error_isolation(self):
        bus = EventBus()
        bus.subscribe(EventKind.SIM_RESET, lambda e: 1 / 0)
        bus.subscribe(EventKind.SIM_RESET, lambda e: None)
        bus.emit(EventKind.SIM_RESET)

    def test_get_log(self):
        bus = EventBus()
        bus.emit(EventKind.SIM_STARTED, {"sim_time": 0.0})
        log = bus.get_log()
        assert len(log) >= 1

    def test_clear(self):
        bus = EventBus()
        bus.emit(EventKind.SIM_STARTED)
        bus.clear()
        assert len(bus.get_log()) == 0


class TestWorldState:
    def test_snapshot_restore(self):
        ws = WorldState()
        ws.sim_time = 1.0
        ws.frame_count = 100
        snap = ws.snapshot()
        assert snap.sim_time == 1.0
        ws.sim_time = 2.0
        ws.restore(snap)
        assert ws.sim_time == 1.0

    def test_to_dict(self):
        ws = WorldState()
        d = ws.to_dict()
        assert "sim_time" in d
        assert "entities" in d


class TestSensorManager:
    def test_add_remove_sensor(self):
        sm = SensorManager()
        cfg = SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="cam0")
        s = sm.add_sensor(cfg)
        assert sm.sensor_count == 1
        assert s.name == "cam0"
        sm.remove_sensor("cam0")
        assert sm.sensor_count == 0

    def test_get_sensor(self):
        sm = SensorManager()
        cfg = SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="cam1")
        sm.add_sensor(cfg)
        assert sm.get_sensor("cam1") is not None
        assert sm.get_sensor("nonexistent") is None

    def test_list_sensors(self):
        sm = SensorManager()
        sm.add_sensor(SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="a"))
        sm.add_sensor(SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="b"))
        names = sm.list_sensors()
        assert len(names) == 2

    def test_get_observations(self):
        sm = SensorManager()
        sm.add_sensor(SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="cam"))
        obs = sm.get_observations()
        assert "cam" in obs

    def test_reset(self):
        sm = SensorManager()
        sm.add_sensor(SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="cam"))
        sm.reset()
        cam = sm.get_sensor("cam")
        assert cam.frame_count == 0

    def test_status(self):
        sm = SensorManager()
        sm.add_sensor(SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="cam"))
        st = sm.status()
        assert st["sensor_count"] == 1

    def test_create_sensor_unknown_type(self):
        sensor = create_sensor(SensorConfig(sensor_type=SensorType.IMU, name="imu0"))
        assert sensor is not None


class TestRgbCameraSensor:
    def test_create(self):
        cfg = SensorConfig(
            sensor_type=SensorType.RGB_CAMERA,
            name="test_cam",
            params={"width": 64, "height": 48},
        )
        cam = RgbCameraSensor(cfg)
        assert cam._width == 64

    def test_capture_no_physics(self):
        cfg = SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="cam")
        cam = RgbCameraSensor(cfg)
        data = cam.update(sim_time=0.1, physics_engine=None)
        assert data["rgb"] is None

    def test_should_update_rate(self):
        cfg = SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="cam", update_rate=10.0)
        cam = RgbCameraSensor(cfg)
        assert cam.should_update(0.0)
        cam.update(0.0, physics_engine=None)
        assert not cam.should_update(0.05)
        assert cam.should_update(0.11)


class TestAgentManager:
    def test_add_remove_agent(self):
        am = AgentManager()
        cfg = AgentConfig(name="robot0", role=AgentRole.ROBOT, action_dim=6)
        h = am.add_agent(cfg)
        assert am.agent_count == 1
        assert h.name == "robot0"
        am.remove_agent("robot0")
        assert am.agent_count == 0

    def test_get_agent(self):
        am = AgentManager()
        am.add_agent(AgentConfig(name="a1", action_dim=4))
        assert am.get_agent("a1") is not None
        assert am.get_agent("missing") is None

    def test_list_agents(self):
        am = AgentManager()
        am.add_agent(AgentConfig(name="a1", action_dim=4))
        am.add_agent(AgentConfig(name="a2", action_dim=6))
        assert len(am.list_agents()) == 2

    def test_reset_all(self):
        am = AgentManager()
        h = am.add_agent(AgentConfig(name="a1", action_dim=4))
        h.record_step([0.1, 0.2, 0.3, 0.4], reward=1.0)
        am.reset_all()
        assert h.step_count == 0
        assert h.cumulative_reward == 0.0

    def test_close(self):
        am = AgentManager()
        am.add_agent(AgentConfig(name="a1", action_dim=4))
        am.close()
        assert am.agent_count == 0

    def test_status(self):
        am = AgentManager()
        am.add_agent(AgentConfig(name="a1", action_dim=4))
        st = am.status()
        assert st["agent_count"] == 1


class TestAgentHandle:
    def test_record_step(self):
        cfg = AgentConfig(name="test", action_dim=2)
        policy = PolicyClient()
        h = AgentHandle(cfg, policy)
        h.record_step([0.1, 0.2], reward=0.5)
        assert h.step_count == 1
        assert h.last_action == [0.1, 0.2]
        assert h.cumulative_reward == 0.5

    def test_done_flag(self):
        cfg = AgentConfig(name="test", action_dim=2)
        policy = PolicyClient()
        h = AgentHandle(cfg, policy)
        h.record_step([0.0], done=True)
        assert h.done

    def test_reset(self):
        cfg = AgentConfig(name="test", action_dim=2)
        policy = PolicyClient()
        h = AgentHandle(cfg, policy)
        h.record_step([0.1], reward=1.0)
        h.reset()
        assert h.step_count == 0
        assert h.cumulative_reward == 0.0
        assert not h.done


class TestKernelConfig:
    def test_defaults(self):
        cfg = KernelConfig()
        assert cfg.physics_dt == 0.01
        assert cfg.headless is True
        assert cfg.seed == 42

    def test_custom(self):
        cfg = KernelConfig(physics_dt=0.005, headless=False, seed=123)
        assert cfg.physics_dt == 0.005
        assert cfg.headless is False


class TestSimulationKernel:
    def test_create(self):
        k = SimulationKernel()
        assert not k.is_initialized

    def test_init_close(self):
        k = SimulationKernel(KernelConfig(headless=True))
        k.init()
        assert k.is_initialized
        k.close()
        assert not k.is_initialized

    def test_step(self):
        k = SimulationKernel(KernelConfig(headless=True))
        k.init()
        st = k.step(num_steps=5)
        assert st.frame_count == 5
        k.close()

    def test_step_without_init(self):
        k = SimulationKernel()
        with pytest.raises(RuntimeError):
            k.step()

    def test_reset(self):
        k = SimulationKernel(KernelConfig(headless=True))
        k.init()
        k.step(10)
        k.reset()
        assert k.clock.frame_count == 0
        k.close()

    def test_pause_resume(self):
        k = SimulationKernel(KernelConfig(headless=True))
        k.init()
        k.pause()
        assert k.clock.paused
        k.resume()
        assert not k.clock.paused
        k.close()

    def test_save_restore_snapshot(self):
        k = SimulationKernel(KernelConfig(headless=True))
        k.init()
        k.step(5)
        snap_id = k.save_snapshot("test_snap")
        assert snap_id == "test_snap"
        k.step(5)
        assert k.clock.frame_count == 10
        ok = k.restore_snapshot("test_snap")
        assert ok
        ok = k.restore_snapshot("nonexistent")
        assert not ok
        k.close()

    def test_status(self):
        k = SimulationKernel(KernelConfig(headless=True))
        k.init()
        st = k.status()
        assert st["initialized"]
        assert st["entity_count"] == 0
        k.close()

    def test_with_sensor_agent_managers(self):
        k = SimulationKernel(KernelConfig(headless=True))
        sm = SensorManager()
        am = AgentManager()
        k.init(sensor_manager=sm, agent_manager=am)
        assert k.sensor_manager is sm
        assert k.agent_manager is am
        k.step(3)
        k.close()


class TestSimulationServer:
    def test_create(self):
        s = SimulationServer()
        assert not s.is_running

    def test_handle_request_init(self):
        s = SimulationServer()
        resp = s.handle_request("init", {})
        assert resp.get("status") == "initialized"

    def test_handle_request_step(self):
        s = SimulationServer()
        s.handle_request("init", {})
        resp = s.handle_request("step", {"num_steps": 3})
        assert "sim_time" in resp
        assert "frame_count" in resp

    def test_handle_request_status(self):
        s = SimulationServer()
        s.handle_request("init", {})
        resp = s.handle_request("status", {})
        assert resp.get("initialized") is True

    def test_handle_request_reset(self):
        s = SimulationServer()
        s.handle_request("init", {})
        s.handle_request("step", {"num_steps": 5})
        resp = s.handle_request("reset", {})
        assert resp.get("status") == "reset"

    def test_handle_request_unknown(self):
        s = SimulationServer()
        resp = s.handle_request("bogus", {})
        assert "error" in resp

    def test_handle_request_add_sensor(self):
        s = SimulationServer()
        s.handle_request("init", {})
        resp = s.handle_request("add_sensor", {"type": "rgb_camera", "name": "cam0"})
        assert resp.get("status") == "added"

    def test_handle_request_add_agent(self):
        s = SimulationServer()
        s.handle_request("init", {})
        resp = s.handle_request("add_agent", {"name": "bot", "action_dim": 6})
        assert resp.get("status") == "added"

    def test_handle_request_close(self):
        s = SimulationServer()
        s.handle_request("init", {})
        resp = s.handle_request("close", {})
        assert resp.get("status") == "closed"

    def test_stop_without_start(self):
        s = SimulationServer()
        s.stop()


class TestObservationManager:
    def test_compute_empty(self):
        om = ObservationManager(groups={"policy": []})
        result = om.compute({"x": [1.0]})
        assert result == {}

    def test_compute_with_keys(self):
        om = ObservationManager(groups={"policy": ["pos", "vel"]})
        result = om.compute({"pos": [1.0, 2.0], "vel": [0.5]})
        assert "policy" in result
        assert result["policy"].shape == (3,)

    def test_noise(self):
        om = ObservationManager(groups={"policy": ["x"]}, noise_scale={"x": 0.1})
        r1 = om.compute({"x": [1.0]})
        r2 = om.compute({"x": [1.0]})
        assert r1["policy"][0] != r2["policy"][0]

    def test_history(self):
        om = ObservationManager(groups={"policy": ["x"]}, history_len=3)
        for i in range(5):
            om.compute({"x": [float(i)]})
        hist = om.get_history("policy")
        assert len(hist) == 3

    def test_reset(self):
        om = ObservationManager(groups={"policy": ["x"]})
        om.compute({"x": [1.0]})
        om.reset()
        assert om.get_history("policy") == []


class TestActionManager:
    def test_process_apply(self):
        am = ActionManager(action_dim=4)
        am.process_action(np.array([0.1, 0.2, 0.3, 0.4]))
        action = am.apply_action()
        assert action is not None
        assert len(action) == 4

    def test_padding(self):
        am = ActionManager(action_dim=6)
        am.process_action(np.array([0.1, 0.2]))
        action = am.apply_action()
        assert len(action) == 6

    def test_clipping(self):
        am = ActionManager(action_dim=2, action_lower=[-1.0, -1.0], action_upper=[1.0, 1.0])
        am.process_action(np.array([5.0, -5.0]))
        action = am.apply_action()
        assert action[0] == 1.0
        assert action[1] == -1.0

    def test_scaling(self):
        am = ActionManager(action_dim=2, action_scale=[2.0, 3.0])
        am.process_action(np.array([1.0, 1.0]))
        action = am.apply_action()
        assert action[0] == 2.0
        assert action[1] == 3.0

    def test_apply_consumes(self):
        am = ActionManager(action_dim=2)
        am.process_action(np.array([1.0, 0.0]))
        am.apply_action()
        assert am.apply_action() is None

    def test_reset(self):
        am = ActionManager(action_dim=2)
        am.process_action(np.array([1.0, 0.0]))
        am.reset()
        assert am.apply_action() is None


class TestRewardManager:
    def test_compute(self):
        rm = RewardManager()
        rm.add_reward_fn("r1", lambda obs, info: 1.0)
        rm.add_reward_fn("r2", lambda obs, info: 0.5)
        r = rm.compute({}, {})
        assert r == 1.5
        assert rm.cumulative == 1.5

    def test_step_reward(self):
        rm = RewardManager()
        rm.add_reward_fn("r1", lambda obs, info: 2.0)
        rm.compute({}, {})
        assert rm.step_reward == 2.0

    def test_reset(self):
        rm = RewardManager()
        rm.add_reward_fn("r1", lambda obs, info: 1.0)
        rm.compute({}, {})
        rm.reset()
        assert rm.cumulative == 0.0

    def test_error_isolation(self):
        rm = RewardManager()
        rm.add_reward_fn("bad", lambda obs, info: 1 / 0)
        rm.add_reward_fn("good", lambda obs, info: 1.0)
        r = rm.compute({}, {})
        assert r == 1.0


class TestTerminationManager:
    def test_no_termination(self):
        tm = TerminationManager(max_steps=100)
        assert not tm.compute_terminated({}, {})

    def test_termination_fn(self):
        tm = TerminationManager(max_steps=100)
        tm.add_termination_fn("fail", lambda obs, info: True)
        assert tm.compute_terminated({}, {})

    def test_timeout(self):
        tm = TerminationManager(max_steps=3)
        assert not tm.compute_time_out()
        assert not tm.compute_time_out()
        assert tm.compute_time_out()

    def test_reset(self):
        tm = TerminationManager(max_steps=2)
        tm.compute_time_out()
        tm.reset()
        assert tm._current_step == 0


class TestFusionGymEnv:
    def test_reset_step_close(self):
        env = FusionGymEnv(max_steps=50)
        obs, info = env.reset()
        assert isinstance(obs, dict)
        obs, reward, terminated, timed_out, info = env.step(np.zeros(6))
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(timed_out, bool)
        env.close()

    def test_timeout(self):
        env = FusionGymEnv(max_steps=3, decimation=1)
        env.reset()
        for _ in range(5):
            obs, reward, terminated, timed_out, info = env.step(np.zeros(6))
            if timed_out:
                break
        assert timed_out
        env.close()

    def test_reward_integration(self):
        env = FusionGymEnv(max_steps=50)
        env._reward_mgr.add_reward_fn("constant", lambda obs, info: 1.0)
        env.reset()
        obs, reward, terminated, timed_out, info = env.step(np.zeros(6))
        assert reward == 1.0
        env.close()

    def test_termination_integration(self):
        env = FusionGymEnv(max_steps=50)
        env._term_mgr.add_termination_fn("always", lambda obs, info: True)
        env.reset()
        obs, reward, terminated, timed_out, info = env.step(np.zeros(6))
        assert terminated
        env.close()
