from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from fusion_simulation.agent.config import AgentConfig, AgentRole
from fusion_simulation.agent.manager import AgentHandle, AgentManager
from fusion_simulation.agent.policy import PolicyClient
from fusion_simulation.core.clock import SimClock, SimTime
from fusion_simulation.core.ecs import (
    Articulation, CameraSensor, Component, EntityId, EntityManager, IMUSensor,
    RigidBody, Transform, _serialize_component, deserialize_component,
)
from fusion_simulation.core.event_bus import Event, EventBus, EventKind
from fusion_simulation.core.kernel import KernelConfig, SimulationKernel
from fusion_simulation.core.world_state import EntitySnapshot, WorldState
from fusion_simulation.dataset.manager import DatasetManager
from fusion_simulation.eval.evaluator import EvalResult, SimulationEvaluator
from fusion_simulation.physics.base import BodyState, PhysicsConfig, PhysicsEngine
from fusion_simulation.physics.pybullet_engine import PyBulletEngine
from fusion_simulation.render.pybullet_render import PyBulletRender
from fusion_simulation.sensor.base import SensorBase, SensorConfig, SensorType
from fusion_simulation.sensor.manager import SensorManager, create_sensor
from fusion_simulation.sensor.rgb_camera import RgbCameraSensor
from fusion_simulation.service.config import ServiceConfig
from fusion_simulation.service.server import SimulationServer
from fusion_simulation.sim.env import EnvConfig, EngineType, SimulationEnv
from fusion_simulation.sim.scene import SceneAsset, SceneConfig, SceneResourceManager
from fusion_simulation.sim.scene_formats.json_loader import JsonSceneLoader
from fusion_simulation.sim.scene_formats.urdf_loader import UrdfLoader
from fusion_simulation.train.gym_env import (
    ActionManager, FusionGymEnv, ObservationManager, RewardManager,
    TerminationManager,
)
from fusion_simulation.train.trainer import BCTrainer


# ── PolicyClient Coverage ──

class TestPolicyClientCoverage:
    def test_endpoint_property(self):
        p = PolicyClient(endpoint="http://test:1234/v1/chat/completions")
        assert p.endpoint == "http://test:1234/v1/chat/completions"

    def test_model_name_property(self):
        p = PolicyClient(model_name="test-model")
        assert p.model_name == "test-model"

    def test_check_available_success(self):
        p = PolicyClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch.object(p._client, "get", return_value=mock_resp):
            assert p.check_available() is True
        p.close()

    def test_check_available_failure(self):
        p = PolicyClient()
        with patch.object(p._client, "get", side_effect=Exception("conn fail")):
            assert p.check_available() is False
        p.close()

    def test_predict_success(self):
        p = PolicyClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "[0.1, 0.2, 0.3]"}}]
        }
        mock_resp.raise_for_status = MagicMock()
        with patch.object(p._client, "post", return_value=mock_resp):
            action = p.predict({"obs": 1.0}, action_dim=3)
        assert action == [0.1, 0.2, 0.3]
        p.close()

    def test_predict_failure_fallback(self):
        p = PolicyClient()
        with patch.object(p._client, "post", side_effect=Exception("fail")):
            action = p.predict({"obs": 1.0}, action_dim=4)
        assert len(action) == 4
        assert all(v == 0.0 for v in action)
        assert not p.is_available
        p.close()

    def test_predict_zero_action_dim(self):
        p = PolicyClient()
        with patch.object(p._client, "post", side_effect=Exception("fail")):
            action = p.predict({"obs": 1.0}, action_dim=0)
        assert len(action) == 1
        p.close()

    def test_parse_action_too_short(self):
        p = PolicyClient()
        result = p._parse_action("[0.1]", 3)
        assert result == [0.1, 0.0, 0.0]

    def test_parse_action_too_long(self):
        p = PolicyClient()
        result = p._parse_action("[0.1, 0.2, 0.3, 0.4]", 2)
        assert result == [0.1, 0.2]

    def test_parse_action_invalid_json(self):
        p = PolicyClient()
        result = p._parse_action("not valid json", 3)
        assert result == [0.0, 0.0, 0.0]

    def test_avg_latency_no_requests(self):
        p = PolicyClient()
        assert p.avg_latency == 0.0
        p.close()

    def test_stats(self):
        p = PolicyClient()
        stats = p.stats()
        assert "endpoint" in stats
        assert "model" in stats
        assert "available" in stats
        assert "request_count" in stats
        p.close()

    def test_close_logs(self):
        p = PolicyClient()
        p.close()


# ── PyBulletEngine Coverage ──

class TestPyBulletEngineCoverage:
    def test_init_already_initialized(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            with patch("pybullet.disconnect"):
                                eng.init(headless=True)
                                eng.init(headless=True)
                                assert eng.is_initialized

    def test_init_import_error(self):
        eng = PyBulletEngine()
        with patch.dict("sys.modules", {"pybullet": None}):
            with pytest.raises(Exception):
                eng.init(headless=True)

    def test_step_not_initialized(self):
        eng = PyBulletEngine()
        eng.step()

    def test_step_with_pybullet(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            with patch("pybullet.stepSimulation"):
                                eng.init(headless=True)
                                eng.step()

    def test_reset_not_initialized(self):
        eng = PyBulletEngine()
        eng.reset()

    def test_reset_with_pybullet(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            with patch("pybullet.resetSimulation"):
                                eng.init(headless=True)
                                eng.reset()

    def test_close_already_closed(self):
        eng = PyBulletEngine()
        eng.close()

    def test_load_urdf(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            with patch("pybullet.loadURDF", return_value=1):
                                eng.init(headless=True)
                                body_id = eng.load_urdf("test.urdf", position=[1, 2, 3])
                                assert body_id == 1

    def test_load_urdf_failure(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.loadURDF", side_effect=Exception("fail")):
                                with pytest.raises(Exception):
                                    eng.load_urdf("bad.urdf")

    def test_load_plane(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            with patch("pybullet_data.getDataPath", return_value="/tmp"):
                                with patch("pybullet.setAdditionalSearchPath"):
                                    with patch("pybullet.loadURDF", return_value=0):
                                        eng.init(headless=True)
                                        body_id = eng.load_plane()
                                        assert body_id == 0

    def test_load_plane_failure(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet_data.getDataPath", return_value="/tmp"):
                                with patch("pybullet.setAdditionalSearchPath"):
                                    with patch("pybullet.loadURDF", side_effect=Exception("fail")):
                                        with pytest.raises(Exception):
                                            eng.load_plane()

    def test_remove_body(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.removeBody"):
                                eng.remove_body(1)

    def test_remove_body_not_initialized(self):
        eng = PyBulletEngine()
        eng.remove_body(1)

    def test_remove_body_failure(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.removeBody", side_effect=Exception("fail")):
                                eng.remove_body(1)

    def test_get_body_state(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.getBasePositionAndOrientation",
                                       return_value=([0, 0, 0], [0, 0, 0, 1])):
                                with patch("pybullet.getBaseVelocity",
                                           return_value=([0, 0, 0], [0, 0, 0])):
                                    with patch("pybullet.getNumJoints", return_value=0):
                                        state = eng.get_body_state(1)
                                        assert state.body_id == 1

    def test_get_body_state_with_joints(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.getBasePositionAndOrientation",
                                       return_value=([0, 0, 0], [0, 0, 0, 1])):
                                with patch("pybullet.getBaseVelocity",
                                           return_value=([0, 0, 0], [0, 0, 0])):
                                    with patch("pybullet.getNumJoints", return_value=2):
                                        with patch("pybullet.getJointStates",
                                                   return_value=[(0.1, 0.2, 0, 0.3), (0.4, 0.5, 0, 0.6)]):
                                            state = eng.get_body_state(1)
                                            assert state.joint_positions == [0.1, 0.4]

    def test_get_body_state_failure(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.getBasePositionAndOrientation",
                                       side_effect=Exception("fail")):
                                state = eng.get_body_state(1)
                                assert state.body_id == 1
                                assert state.position == [0.0, 0.0, 0.0]

    def test_set_body_position(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.resetBasePositionAndOrientation"):
                                eng.set_body_position(1, [1, 2, 3])

    def test_apply_force(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.applyExternalForce"):
                                with patch("pybullet.WORLD_FRAME", 1):
                                    eng.apply_force(1, [10, 0, 0])

    def test_apply_joint_action_position(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.setJointMotorControl2"):
                                with patch("pybullet.POSITION_CONTROL", 0):
                                    eng.apply_joint_action(1, [0], [0.5], mode="position")

    def test_apply_joint_action_velocity(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.setJointMotorControl2"):
                                with patch("pybullet.VELOCITY_CONTROL", 1):
                                    eng.apply_joint_action(1, [0], [1.0], mode="velocity")

    def test_apply_joint_action_effort(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.setJointMotorControl2"):
                                with patch("pybullet.TORQUE_CONTROL", 2):
                                    eng.apply_joint_action(1, [0], [2.0], mode="effort")

    def test_apply_joint_action_torque(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.setJointMotorControl2"):
                                with patch("pybullet.TORQUE_CONTROL", 2):
                                    eng.apply_joint_action(1, [0], [2.0], mode="torque")

    def test_apply_joint_action_failure(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.setJointMotorControl2", side_effect=Exception("fail")):
                                eng.apply_joint_action(1, [0], [0.5])

    def test_get_joint_info(self):
        eng = PyBulletEngine()
        joint_info = [0, b"joint0", 1, 0, 0, 0, 0, 0, -3.14, 3.14, 100, 10, b"link0"]
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.getNumJoints", return_value=1):
                                with patch("pybullet.getJointInfo", return_value=joint_info):
                                    infos = eng.get_joint_info(1)
                                    assert len(infos) == 1
                                    assert infos[0]["name"] == "joint0"

    def test_get_joint_info_failure(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.getNumJoints", side_effect=Exception("fail")):
                                infos = eng.get_joint_info(1)
                                assert infos == []

    def test_ray_test(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.rayTest",
                                       return_value=[(1, 0, 0.5, [1, 2, 3], [0, 0, 1])]):
                                result = eng.ray_test([0, 0, 0], [1, 0, 0])
                                assert result is not None
                                assert result["object_id"] == 1

    def test_ray_test_failure(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.rayTest", side_effect=Exception("fail")):
                                result = eng.ray_test([0, 0, 0], [1, 0, 0])
                                assert result is None

    def test_get_contact_points(self):
        eng = PyBulletEngine()
        contact = [None, 1, 2, 0, 0, [0, 0, 0], [0, 0, 0], [0, 0, 1], 0, 10.0, 0.5, None, 0.3]
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.getContactPoints", return_value=[contact]):
                                pts = eng.get_contact_points(1)
                                assert len(pts) == 1

    def test_get_contact_points_failure(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.getContactPoints", side_effect=Exception("fail")):
                                pts = eng.get_contact_points(1)
                                assert pts == []

    def test_get_camera_image(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.computeViewMatrixFromYawPitchRoll", return_value=[0]*16):
                                with patch("pybullet.computeProjectionMatrixFOV", return_value=[0]*16):
                                    with patch("pybullet.getCameraImage",
                                               return_value=(640, 480, list(range(768)), [0.5]*120, [1]*120)):
                                        with patch("pybullet.ER_BULLET_HARDWARE_OPENGL", 1):
                                            result = eng.get_camera_image(8, 10)
                                            assert result["width"] == 8

    def test_get_camera_image_failure(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.computeViewMatrixFromYawPitchRoll", side_effect=Exception("fail")):
                                result = eng.get_camera_image(64, 48)
                                assert result["rgb"] is None

    def test_ensure_initialized_raises(self):
        eng = PyBulletEngine()
        with pytest.raises(RuntimeError, match="not initialized"):
            eng.load_urdf("test.urdf")

    def test_reset_failure(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.resetSimulation", side_effect=Exception("fail")):
                                eng.reset()


# ── PyBulletRender Coverage ──

class TestPyBulletRenderCoverage:
    def test_capture_not_initialized(self):
        r = PyBulletRender()
        result = r.capture_camera()
        assert result["rgb"] is None

    def test_capture_no_physics(self):
        r = PyBulletRender()
        r.init()
        result = r.capture_camera()
        assert result["rgb"] is None

    def test_capture_with_physics(self):
        mock_physics = MagicMock()
        mock_physics.get_camera_image.return_value = {
            "rgb": np.zeros((48, 64, 3)),
            "depth": np.zeros((48, 64)),
            "segmentation": np.zeros((48, 64), dtype=int),
        }
        r = PyBulletRender(physics_engine=mock_physics)
        r.init()
        result = r.capture_camera(width=64, height=48)
        assert result["rgb"] is not None

    def test_close(self):
        r = PyBulletRender()
        r.init()
        r.close()
        assert not r.is_initialized

    def test_is_initialized(self):
        r = PyBulletRender()
        assert not r.is_initialized
        r.init()
        assert r.is_initialized

    def test_attach_physics(self):
        r = PyBulletRender()
        mock_physics = MagicMock()
        r.attach_physics(mock_physics)
        assert r._physics is mock_physics

    def test_capture_with_custom_dimensions(self):
        mock_physics = MagicMock()
        mock_physics.get_camera_image.return_value = {"rgb": None, "depth": None, "segmentation": None}
        r = PyBulletRender(physics_engine=mock_physics)
        r.init()
        r.capture_camera(width=128, height=96, distance=2.0, yaw=90.0, pitch=-45.0)
        mock_physics.get_camera_image.assert_called_once()


# ── SimulationServer Coverage ──

class TestSimulationServerCoverage:
    def test_kernel_property(self):
        s = SimulationServer()
        assert s.kernel is None

    def test_start_already_running(self):
        s = SimulationServer()
        s._running = True
        s.start()
        assert s._running

    def test_start_grpc_import_error(self):
        s = SimulationServer()
        with patch.dict("sys.modules", {"grpc": None}):
            s.start()
            assert s.is_running

    def test_start_and_stop(self):
        s = SimulationServer()
        mock_grpc_server = MagicMock()
        with patch("grpc.server", return_value=mock_grpc_server):
            with patch("fusion_simulation.service.proto.simulation_pb2_grpc.add_SimulationServiceServicer_to_server"):
                s.start()
                assert s.is_running
                s.stop()

    def test_stop_with_kernel(self):
        s = SimulationServer()
        s._running = True
        mock_kernel = MagicMock()
        s._kernel = mock_kernel
        s.stop()
        mock_kernel.close.assert_called_once()
        assert not s.is_running

    def test_wait_for_termination(self):
        s = SimulationServer()
        s._server = MagicMock()
        s.wait_for_termination()
        s._server.wait_for_termination.assert_called_once()

    def test_wait_for_termination_no_server(self):
        s = SimulationServer()
        s.wait_for_termination()

    def test_handle_request_error(self):
        s = SimulationServer()
        with patch.object(s, "_rpc_init", side_effect=Exception("boom")):
            resp = s.handle_request("init", {})
            assert "error" in resp

    def test_rpc_status_no_kernel(self):
        s = SimulationServer()
        resp = s._rpc_status({})
        assert resp["initialized"] is False

    def test_rpc_add_sensor_no_manager(self):
        s = SimulationServer()
        s._kernel = MagicMock()
        resp = s._rpc_add_sensor({"type": "rgb_camera", "name": "c1"})
        assert "error" in resp

    def test_rpc_add_agent_no_manager(self):
        s = SimulationServer()
        s._kernel = MagicMock()
        resp = s._rpc_add_agent({"name": "bot", "action_dim": 6})
        assert "error" in resp

    def test_rpc_get_observations_no_manager(self):
        s = SimulationServer()
        resp = s._rpc_get_observations({})
        assert "error" in resp

    def test_rpc_save_snapshot(self):
        s = SimulationServer()
        mock_kernel = MagicMock()
        mock_kernel.save_snapshot.return_value = "snap_1"
        s._kernel = mock_kernel
        resp = s._rpc_save_snapshot({"name": "test"})
        assert resp["snapshot_id"] == "snap_1"

    def test_rpc_restore_snapshot(self):
        s = SimulationServer()
        mock_kernel = MagicMock()
        mock_kernel.restore_snapshot.return_value = True
        s._kernel = mock_kernel
        resp = s._rpc_restore_snapshot({"snapshot_id": "snap_1"})
        assert resp["restored"] is True

    def test_rpc_load_scene(self):
        s = SimulationServer()
        mock_kernel = MagicMock()
        mock_kernel.load_builtin_scene.return_value = {"status": "loaded"}
        s._kernel = mock_kernel
        resp = s._rpc_load_scene({"name": "default"})
        assert resp["status"] == "loaded"


# ── Evaluator Coverage ──

class TestEvaluatorCoverage:
    @pytest.mark.asyncio
    async def test_evaluate_normal(self):
        evaluator = SimulationEvaluator()
        with patch.object(evaluator, "_run_episode",
                          AsyncMock(return_value={"success": True, "latency_ms": 10.0, "trajectory_error": 0.1, "steps": 10})):
            result = await evaluator.evaluate("test", episodes=3)
            assert result.task_success_rate == 1.0
            assert result.total_episodes == 3

    @pytest.mark.asyncio
    async def test_evaluate_with_kernel(self):
        evaluator = SimulationEvaluator()
        kernel = MagicMock()
        kernel.agent_manager = None
        with patch.object(evaluator, "_run_kernel_episode",
                          AsyncMock(return_value={"success": True, "latency_ms": 5.0, "trajectory_error": 0.2, "steps": 100})):
            result = await evaluator.evaluate_with_kernel(kernel, episodes=2)
            assert result.task_success_rate == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_with_kernel_agent(self):
        evaluator = SimulationEvaluator()
        kernel = MagicMock()
        mock_am = MagicMock()
        kernel.agent_manager = mock_am
        with patch.object(evaluator, "_run_kernel_episode",
                          AsyncMock(return_value={"success": True, "latency_ms": 5.0, "trajectory_error": 0.2, "steps": 50})):
            result = await evaluator.evaluate_with_kernel(kernel, agent_name="bot", episodes=2)
            mock_am.reset_agent.assert_called_with("bot")

    @pytest.mark.asyncio
    async def test_evaluate_with_kernel_reset_all(self):
        evaluator = SimulationEvaluator()
        kernel = MagicMock()
        mock_am = MagicMock()
        kernel.agent_manager = mock_am
        with patch.object(evaluator, "_run_kernel_episode",
                          AsyncMock(return_value={"success": False, "latency_ms": 0, "trajectory_error": 1.0, "steps": 0})):
            result = await evaluator.evaluate_with_kernel(kernel, episodes=1)
            mock_am.reset_all.assert_called()

    @pytest.mark.asyncio
    async def test_run_episode_exception(self):
        evaluator = SimulationEvaluator()
        with patch("asyncio.sleep", side_effect=RuntimeError("crash")):
            result = await evaluator._run_episode("test", "lerobot")
            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_run_kernel_episode(self):
        evaluator = SimulationEvaluator()
        kernel = MagicMock()
        kernel.agent_manager = None
        result = await evaluator._run_kernel_episode(kernel, "", 10)
        assert result["steps"] == 10

    @pytest.mark.asyncio
    async def test_run_kernel_episode_with_agent(self):
        evaluator = SimulationEvaluator()
        kernel = MagicMock()
        mock_am = MagicMock()
        mock_agent = MagicMock()
        mock_agent.cumulative_reward = 0.8
        mock_am.get_agent.return_value = mock_agent
        kernel.agent_manager = mock_am
        result = await evaluator._run_kernel_episode(kernel, "bot", 10)
        assert result["success"] is True
        assert result["cumulative_reward"] == 0.8

    @pytest.mark.asyncio
    async def test_run_kernel_episode_exception(self):
        evaluator = SimulationEvaluator()
        kernel = MagicMock()
        kernel.step.side_effect = Exception("boom")
        result = await evaluator._run_kernel_episode(kernel, "", 10)
        assert result["success"] is False

    def test_generate_report_markdown(self):
        evaluator = SimulationEvaluator()
        result = EvalResult(task_success_rate=0.8, trajectory_error=0.1, inference_latency_ms=50.0, fps=30.0, total_episodes=10)
        report = evaluator.generate_report(result)
        assert "80.0%" in report

    def test_generate_report_json(self):
        evaluator = SimulationEvaluator()
        result = EvalResult(task_success_rate=0.8, trajectory_error=0.1, inference_latency_ms=50.0, fps=30.0, total_episodes=10)
        report = evaluator.generate_report(result, fmt="json")
        data = json.loads(report)
        assert data["task_success_rate"] == 0.8


# ── Scene Coverage ──

class TestSceneCoverage:
    def test_load_scene_ground_plane_error(self):
        ecs = EntityManager()
        physics = MagicMock()
        physics.load_plane.side_effect = Exception("no plane")
        srm = SceneResourceManager(ecs, physics)
        result = srm.load_scene(SceneConfig(name="test", ground_plane=True))
        assert len(result["errors"]) == 1

    def test_load_scene_asset_error(self):
        ecs = EntityManager()
        physics = MagicMock()
        physics.load_plane.return_value = 0
        physics.load_urdf.side_effect = Exception("no urdf")
        srm = SceneResourceManager(ecs, physics)
        cfg = SceneConfig(
            name="test",
            ground_plane=True,
            assets=[SceneAsset(asset_type="urdf", path="bad.urdf", name="robot")],
        )
        result = srm.load_scene(cfg)
        assert len(result["errors"]) == 1

    def test_load_scene_from_dict(self):
        ecs = EntityManager()
        physics = MagicMock()
        physics.load_plane.return_value = 0
        srm = SceneResourceManager(ecs, physics)
        result = srm.load_scene_from_dict({
            "name": "test",
            "ground_plane": True,
            "assets": [],
        })
        assert result["status"] == "loaded"

    def test_load_scene_from_file_not_found(self):
        ecs = EntityManager()
        physics = MagicMock()
        srm = SceneResourceManager(ecs, physics)
        with pytest.raises(FileNotFoundError):
            srm.load_scene_from_file("/nonexistent/scene.json")

    def test_load_scene_from_file(self):
        ecs = EntityManager()
        physics = MagicMock()
        physics.load_plane.return_value = 0
        srm = SceneResourceManager(ecs, physics)
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_file = Path(tmpdir) / "test_scene.json"
            scene_file.write_text(json.dumps({
                "name": "file_scene",
                "ground_plane": True,
                "assets": [],
            }))
            result = srm.load_scene_from_file(str(scene_file))
            assert result["scene"] == "file_scene"

    def test_load_builtin_unknown(self):
        ecs = EntityManager()
        physics = MagicMock()
        srm = SceneResourceManager(ecs, physics)
        with pytest.raises(ValueError, match="Unknown builtin scene"):
            srm.load_builtin("nonexistent")

    def test_unload_all(self):
        ecs = EntityManager()
        physics = MagicMock()
        physics.load_plane.return_value = 0
        srm = SceneResourceManager(ecs, physics)
        srm.load_scene(SceneConfig(name="test", ground_plane=True))
        physics.remove_body = MagicMock()
        srm.unload_all()
        assert len(srm._loaded_bodies) == 0

    def test_unload_all_body_removal_error(self):
        ecs = EntityManager()
        physics = MagicMock()
        physics.load_plane.return_value = 0
        srm = SceneResourceManager(ecs, physics)
        srm.load_scene(SceneConfig(name="test", ground_plane=True))
        physics.remove_body.side_effect = Exception("fail")
        srm.unload_all()

    def test_list_builtin_scenes(self):
        ecs = EntityManager()
        physics = MagicMock()
        srm = SceneResourceManager(ecs, physics)
        scenes = srm.list_builtin_scenes()
        assert len(scenes) >= 1
        assert any(s["name"] == "default" for s in scenes)

    def test_save_scene(self):
        ecs = EntityManager()
        physics = MagicMock()
        srm = SceneResourceManager(ecs, physics)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "saved_scene.json")
            cfg = SceneConfig(name="saved", assets=[SceneAsset(asset_type="urdf", path="test.urdf", name="bot")])
            srm.save_scene(cfg, out_path)
            data = json.loads(Path(out_path).read_text())
            assert data["name"] == "saved"
            assert len(data["assets"]) == 1

    def test_load_asset_unsupported_type(self):
        ecs = EntityManager()
        physics = MagicMock()
        srm = SceneResourceManager(ecs, physics)
        asset = SceneAsset(asset_type="stl", path="model.stl", name="model")
        with pytest.raises(ValueError, match="Unsupported asset type"):
            srm._load_asset(asset)

    def test_load_asset_fixed_base(self):
        ecs = EntityManager()
        physics = MagicMock()
        physics.load_urdf.return_value = 1
        srm = SceneResourceManager(ecs, physics)
        asset = SceneAsset(asset_type="urdf", path="robot.urdf", fixed_base=True, name="robot")
        body_id = srm._load_asset(asset)
        assert body_id == 1

    def test_load_asset_no_name(self):
        ecs = EntityManager()
        physics = MagicMock()
        physics.load_urdf.return_value = 2
        srm = SceneResourceManager(ecs, physics)
        asset = SceneAsset(asset_type="urdf", path="thing.urdf")
        body_id = srm._load_asset(asset)
        assert body_id == 2


# ── JsonSceneLoader Coverage ──

class TestJsonSceneLoaderCoverage:
    def test_load_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_file = Path(tmpdir) / "scene.json"
            scene_file.write_text(json.dumps({
                "name": "test_scene",
                "description": "A test scene",
                "assets": [{"asset_type": "urdf", "path": "robot.urdf", "name": "robot"}],
            }))
            config = JsonSceneLoader.load(str(scene_file))
            assert config.name == "test_scene"
            assert len(config.assets) == 1

    def test_load_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            JsonSceneLoader.load("/nonexistent/scene.json")

    def test_from_dict(self):
        data = {"name": "scene1", "ground_plane": False, "assets": []}
        config = JsonSceneLoader.from_dict(data)
        assert config.name == "scene1"

    def test_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "out.json")
            config = SceneConfig(name="saved_scene")
            JsonSceneLoader.save(config, out)
            data = json.loads(Path(out).read_text())
            assert data["name"] == "saved_scene"

    def test_validate_no_name(self):
        errors = JsonSceneLoader.validate({"assets": []})
        assert any("name" in e for e in errors)

    def test_validate_asset_no_path(self):
        errors = JsonSceneLoader.validate({"name": "test", "assets": [{"name": "bot"}]})
        assert any("missing path" in e for e in errors)

    def test_validate_ok(self):
        errors = JsonSceneLoader.validate({"name": "test", "assets": [{"path": "x.urdf"}]})
        assert len(errors) == 0


# ── UrdfLoader Coverage ──

class TestUrdfLoaderCoverage:
    def test_parse_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            UrdfLoader.parse("/nonexistent/robot.urdf")

    def test_parse_valid_urdf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            urdf_file = Path(tmpdir) / "robot.urdf"
            urdf_file.write_text("""<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <visual><geometry><box size="1 1 1"/></geometry></visual>
    <inertial><mass value="1"/></inertial>
  </link>
  <link name="child_link"/>
  <joint name="joint0" type="revolute">
    <parent link="base_link"/>
    <child link="child_link"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="5"/>
  </joint>
</robot>""")
            result = UrdfLoader.parse(str(urdf_file))
            assert result["name"] == "test_robot"
            assert result["num_links"] == 2
            assert result["num_joints"] == 1
            assert result["joints"][0]["lower"] == -1.57

    def test_get_joint_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            urdf_file = Path(tmpdir) / "robot.urdf"
            urdf_file.write_text("""<?xml version="1.0"?>
<robot name="r">
  <link name="base"/>
  <link name="arm"/>
  <joint name="j1" type="fixed">
    <parent link="base"/><child link="arm"/>
  </joint>
</robot>""")
            names = UrdfLoader.get_joint_names(str(urdf_file))
            assert "j1" in names

    def test_get_joint_limits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            urdf_file = Path(tmpdir) / "robot.urdf"
            urdf_file.write_text("""<?xml version="1.0"?>
<robot name="r">
  <link name="base"/>
  <link name="arm"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="arm"/>
    <limit lower="-3.14" upper="3.14" effort="50" velocity="10"/>
  </joint>
</robot>""")
            limits = UrdfLoader.get_joint_limits(str(urdf_file))
            assert len(limits) == 1
            assert limits[0]["lower"] == -3.14


# ── ECS Additional Coverage ──

class TestECSCoverage:
    def test_entity_id_eq_non_entity(self):
        eid = EntityId()
        assert eid.__eq__("not_an_entity") is NotImplemented

    def test_destroy_nonexistent(self):
        mgr = EntityManager()
        eid = EntityId(value="nonexistent")
        assert mgr.destroy_entity(eid) is False

    def test_remove_component_nonexistent_entity(self):
        mgr = EntityManager()
        eid = EntityId()
        result = mgr.remove_component(eid, Transform)
        assert result is False

    def test_remove_component_not_found(self):
        mgr = EntityManager()
        eid = mgr.create_entity()
        result = mgr.remove_component(eid, RigidBody)
        assert result is False

    def test_get_component_nonexistent_entity(self):
        mgr = EntityManager()
        eid = EntityId()
        assert mgr.get_component(eid, Transform) is None

    def test_add_component_nonexistent_entity(self):
        mgr = EntityManager()
        eid = EntityId()
        with pytest.raises(KeyError):
            mgr.add_component(eid, Transform())

    def test_serialize_nonexistent(self):
        mgr = EntityManager()
        eid = EntityId()
        assert mgr.serialize_entity(eid) is None

    def test_deserialize_component_unknown(self):
        result = deserialize_component({"_type": "UnknownComponent"})
        assert result is None

    def test_deserialize_component_bad_kwargs(self):
        result = deserialize_component({"_type": "Transform", "bad_field": 42})
        assert result is None


# ── EventBus Additional Coverage ──

class TestEventBusCoverage:
    def test_unsubscribe_not_found(self):
        bus = EventBus()
        result = bus.unsubscribe(EventKind.SIM_STARTED, lambda e: None)
        assert result is False

    def test_global_handler_error(self):
        bus = EventBus()
        bus.subscribe_all(lambda e: 1 / 0)
        bus.emit(EventKind.SIM_STARTED)
        bus.emit(EventKind.SIM_STOPPED)

    def test_get_log_with_kind(self):
        bus = EventBus()
        bus.emit(EventKind.SIM_STARTED)
        bus.emit(EventKind.SIM_STOPPED)
        log = bus.get_log(kind=EventKind.SIM_STARTED)
        assert len(log) == 1

    def test_handler_count_no_kind(self):
        bus = EventBus()
        bus.subscribe(EventKind.SIM_STARTED, lambda e: None)
        bus.subscribe_all(lambda e: None)
        count = bus.handler_count()
        assert count == 2

    def test_handler_count_with_kind(self):
        bus = EventBus()
        bus.subscribe(EventKind.SIM_STARTED, lambda e: None)
        count = bus.handler_count(EventKind.SIM_STARTED)
        assert count == 1

    def test_max_log_size(self):
        bus = EventBus()
        bus._max_log_size = 5
        for i in range(10):
            bus.emit(EventKind.SIM_STARTED, {"i": i})
        assert len(bus._event_log) <= 5


# ── WorldState Additional Coverage ──

class TestWorldStateCoverage:
    def test_get_entity(self):
        ws = WorldState()
        ws.entities["test"] = EntitySnapshot(entity_id="test")
        assert ws.get_entity("test") is not None
        assert ws.get_entity("missing") is None

    def test_set_entity(self):
        ws = WorldState()
        snap = EntitySnapshot(entity_id="e1", position=[1.0, 2.0, 3.0])
        ws.set_entity("e1", snap)
        assert ws.entities["e1"].position == [1.0, 2.0, 3.0]

    def test_remove_entity(self):
        ws = WorldState()
        ws.entities["e1"] = EntitySnapshot(entity_id="e1")
        assert ws.remove_entity("e1") is True
        assert ws.remove_entity("missing") is False

    def test_entity_count(self):
        ws = WorldState()
        ws.entities["e1"] = EntitySnapshot(entity_id="e1", active=True)
        ws.entities["e2"] = EntitySnapshot(entity_id="e2", active=False)
        assert ws.entity_count() == 2
        assert ws.active_entity_count() == 1


# ── SensorBase Additional Coverage ──

class TestSensorBaseCoverage:
    def test_config_property(self):
        cfg = SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="test", entity_id="e1")
        s = RgbCameraSensor(cfg)
        assert s.config is cfg

    def test_entity_id_property(self):
        cfg = SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="test", entity_id="e1")
        s = RgbCameraSensor(cfg)
        assert s.entity_id == "e1"

    def test_name_with_default(self):
        cfg = SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="", entity_id="e1")
        s = RgbCameraSensor(cfg)
        assert "rgb_camera" in s.name

    def test_enabled_setter(self):
        cfg = SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="test")
        s = RgbCameraSensor(cfg)
        s.enabled = False
        assert not s.enabled

    def test_data_property(self):
        cfg = SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="test")
        s = RgbCameraSensor(cfg)
        assert isinstance(s.data, dict)

    def test_last_update_time(self):
        cfg = SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="test")
        s = RgbCameraSensor(cfg)
        assert s.last_update_time == -1.0

    def test_should_update_disabled(self):
        cfg = SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="test", enabled=False)
        s = RgbCameraSensor(cfg)
        assert not s.should_update(0.0)

    def test_should_update_zero_rate(self):
        cfg = SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="test", update_rate=0.0)
        s = RgbCameraSensor(cfg)
        assert s.should_update(0.0)

    def test_get_observation(self):
        cfg = SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="test_cam")
        s = RgbCameraSensor(cfg)
        obs = s.get_observation()
        assert obs["type"] == "rgb_camera"
        assert obs["name"] == "test_cam"

    def test_base_capture_not_implemented(self):
        cfg = SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="test")
        s = SensorBase(cfg)
        with pytest.raises(NotImplementedError):
            s._capture(0.0)


# ── SensorManager Additional Coverage ──

class TestSensorManagerCoverage:
    def test_duplicate_sensor_replaces(self):
        sm = SensorManager()
        cfg1 = SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="cam", params={"width": 64})
        cfg2 = SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="cam", params={"width": 128})
        sm.add_sensor(cfg1)
        sm.add_sensor(cfg2)
        assert sm.sensor_count == 1
        assert sm.get_sensor("cam")._width == 128

    def test_remove_nonexistent(self):
        sm = SensorManager()
        assert sm.remove_sensor("nonexistent") is False

    def test_update_with_exception(self):
        sm = SensorManager()
        cfg = SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="bad_cam")
        sensor = RgbCameraSensor(cfg)
        sensor._capture = MagicMock(side_effect=RuntimeError("boom"))
        sm._sensors["bad_cam"] = sensor
        sm.set_sim_time(0.1)
        sm.update()

    def test_get_sensor_data_not_found(self):
        sm = SensorManager()
        assert sm.get_sensor_data("missing") is None

    def test_enable_sensor_not_found(self):
        sm = SensorManager()
        assert sm.enable_sensor("missing") is False

    def test_enable_disable_sensor(self):
        sm = SensorManager()
        sm.add_sensor(SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="cam"))
        assert sm.enable_sensor("cam", False) is True
        assert not sm.get_sensor("cam").enabled
        assert sm.enable_sensor("cam", True) is True
        assert sm.get_sensor("cam").enabled

    def test_set_physics_engine(self):
        sm = SensorManager()
        mock_engine = MagicMock()
        sm.set_physics_engine(mock_engine)
        assert sm._physics_engine is mock_engine


# ── RgbCameraSensor Additional Coverage ──

class TestRgbCameraSensorCoverage:
    def test_rgb_depth_seg_properties(self):
        cfg = SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="cam")
        cam = RgbCameraSensor(cfg)
        assert cam.rgb is None
        assert cam.depth is None
        assert cam.segmentation is None

    def test_capture_with_physics_engine(self):
        cfg = SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="cam",
                           params={"width": 32, "height": 24})
        cam = RgbCameraSensor(cfg)
        mock_physics = MagicMock()
        mock_physics.get_camera_image.return_value = {
            "rgb": np.zeros((24, 32, 3)),
            "depth": np.zeros((24, 32)),
            "segmentation": np.zeros((24, 32)),
        }
        data = cam.update(sim_time=0.1, physics_engine=mock_physics)
        assert data["rgb"] is True

    def test_capture_physics_exception(self):
        cfg = SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="cam")
        cam = RgbCameraSensor(cfg)
        mock_physics = MagicMock()
        mock_physics.get_camera_image.side_effect = Exception("fail")
        data = cam.update(sim_time=0.1, physics_engine=mock_physics)
        assert data["rgb"] is None

    def test_get_observation_with_shape(self):
        cfg = SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="cam",
                           params={"width": 64, "height": 48})
        cam = RgbCameraSensor(cfg)
        obs = cam.get_observation()
        assert obs["shape"]["width"] == 64

    def test_reset_clears_rgb(self):
        cfg = SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="cam")
        cam = RgbCameraSensor(cfg)
        cam._rgb = np.zeros((10, 10, 3))
        cam.reset()
        assert cam.rgb is None


# ── AgentManager Additional Coverage ──

class TestAgentManagerCoverage:
    def test_duplicate_agent_replaces(self):
        am = AgentManager()
        cfg1 = AgentConfig(name="bot", action_dim=4)
        cfg2 = AgentConfig(name="bot", action_dim=6)
        am.add_agent(cfg1)
        am.add_agent(cfg2)
        assert am.agent_count == 1
        assert am.get_agent("bot").config.action_dim == 6

    def test_remove_nonexistent(self):
        am = AgentManager()
        assert am.remove_agent("missing") is False

    def test_collect_observations_no_agent(self):
        am = AgentManager()
        obs = am.collect_observations("missing")
        assert obs == {}

    def test_collect_observations_with_sensor_manager(self):
        am = AgentManager()
        sm = SensorManager()
        sm.add_sensor(SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="cam"))
        am.set_sensor_manager(sm)
        cfg = AgentConfig(name="bot", action_dim=4)
        am.add_agent(cfg)
        obs = am.collect_observations("bot")
        assert "agent_name" in obs

    def test_collect_observations_with_obs_keys(self):
        am = AgentManager()
        sm = SensorManager()
        sm.add_sensor(SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="cam"))
        am.set_sensor_manager(sm)
        cfg = AgentConfig(name="bot", action_dim=4, obs_keys=["cam"])
        am.add_agent(cfg)
        obs = am.collect_observations("bot")
        assert "cam" in obs

    def test_compute_action_no_agent(self):
        am = AgentManager()
        action = am.compute_action("missing")
        assert action == []

    def test_scale_action_with_bounds(self):
        am = AgentManager()
        cfg = AgentConfig(name="bot", action_dim=2,
                          action_scale=2.0, action_lower=[-1.0, -1.0], action_upper=[1.0, 1.0])
        am.add_agent(cfg)
        scaled = am._scale_action([5.0, -5.0], cfg)
        assert scaled == [1.0, -1.0]

    def test_step_all_done_agent(self):
        am = AgentManager()
        cfg = AgentConfig(name="bot", action_dim=2)
        h = am.add_agent(cfg)
        h._done = True
        actions = am.step_all()
        assert "bot" not in actions

    def test_step_all_decimation(self):
        am = AgentManager()
        cfg = AgentConfig(name="bot", action_dim=2, decimation=2)
        h = am.add_agent(cfg)
        h.record_step([0.1, 0.2])
        actions = am.step_all()
        assert actions["bot"] == [0.1, 0.2]

    def test_reset_agent_not_found(self):
        am = AgentManager()
        assert am.reset_agent("missing") is False


# ── Kernel Additional Coverage ──

class TestKernelCoverage:
    def test_properties(self):
        k = SimulationKernel()
        assert k.physics is None
        assert k.render_engine is None
        assert k.scene is None
        assert k.sensor_manager is None
        assert k.agent_manager is None
        assert not k.is_running

    def test_init_already_initialized(self):
        k = SimulationKernel(KernelConfig(headless=True))
        k.init()
        k.init()
        k.close()

    def test_start_not_initialized(self):
        k = SimulationKernel()
        with pytest.raises(RuntimeError, match="not initialized"):
            k.start()

    def test_start_stop(self):
        k = SimulationKernel(KernelConfig(headless=True))
        k.init()
        k.start()
        assert k.is_running
        k.stop()
        assert not k.is_running
        k.close()

    def test_load_scene(self):
        k = SimulationKernel(KernelConfig(headless=True))
        sm = SensorManager()
        am = AgentManager()
        k.init(sensor_manager=sm, agent_manager=am)
        with patch.object(k._scene, "load_scene", return_value={"status": "loaded"}):
            result = k.load_scene(SceneConfig(name="test"))
            assert result["status"] == "loaded"
        k.close()

    def test_load_scene_no_scene_resource(self):
        k = SimulationKernel(KernelConfig(headless=True))
        k.init()
        k._scene = None
        with pytest.raises(RuntimeError, match="SceneResourceManager not available"):
            k.load_scene(SceneConfig(name="test"))
        k.close()

    def test_load_builtin_scene_no_scene_resource(self):
        k = SimulationKernel(KernelConfig(headless=True))
        k.init()
        k._scene = None
        with pytest.raises(RuntimeError, match="SceneResourceManager not available"):
            k.load_builtin_scene("default")
        k.close()

    def test_close_not_initialized(self):
        k = SimulationKernel()
        k.close()

    def test_reset_not_initialized(self):
        k = SimulationKernel()
        k.reset()

    def test_get_world_state(self):
        k = SimulationKernel(KernelConfig(headless=True))
        k.init()
        ws = k.get_world_state()
        assert isinstance(ws, WorldState)
        k.close()

    def test_save_snapshot_auto_name(self):
        k = SimulationKernel(KernelConfig(headless=True))
        k.init()
        k.step(3)
        snap_id = k.save_snapshot()
        assert "snapshot_" in snap_id
        k.close()

    def test_sync_world_state_with_entities(self):
        k = SimulationKernel(KernelConfig(headless=True))
        k.init()
        eid = k.ecs.create_entity()
        k.ecs.add_component(eid, Transform(position=[1.0, 2.0, 3.0]))
        k.ecs.add_component(eid, RigidBody(linear_velocity=[0.5, 0.0, 0.0]))
        k.ecs.add_component(eid, Articulation(joint_positions=[0.1, 0.2]))
        k.step()
        ws = k.get_world_state()
        eid_str = str(eid)
        assert eid_str in ws.entities
        assert ws.entities[eid_str].position == [1.0, 2.0, 3.0]
        k.close()

    def test_apply_actions_with_agents(self):
        k = SimulationKernel(KernelConfig(headless=True))
        sm = SensorManager()
        am = AgentManager()
        cfg = AgentConfig(name="bot", action_dim=2, entity_id="test_eid")
        am.add_agent(cfg)
        k.init(sensor_manager=sm, agent_manager=am)
        with patch.object(k._physics, "apply_joint_action"):
            k._apply_actions({"bot": [0.1, 0.2]})
        k.close()

    def test_apply_actions_no_physics(self):
        k = SimulationKernel(KernelConfig(headless=True))
        k._physics = None
        k._apply_actions({"bot": [0.1]})

    def test_apply_actions_no_agent_manager(self):
        k = SimulationKernel(KernelConfig(headless=True))
        k.init()
        k._agent_manager = None
        k._apply_actions({"bot": [0.1]})
        k.close()

    def test_apply_actions_agent_not_found(self):
        k = SimulationKernel(KernelConfig(headless=True))
        sm = SensorManager()
        am = AgentManager()
        k.init(sensor_manager=sm, agent_manager=am)
        k._apply_actions({"nonexistent": [0.1]})
        k.close()


# ── CLI Additional Coverage ──

class TestCLICoverageAdditional:
    def test_no_command(self):
        from fusion_simulation.cli import main
        with patch.object(sys, "argv", ["fusion"]):
            with pytest.raises(SystemExit):
                main()

    def test_scene_load(self):
        from fusion_simulation.cli import main
        with patch.object(sys, "argv", ["fusion", "scene", "load", "--name=default"]):
            with patch("fusion_simulation.core.kernel.SimulationKernel.init"):
                with patch("fusion_simulation.core.kernel.SimulationKernel.close"):
                    with patch("fusion_simulation.core.kernel.SimulationKernel.load_builtin_scene",
                               return_value={"status": "loaded"}):
                        with patch("fusion_simulation.core.kernel.SimulationKernel.status",
                                   return_value={"initialized": True}):
                            main()

    def test_train_with_kernel(self):
        from fusion_simulation.cli import main
        with patch.object(sys, "argv", ["fusion", "train", "--dataset=ds", "--use-kernel", "--epochs=1"]):
            with patch("fusion_simulation.dataset.manager.DatasetManager.get", return_value=None):
                with patch("fusion_simulation.dataset.manager.DatasetManager.collect_samples", return_value=[]):
                    with patch("fusion_simulation.train.trainer.BCTrainer.train_with_kernel",
                               AsyncMock(return_value={"status": "completed", "final_loss": 0.5, "elapsed_seconds": 1.0})):
                        with patch("fusion_simulation.train.trainer.BCTrainer.close", AsyncMock()):
                            main()

    def test_test_command(self):
        from fusion_simulation.cli import main
        with patch.object(sys, "argv", ["fusion", "test", "--model=test", "--episodes=1"]):
            with patch("fusion_simulation.eval.evaluator.SimulationEvaluator.evaluate",
                       AsyncMock(return_value=EvalResult(task_success_rate=1.0, total_episodes=1))):
                with patch("fusion_simulation.eval.evaluator.SimulationEvaluator.generate_report",
                           return_value="report"):
                    main()

    def test_bench_command(self):
        from fusion_simulation.cli import main
        with patch.object(sys, "argv", ["fusion", "bench", "--model=test", "--output=/tmp/report.md"]):
            with patch("fusion_simulation.eval.evaluator.SimulationEvaluator.evaluate",
                       AsyncMock(return_value=EvalResult())):
                with patch("fusion_simulation.eval.evaluator.SimulationEvaluator.generate_report",
                           return_value="report"):
                    main()

    def test_kernel_run(self):
        from fusion_simulation.cli import main
        mock_kernel = MagicMock()
        mock_kernel.step.return_value = SimTime(sim_time=0.05, frame_count=5)
        mock_kernel.status.return_value = {"initialized": True}
        with patch.object(sys, "argv", ["fusion", "kernel", "run", "--steps=5", "--headless"]):
            with patch("fusion_simulation.core.kernel.SimulationKernel", return_value=mock_kernel):
                main()

    def test_kernel_status(self):
        from fusion_simulation.cli import main
        with patch.object(sys, "argv", ["fusion", "kernel", "status"]):
            main()

    def test_service_start(self):
        from fusion_simulation.cli import main
        with patch.object(sys, "argv", ["fusion", "service", "start", "--headless"]):
            with patch("fusion_simulation.service.server.SimulationServer.start"):
                with patch("fusion_simulation.service.server.SimulationServer.wait_for_termination"):
                    with patch("fusion_simulation.service.server.SimulationServer.stop"):
                        main()


# ── Trainer Coverage ──

class TestTrainerCoverage:
    @pytest.mark.asyncio
    async def test_train_with_kernel(self):
        trainer = BCTrainer()
        kernel = MagicMock()
        mock_am = MagicMock()
        mock_agent = MagicMock()
        mock_agent.cumulative_reward = 0.5
        mock_am.get_agent.return_value = mock_agent
        kernel.agent_manager = mock_am
        result = await trainer.train_with_kernel(kernel, "bot", epochs=2, steps_per_epoch=5)
        assert result["status"] == "completed"
        assert result["epochs"] == 2
        await trainer.close()

    @pytest.mark.asyncio
    async def test_train_with_kernel_no_agent(self):
        trainer = BCTrainer()
        kernel = MagicMock()
        mock_am = MagicMock()
        mock_am.get_agent.return_value = None
        kernel.agent_manager = mock_am
        result = await trainer.train_with_kernel(kernel, "missing", epochs=1, steps_per_epoch=5)
        assert result["status"] == "completed"
        await trainer.close()

    @pytest.mark.asyncio
    async def test_train_with_kernel_no_agent_manager(self):
        trainer = BCTrainer()
        kernel = MagicMock()
        kernel.agent_manager = None
        result = await trainer.train_with_kernel(kernel, "bot", epochs=1, steps_per_epoch=5)
        assert result["status"] == "completed"
        await trainer.close()


# ── SimClock Additional Coverage ──

class TestSimClockCoverage:
    def test_physics_dt_property(self):
        c = SimClock(physics_dt=0.005)
        assert c.physics_dt == 0.005

    def test_render_dt_property(self):
        c = SimClock(render_dt=0.05)
        assert c.render_dt == 0.05

    def test_wall_elapsed_property(self):
        c = SimClock()
        assert c.wall_elapsed == 0.0

    def test_tick_paused(self):
        c = SimClock()
        c.start()
        c.pause()
        t = c.tick()
        assert t.frame_count == 0

    def test_tick_max_step(self):
        c = SimClock(max_step=2)
        c.start()
        c.tick()
        c.tick()
        t = c.tick()
        assert t.frame_count == 2

    def test_should_render_zero_dt(self):
        c = SimClock(render_dt=0)
        assert c.should_render()


# ── GymEnv Additional Coverage ──

class TestGymEnvCoverage:
    def test_observation_with_groups(self):
        om = ObservationManager(groups={"full": ["x", "y"], "policy": ["x"]})
        r = om.compute({"x": [1.0], "y": [2.0]})
        assert "full" in r
        assert "policy" in r

    def test_action_clipping(self):
        am = ActionManager(action_dim=2, action_lower=[-1, -1], action_upper=[1, 1])
        am.process_action(np.array([5.0, -5.0]))
        action = am.apply_action()
        assert action[0] == 1.0
        assert action[1] == -1.0

    def test_fusion_gym_env_kernel_property(self):
        env = FusionGymEnv(max_steps=10)
        obs, info = env.reset()
        assert env.kernel is not None
        env.close()

    def test_fusion_gym_env_step_decimation(self):
        env = FusionGymEnv(max_steps=50, decimation=2)
        obs, info = env.reset()
        obs, reward, term, tout, info = env.step(np.zeros(6))
        assert isinstance(reward, float)
        env.close()

    def test_fusion_gym_env_done(self):
        env = FusionGymEnv(max_steps=50)
        env._term_mgr.add_termination_fn("always", lambda obs, info: True)
        obs, info = env.reset()
        obs, reward, terminated, timed_out, info = env.step(np.zeros(6))
        assert terminated
        env.close()


# ── SimulationEnv Additional Coverage ──

class TestSimEnvCoverage:
    def test_step_kernel(self):
        env = SimulationEnv(EnvConfig(headless=True))
        with patch("fusion_simulation.core.kernel.SimulationKernel.init"):
            with patch("fusion_simulation.core.kernel.SimulationKernel.step"):
                with patch("fusion_simulation.core.kernel.SimulationKernel.close"):
                    env.init()
                    env._kernel = MagicMock()
                    env._kernel.clock.frame_count = 5
                    state = env.step()
                    assert state.step == 5

    def test_reset_with_kernel(self):
        env = SimulationEnv(EnvConfig(headless=True))
        env._kernel = MagicMock()
        env.reset()
        env._kernel.reset.assert_called_once()
        assert env._state.step == 0

    def test_close_with_kernel(self):
        env = SimulationEnv(EnvConfig(headless=True))
        env._kernel = MagicMock()
        env.close()
        env._kernel.close.assert_called_once()

    def test_step_exception(self):
        env = SimulationEnv()
        env._kernel = MagicMock()
        env._kernel.step.side_effect = Exception("crash")
        state = env.step()
        assert "crash" in state.error

    def test_init_exception(self):
        env = SimulationEnv()
        with patch("fusion_simulation.core.kernel.SimulationKernel.init", side_effect=Exception("fail")):
            result = env.init()
            assert result["status"] == "error"

    def test_kernel_property(self):
        env = SimulationEnv()
        assert env.kernel is None


# ── DatasetManager Additional Coverage ──

class TestDatasetManagerCoverage:
    def test_load_index_bad_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_file = Path(tmpdir) / "index.json"
            idx_file.write_text("not valid json{{{")
            mgr = DatasetManager(data_path=tmpdir)
            assert len(mgr.list()) == 0

    def test_import_source_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DatasetManager(data_path=str(Path(tmpdir) / "ds"))
            result = mgr.import_dataset("test", "/nonexistent/path")
            assert result["status"] == "error"

    def test_delete_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DatasetManager(data_path=tmpdir)
            assert mgr.delete("missing") is False

    def test_collect_samples_no_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DatasetManager(data_path=str(Path(tmpdir) / "ds"))
            samples = mgr.collect_samples("missing", num_samples=5)
            assert len(samples) == 5


# ── Remaining Gap Coverage ──

class TestRemainingGaps:
    def test_agent_manager_reset_agent(self):
        am = AgentManager()
        cfg = AgentConfig(name="bot", action_dim=2)
        am.add_agent(cfg)
        assert am.reset_agent("bot") is True
        assert am.reset_agent("missing") is False

    def test_ecs_has_component(self):
        mgr = EntityManager()
        eid = mgr.create_entity()
        mgr.add_component(eid, Transform())
        assert mgr.has_component(eid, Transform)
        assert not mgr.has_component(eid, RigidBody)
        missing = EntityId()
        assert not mgr.has_component(missing, Transform)

    def test_ecs_get_components(self):
        mgr = EntityManager()
        eid = mgr.create_entity()
        mgr.add_component(eid, Transform())
        comps = mgr.get_components(eid)
        assert Transform in comps

    def test_clock_tick_render(self):
        c = SimClock()
        c.start()
        t = c.tick_render()
        assert t.frame_count == 0

    def test_kernel_world_property(self):
        k = SimulationKernel(KernelConfig(headless=True))
        k.init()
        ws = k.world
        assert isinstance(ws, WorldState)
        k.close()

    def test_kernel_sync_world_no_transform(self):
        k = SimulationKernel(KernelConfig(headless=True))
        k.init()
        eid = k.ecs.create_entity()
        k.ecs.add_component(eid, RigidBody(linear_velocity=[1.0, 0.0, 0.0]))
        k.step()
        k.close()

    def test_cli_print_help(self):
        from fusion_simulation.cli import main
        with patch.object(sys, "argv", ["fusion", "unknown_cmd"]):
            with pytest.raises(SystemExit):
                main()

    def test_cli_service_start_keyboard_interrupt(self):
        from fusion_simulation.cli import main
        with patch.object(sys, "argv", ["fusion", "service", "start"]):
            with patch("fusion_simulation.service.server.SimulationServer.start"):
                with patch("fusion_simulation.service.server.SimulationServer.wait_for_termination",
                           side_effect=KeyboardInterrupt):
                    with patch("fusion_simulation.service.server.SimulationServer.stop"):
                        main()

    def test_scene_unload_destroys_entities(self):
        ecs = EntityManager()
        physics = MagicMock()
        physics.load_plane.return_value = 0
        physics.load_urdf.return_value = 1
        srm = SceneResourceManager(ecs, physics)
        cfg = SceneConfig(name="test", ground_plane=True,
                          assets=[SceneAsset(asset_type="urdf", path="r.urdf", name="robot")])
        srm.load_scene(cfg)
        physics.remove_body = MagicMock()
        srm.unload_all()
        assert len(srm._loaded_entities) == 0

    def test_dataset_count_samples_exception(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DatasetManager(data_path=tmpdir)
            ds_dir = Path(tmpdir) / "bad_ds"
            ds_dir.mkdir()
            bad_file = ds_dir / "data.json"
            bad_file.write_text("not valid json{{{")
            mgr._index = {"bad_ds": {"path": str(ds_dir), "name": "bad_ds"}}
            count = mgr._count_samples(ds_dir)
            assert count == 100

    def test_pybullet_step_exception_path(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.stepSimulation", side_effect=Exception("step fail")):
                                eng.step()

    def test_pybullet_close_exception_path(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.disconnect", side_effect=Exception("fail")):
                                eng.close()
                                assert not eng.is_initialized

    def test_pybullet_set_position_exception(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.resetBasePositionAndOrientation", side_effect=Exception("fail")):
                                eng.set_body_position(1, [1, 2, 3])

    def test_pybullet_apply_force_exception(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.applyExternalForce", side_effect=Exception("fail")):
                                eng.apply_force(1, [10, 0, 0])

    def test_pybullet_ray_test_empty_result(self):
        eng = PyBulletEngine()
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.rayTest", return_value=[]):
                                result = eng.ray_test([0, 0, 0], [1, 0, 0])
                                assert result is None

    def test_pybullet_get_camera_image_success(self):
        eng = PyBulletEngine()
        w, h = 8, 8
        rgb_data = list(range(w * h * 4))
        depth_data = [0.5] * (w * h)
        seg_data = [1] * (w * h)
        with patch("pybullet.connect", return_value=0):
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        with patch("pybullet.setPhysicsEngineParameter"):
                            eng.init(headless=True)
                            with patch("pybullet.computeViewMatrixFromYawPitchRoll", return_value=[0]*16):
                                with patch("pybullet.computeProjectionMatrixFOV", return_value=[0]*16):
                                    with patch("pybullet.getCameraImage",
                                               return_value=(w, h, rgb_data, depth_data, seg_data)):
                                        with patch("pybullet.ER_BULLET_HARDWARE_OPENGL", 1):
                                            result = eng.get_camera_image(w, h)
                                            assert result["rgb"].shape == (h, w, 3)
                                            assert result["depth"].shape == (h, w)

    def test_render_base_concrete(self):
        from fusion_simulation.render.base import RenderConfig, RenderEngine
        class TestRender(RenderEngine):
            def __init__(self):
                self._init = False
            def init(self, config=None):
                self._init = True
            def render(self):
                pass
            def capture_camera(self, **kwargs):
                return {}
            def close(self):
                self._init = False
            @property
            def is_initialized(self):
                return self._init
        r = TestRender()
        r.init()
        assert r.is_initialized
        cfg = RenderConfig()
        assert cfg.width == 640
        r.close()

    def test_physics_base_concrete(self):
        from fusion_simulation.physics.base import PhysicsConfig, PhysicsEngine
        class TestPhysics(PhysicsEngine):
            def __init__(self):
                self._init = False
            def init(self, config=None, headless=True):
                self._init = True
            def step(self):
                pass
            def reset(self):
                pass
            def close(self):
                self._init = False
            def load_urdf(self, **kwargs):
                return 0
            def load_plane(self, **kwargs):
                return 0
            def remove_body(self, body_id):
                pass
            def get_body_state(self, body_id):
                return BodyState(body_id=body_id)
            def set_body_position(self, **kwargs):
                pass
            def apply_force(self, **kwargs):
                pass
            def apply_joint_action(self, **kwargs):
                pass
            def get_joint_info(self, body_id):
                return []
            def ray_test(self, **kwargs):
                return None
            def get_contact_points(self, body_id):
                return []
            @property
            def is_initialized(self):
                return self._init
        p = TestPhysics()
        p.init()
        assert p.is_initialized
        cfg = PhysicsConfig()
        assert cfg.gravity == [0.0, 0.0, -9.81]

    def test_gym_env_obs_missing_key(self):
        om = ObservationManager(groups={"full": ["x", "missing_key"]})
        r = om.compute({"x": [1.0]})
        assert "full" in r
        assert len(r["full"]) == 1

    def test_gym_env_action_wrong_dim(self):
        am = ActionManager(action_dim=4)
        am.process_action(np.array([1.0, 2.0]))
        result = am.apply_action()
        assert len(result) == 4

    def test_gym_env_action_overflow_dim(self):
        am = ActionManager(action_dim=2)
        am.process_action(np.array([1.0, 2.0, 3.0]))
        result = am.apply_action()
        assert len(result) == 2

    def test_gym_env_termination_fn_exception(self):
        tm = TerminationManager()
        tm.add_termination_fn("bad", lambda obs, info: 1/0)
        assert tm.compute_terminated({}, {}) is False

    def test_gym_env_compute_time_out_with_fn(self):
        tm = TerminationManager(max_steps=10)
        tm._timeout_fn = lambda: True
        assert tm.compute_time_out() is True

    def test_gym_env_compute_time_out_fn_exception(self):
        tm = TerminationManager(max_steps=100)
        tm._timeout_fn = lambda: 1/0
        assert tm.compute_time_out() is False

    def test_gym_env_reset_with_seed(self):
        env = FusionGymEnv(max_steps=10)
        obs, info = env.reset(seed=42)
        assert isinstance(obs, dict)
        env.close()

    def test_gym_env_observation_space(self):
        env = FusionGymEnv(max_steps=10)
        assert "policy" in env.single_observation_space
        assert env.single_action_space is not None
        assert env.action_space is not None
        env.close()

    def test_kernel_apply_actions_entity_match(self):
        k = SimulationKernel(KernelConfig(headless=True))
        sm = SensorManager()
        am = AgentManager()
        cfg = AgentConfig(name="bot", action_dim=2, entity_id="test_eid")
        am.add_agent(cfg)
        k.init(sensor_manager=sm, agent_manager=am)
        eid = k.ecs.create_entity()
        k.ecs.add_component(eid, Transform())
        with patch.object(k._physics, "apply_joint_action"):
            k._apply_actions({"bot": [0.1, 0.2]})
        k.close()
