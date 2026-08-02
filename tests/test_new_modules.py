from __future__ import annotations

import logging

from fusion_simulation.agent.joint_controller import (
    JointController,
    JointControlMode,
)
from fusion_simulation.agent.task_templates import (
    TaskTemplate,
    get_template,
    list_templates,
    register_template,
)
from fusion_simulation.core.ecs import EntityId, EntityManager
from fusion_simulation.core.fault import FaultIsolationManager, FaultLevel
from fusion_simulation.core.plugin import PluginInfo, PluginManager
from fusion_simulation.core.resource import ResourceQuota, ResourceQuotaManager
from fusion_simulation.dataset.collector import (
    DataCollector,
    DRConfig,
    RecordingConfig,
)
from fusion_simulation.sensor.base import SensorConfig, SensorType
from fusion_simulation.sensor.contact import ContactSensor
from fusion_simulation.sensor.depth_camera import DepthCameraSensor
from fusion_simulation.sensor.imu import ImuSensor
from fusion_simulation.sensor.semantic_camera import SemanticCameraSensor
from fusion_simulation.sim.scene import SceneAsset, SceneConfig

logger = logging.getLogger(__name__)


class TestEntityManager:
    def test_create_entity(self):
        ecs = EntityManager()
        eid = ecs.create_entity()
        assert isinstance(eid, EntityId)

    def test_add_get_component(self):
        ecs = EntityManager()
        eid = ecs.create_entity()
        from fusion_simulation.core.ecs import Transform

        t = Transform(entity_id=eid, position=[0, 0, 0], orientation=[0, 0, 0, 1])
        ecs.add_component(eid, t)
        got = ecs.get_component(eid, Transform)
        assert got is not None


class TestFaultIsolation:
    def test_report_warning(self):
        fm = FaultIsolationManager()
        ok = fm.report_fault("agent", "robot0", FaultLevel.WARNING, "slow")
        assert ok is True

    def test_critical_isolates(self):
        fm = FaultIsolationManager()
        ok = fm.report_fault("sensor", "cam0", FaultLevel.CRITICAL, "crash")
        assert ok is False
        assert fm.is_isolated("sensor", "cam0") is True

    def test_max_faults_isolates(self):
        fm = FaultIsolationManager()
        fm.set_max_faults(3)
        for i in range(3):
            fm.report_fault("agent", "a0", FaultLevel.ERROR, f"err{i}")
        assert fm.is_isolated("agent", "a0") is True

    def test_reset_component(self):
        fm = FaultIsolationManager()
        fm.report_fault("agent", "a0", FaultLevel.CRITICAL, "boom")
        assert fm.reset_component("agent", "a0") is True
        assert fm.is_isolated("agent", "a0") is False


class TestPluginManager:
    def test_register_and_emit(self):
        pm = PluginManager()
        results = []
        pm.register_hook("on_step", lambda: results.append("step"))
        pm.emit("on_step")
        assert results == ["step"]

    def test_register_plugin(self):
        pm = PluginManager()
        pm.register_plugin("test", object(), PluginInfo(name="test", version="1.0"))
        info = pm.list_plugins()
        assert len(info) == 1
        assert info[0]["name"] == "test"

    def test_unregister(self):
        pm = PluginManager()
        pm.register_plugin("test", object(), PluginInfo(name="test"))
        assert pm.unregister_plugin("test") is True
        assert pm.get_plugin("test") is None


class TestResourceQuota:
    def test_check_limits(self):
        rqm = ResourceQuotaManager()
        assert rqm.check_agent_limit(5) is True
        assert rqm.check_agent_limit(16) is False

    def test_step_timing(self):
        rqm = ResourceQuotaManager(ResourceQuota(max_step_time_ms=100.0))
        rqm.step_begin()
        ok = rqm.step_end()
        assert ok is True

    def test_get_status(self):
        rqm = ResourceQuotaManager()
        status = rqm.get_status()
        assert "quota" in status
        assert "usage" in status


class TestDataCollector:
    def test_start_stop(self):
        dc = DataCollector()
        rec_id = dc.start_recording()
        assert rec_id.startswith("rec_")
        result = dc.stop_recording()
        assert result.recording_id == rec_id

    def test_collect_frames(self):
        dc = DataCollector()
        rec_id = dc.start_recording(RecordingConfig(max_frames=5))
        for i in range(5):
            dc.collect_frame(sim_time=i * 0.01)
        result = dc.stop_recording()
        assert result.total_frames == 5

    def test_export_raw(self, tmp_path):
        dc = DataCollector()
        rec_id = dc.start_recording()
        dc.collect_frame(sim_time=0.0)
        dc.stop_recording()
        out = dc.export_dataset(rec_id, fmt="raw", output_dir=str(tmp_path))
        assert "recording.json" in out

    def test_domain_randomize(self):
        dc = DataCollector()
        result = dc.domain_randomize(
            DRConfig(
                randomize_lighting=True,
                randomize_positions=True,
                num_variations=3,
            )
        )
        assert result["num_variations"] == 3
        assert len(result["variations"]) == 3


class TestJointController:
    def test_create(self):
        jc = JointController(body_id=1, num_joints=6)
        assert jc.num_joints == 6
        assert jc.mode == JointControlMode.POSITION

    def test_set_mode(self):
        jc = JointController(body_id=1, num_joints=6)
        jc.set_mode(JointControlMode.VELOCITY)
        assert jc.mode == JointControlMode.VELOCITY

    def test_set_targets_clamped(self):
        jc = JointController(body_id=1, num_joints=2)
        jc.set_position_limits([(-1.0, 1.0), (-2.0, 2.0)])
        jc.set_targets([5.0, -5.0])
        targets = jc.get_targets()
        assert targets[0] == 1.0
        assert targets[1] == -2.0

    def test_reset(self):
        jc = JointController(body_id=1, num_joints=3)
        jc.set_targets([1.0, 2.0, 3.0])
        jc.reset()
        assert jc.get_targets() == [0.0, 0.0, 0.0]


class TestTaskTemplates:
    def test_builtins_exist(self):
        templates = list_templates()
        names = [t["name"] for t in templates]
        assert "single_robot_pick" in names
        assert "dual_robot_coop" in names

    def test_get_template(self):
        t = get_template("single_robot_pick")
        assert t is not None
        assert t.scene_name == "pick"
        assert len(t.agent_configs) == 1

    def test_register_custom(self):
        register_template(
            TaskTemplate(
                name="custom_test",
                description="test",
                agent_configs=[],
            )
        )
        t = get_template("custom_test")
        assert t is not None


class TestSceneConfigRich:
    def test_scene_config_default(self):
        cfg = SceneConfig()
        assert cfg is not None

    def test_scene_asset_extended_fields(self):
        asset = SceneAsset(name="arm")
        assert asset.name == "arm"


class TestSensorTypes:
    def test_depth_camera_create(self):
        cfg = SensorConfig(sensor_type=SensorType.DEPTH_CAMERA, name="depth0")
        sensor = DepthCameraSensor(cfg)
        assert sensor.config.name == "depth0"

    def test_semantic_camera_create(self):
        cfg = SensorConfig(sensor_type=SensorType.SEGMENTATION_CAMERA, name="seg0")
        sensor = SemanticCameraSensor(cfg)
        assert sensor.config.name == "seg0"

    def test_imu_create(self):
        cfg = SensorConfig(sensor_type=SensorType.IMU, name="imu0")
        sensor = ImuSensor(cfg)
        assert sensor.config.name == "imu0"

    def test_contact_create(self):
        cfg = SensorConfig(sensor_type=SensorType.CONTACT, name="contact0")
        sensor = ContactSensor(cfg)
        assert sensor.config.name == "contact0"
