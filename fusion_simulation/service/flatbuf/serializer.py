from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_FLATBUFS_AVAILABLE = False
try:
    from fusion_simulation.service.flatbuf import sensor_flatbuf as fb

    _FLATBUFS_AVAILABLE = True
except ImportError:
    logger.debug("FlatBuffers generated module not available. Run flatc to generate.")


def _build_vec3(builder, x: float, y: float, z: float) -> int:
    fb.Vec3Start(builder)
    fb.Vec3AddX(builder, x)
    fb.Vec3AddY(builder, y)
    fb.Vec3AddZ(builder, z)
    return fb.Vec3End(builder)


def _build_quaternion(builder, w: float, x: float, y: float, z: float) -> int:
    fb.QuaternionStart(builder)
    fb.QuaternionAddW(builder, w)
    fb.QuaternionAddX(builder, x)
    fb.QuaternionAddY(builder, y)
    fb.QuaternionAddZ(builder, z)
    return fb.QuaternionEnd(builder)


def _build_rgb_image(builder, width: int, height: int, data: bytes, channels: int = 3) -> int:
    data_vec = builder.CreateByteVector(data)
    fb.RGBImageStart(builder)
    fb.RGBImageAddWidth(builder, width)
    fb.RGBImageAddHeight(builder, height)
    fb.RGBImageAddChannels(builder, channels)
    fb.RGBImageAddData(builder, data_vec)
    return fb.RGBImageEnd(builder)


def _build_depth_image(builder, width: int, height: int, data: np.ndarray) -> int:
    raw = data.astype(np.float32).tobytes()
    data_vec = builder.CreateByteVector(raw)
    fb.DepthImageStart(builder)
    fb.DepthImageAddWidth(builder, width)
    fb.DepthImageAddHeight(builder, height)
    fb.DepthImageAddData(builder, data_vec)
    return fb.DepthImageEnd(builder)


def _build_imu(builder, accel: list[float], gyro: list[float], orientation: list[float], timestamp: float) -> int:
    accel_off = _build_vec3(builder, *accel[:3])
    gyro_off = _build_vec3(builder, *gyro[:3])
    orient_off = _build_quaternion(builder, *orientation[:4])
    fb.IMUReadingStart(builder)
    fb.IMUReadingAddAcceleration(builder, accel_off)
    fb.IMUReadingAddAngularVelocity(builder, gyro_off)
    fb.IMUReadingAddOrientation(builder, orient_off)
    fb.IMUReadingAddTimestamp(builder, timestamp)
    return fb.IMUReadingEnd(builder)


def _build_contact(builder, body_a: int, body_b: int, position: list[float], normal_force: float) -> int:
    pos_off = _build_vec3(builder, *position[:3])
    fb.ContactInfoStart(builder)
    fb.ContactInfoAddBodyA(builder, body_a)
    fb.ContactInfoAddBodyB(builder, body_b)
    fb.ContactInfoAddPosition(builder, pos_off)
    fb.ContactInfoAddNormalForce(builder, normal_force)
    return fb.ContactInfoEnd(builder)


def _build_bbox(builder, x_min: float, y_min: float, x_max: float, y_max: float, label: str) -> int:
    label_off = builder.CreateString(label)
    fb.BoundingBoxStart(builder)
    fb.BoundingBoxAddXMin(builder, x_min)
    fb.BoundingBoxAddYMin(builder, y_min)
    fb.BoundingBoxAddXMax(builder, x_max)
    fb.BoundingBoxAddYMax(builder, y_max)
    fb.BoundingBoxAddLabel(builder, label_off)
    return fb.BoundingBoxEnd(builder)


def serialize_sensor_data(sensor_data: dict[str, Any]) -> bytes:
    if not _FLATBUFS_AVAILABLE:
        return _fallback_serialize(sensor_data)
    try:
        import flatbuffers

        builder = flatbuffers.Builder(4096)

        sensor_id_off = builder.CreateString(sensor_data.get("sensor_id", ""))
        sensor_type_off = builder.CreateString(sensor_data.get("sensor_type", ""))

        rgb_off = 0
        rgb = sensor_data.get("rgb")
        if rgb is not None:
            img = np.asarray(rgb)
            h, w = img.shape[:2]
            ch = img.shape[2] if img.ndim == 3 else 1
            raw = img.astype(np.uint8).tobytes()
            rgb_off = _build_rgb_image(builder, w, h, raw, ch)

        depth_off = 0
        depth = sensor_data.get("depth")
        if depth is not None:
            img = np.asarray(depth)
            h, w = img.shape[:2]
            depth_off = _build_depth_image(builder, w, h, img)

        imu_off = 0
        imu = sensor_data.get("imu")
        if imu is not None:
            imu_off = _build_imu(
                builder,
                accel=imu.get("acceleration", [0, 0, 0]),
                gyro=imu.get("angular_velocity", [0, 0, 0]),
                orientation=imu.get("orientation", [1, 0, 0, 0]),
                timestamp=imu.get("timestamp", 0.0),
            )

        contacts = sensor_data.get("contacts", [])
        contact_offsets = []
        for c in contacts:
            contact_offsets.append(
                _build_contact(
                    builder,
                    body_a=c.get("body_a", 0),
                    body_b=c.get("body_b", 0),
                    position=c.get("position", [0, 0, 0]),
                    normal_force=c.get("normal_force", 0.0),
                )
            )
        fb.SensorDataFlatStartContactsVector(builder, len(contact_offsets))
        for off in reversed(contact_offsets):
            builder.PrependUOffsetTRelative(off)
        contacts_vec = builder.EndVector()

        bboxes = sensor_data.get("bboxes", [])
        bbox_offsets = []
        for b in bboxes:
            bbox_offsets.append(
                _build_bbox(
                    builder,
                    x_min=b.get("x_min", 0),
                    y_min=b.get("y_min", 0),
                    x_max=b.get("x_max", 0),
                    y_max=b.get("y_max", 0),
                    label=b.get("label", ""),
                )
            )
        fb.SensorDataFlatStartBboxesVector(builder, len(bbox_offsets))
        for off in reversed(bbox_offsets):
            builder.PrependUOffsetTRelative(off)
        bboxes_vec = builder.EndVector()

        fb.SensorDataFlatStart(builder)
        fb.SensorDataFlatAddSensorId(builder, sensor_id_off)
        fb.SensorDataFlatAddSensorType(builder, sensor_type_off)
        fb.SensorDataFlatAddTimestamp(builder, sensor_data.get("timestamp", 0.0))
        fb.SensorDataFlatAddFrameCount(builder, sensor_data.get("frame_count", 0))
        if rgb_off:
            fb.SensorDataFlatAddRgb(builder, rgb_off)
        if depth_off:
            fb.SensorDataFlatAddDepth(builder, depth_off)
        if imu_off:
            fb.SensorDataFlatAddImu(builder, imu_off)
        if contact_offsets:
            fb.SensorDataFlatAddContacts(builder, contacts_vec)
        if bbox_offsets:
            fb.SensorDataFlatAddBboxes(builder, bboxes_vec)
        root = fb.SensorDataFlatEnd(builder)
        builder.Finish(root)
        return bytes(builder.Output())

    except Exception as e:
        logger.error("FlatBuffer sensor serialization failed: %s", e)
        return _fallback_serialize(sensor_data)


def serialize_sim_state(state: dict[str, Any]) -> bytes:
    if not _FLATBUFS_AVAILABLE:
        return _fallback_serialize(state)
    try:
        import flatbuffers

        builder = flatbuffers.Builder(8192)

        entity_offsets = []
        for eid, ent in state.get("entities", {}).items():
            eid_off = builder.CreateString(str(eid))
            pos_off = _build_vec3(builder, *ent.get("position", [0, 0, 0])[:3])
            orn_off = _build_quaternion(builder, *ent.get("orientation", [1, 0, 0, 0])[:4])
            lin_off = _build_vec3(builder, *ent.get("linear_velocity", [0, 0, 0])[:3])
            ang_off = _build_vec3(builder, *ent.get("angular_velocity", [0, 0, 0])[:3])

            jp_off = 0
            jp = ent.get("joint_positions", [])
            if jp:
                fb.EntityStateFlatStartJointPositionsVector(builder, len(jp))
                for v in reversed(jp):
                    builder.PrependFloat32(v)
                jp_off = builder.EndVector()

            jv_off = 0
            jv = ent.get("joint_velocities", [])
            if jv:
                fb.EntityStateFlatStartJointVelocitiesVector(builder, len(jv))
                for v in reversed(jv):
                    builder.PrependFloat32(v)
                jv_off = builder.EndVector()

            fb.EntityStateFlatStart(builder)
            fb.EntityStateFlatAddEntityId(builder, eid_off)
            fb.EntityStateFlatAddPosition(builder, pos_off)
            fb.EntityStateFlatAddOrientation(builder, orn_off)
            fb.EntityStateFlatAddLinearVelocity(builder, lin_off)
            fb.EntityStateFlatAddAngularVelocity(builder, ang_off)
            if jp_off:
                fb.EntityStateFlatAddJointPositions(builder, jp_off)
            if jv_off:
                fb.EntityStateFlatAddJointVelocities(builder, jv_off)
            entity_offsets.append(fb.EntityStateFlatEnd(builder))

        fb.SimStateFrameFlatStartEntitiesVector(builder, len(entity_offsets))
        for off in reversed(entity_offsets):
            builder.PrependUOffsetTRelative(off)
        entities_vec = builder.EndVector()

        fb.SimStateFrameFlatStart(builder)
        fb.SimStateFrameFlatAddSimTime(builder, state.get("sim_time", 0.0))
        fb.SimStateFrameFlatAddFrameCount(builder, state.get("frame_count", 0))
        fb.SimStateFrameFlatAddEntities(builder, entities_vec)
        root = fb.SimStateFrameFlatEnd(builder)
        builder.Finish(root)
        return bytes(builder.Output())

    except Exception as e:
        logger.error("FlatBuffer sim state serialization failed: %s", e)
        return _fallback_serialize(state)


def deserialize_sensor_data(buf: bytes) -> dict[str, Any]:
    if not _FLATBUFS_AVAILABLE:
        return _fallback_deserialize(buf)
    try:
        result = fb.SensorDataFlat.GetRootAsSensorDataFlat(buf, 0)
        data: dict[str, Any] = {
            "sensor_id": result.SensorId().decode() if result.SensorId() else "",
            "sensor_type": result.SensorType().decode() if result.SensorType() else "",
            "timestamp": result.Timestamp(),
            "frame_count": result.FrameCount(),
        }
        rgb = result.Rgb()
        if rgb:
            w, h, ch = rgb.Width(), rgb.Height(), rgb.Channels()
            raw = bytes(rgb.DataAsNumpy()) if hasattr(rgb, "DataAsNumpy") else bytes(rgb.Data())
            data["rgb"] = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, ch)
        depth = result.Depth()
        if depth:
            w, h = depth.Width(), depth.Height()
            raw = bytes(depth.DataAsNumpy()) if hasattr(depth, "DataAsNumpy") else bytes(depth.Data())
            data["depth"] = np.frombuffer(raw, dtype=np.float32).reshape(h, w)
        return data
    except Exception as e:
        logger.error("FlatBuffer sensor deserialization failed: %s", e)
        return _fallback_deserialize(buf)


def deserialize_sim_state(buf: bytes) -> dict[str, Any]:
    if not _FLATBUFS_AVAILABLE:
        return _fallback_deserialize(buf)
    try:
        result = fb.SimStateFrameFlat.GetRootAsSimStateFrameFlat(buf, 0)
        entities = {}
        for i in range(result.EntitiesLength()):
            ent = result.Entities(i)
            eid = ent.EntityId().decode() if ent.EntityId() else str(i)
            entities[eid] = {
                "position": [ent.Position().X(), ent.Position().Y(), ent.Position().Z()],
                "orientation": [
                    ent.Orientation().W(),
                    ent.Orientation().X(),
                    ent.Orientation().Y(),
                    ent.Orientation().Z(),
                ],
                "linear_velocity": [ent.LinearVelocity().X(), ent.LinearVelocity().Y(), ent.LinearVelocity().Z()],
                "angular_velocity": [ent.AngularVelocity().X(), ent.AngularVelocity().Y(), ent.AngularVelocity().Z()],
            }
        return {
            "sim_time": result.SimTime(),
            "frame_count": result.FrameCount(),
            "entities": entities,
        }
    except Exception as e:
        logger.error("FlatBuffer sim state deserialization failed: %s", e)
        return _fallback_deserialize(buf)


def _fallback_serialize(data: dict[str, Any]) -> bytes:
    import json

    try:
        return json.dumps(data, default=str).encode("utf-8")
    except Exception as e:
        logger.error("Fallback serialization failed: %s", e)
        return b"{}"


def _fallback_deserialize(buf: bytes) -> dict[str, Any]:
    import json

    try:
        return json.loads(buf.decode("utf-8"))
    except Exception as e:
        logger.error("Fallback deserialization failed: %s", e)
        return {}
