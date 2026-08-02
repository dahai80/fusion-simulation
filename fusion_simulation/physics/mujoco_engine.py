from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from fusion_simulation.physics.base import BodyState, PhysicsConfig, PhysicsEngine

logger = logging.getLogger(__name__)


class MuJoCoEngine(PhysicsEngine):
    def __init__(self) -> None:
        self._model: Any = None
        self._data: Any = None
        self._config: PhysicsConfig = PhysicsConfig()
        self._initialized: bool = False
        self._body_map: dict[int, str] = {}
        self._loaded_models: dict[int, str] = {}
        self._next_body_id: int = 1

    def init(self, config: PhysicsConfig | None = None, headless: bool = True) -> None:
        if self._initialized:
            logger.warning("MuJoCoEngine already initialized, closing first")
            self.close()
        self._config = config or PhysicsConfig()
        try:
            import mujoco

            self._mj = mujoco
            self._initialized = True
            logger.info(
                "MuJoCoEngine initialized: headless=%s, dt=%.4f, gravity=%s",
                headless,
                self._config.time_step,
                self._config.gravity,
            )
        except ImportError:
            logger.error("MuJoCo not installed. Install with: pip install mujoco")
            raise
        except Exception as e:
            logger.error("MuJoCoEngine init failed: %s", e)
            raise

    def step(self) -> None:
        if not self._initialized or self._data is None:
            return
        try:
            self._mj.mj_step(self._model, self._data)
        except Exception as e:
            logger.error("MuJoCo step failed: %s", e)

    def reset(self) -> None:
        if not self._initialized or self._model is None:
            return
        try:
            self._mj.mj_resetData(self._model, self._data)
            self._mj.mj_forward(self._model, self._data)
            self._body_map.clear()
            self._loaded_models.clear()
            logger.info("MuJoCoEngine reset")
        except Exception as e:
            logger.error("MuJoCo reset failed: %s", e)

    def close(self) -> None:
        self._model = None
        self._data = None
        self._initialized = False
        self._body_map.clear()
        self._loaded_models.clear()
        logger.info("MuJoCoEngine closed")

    def load_urdf(
        self,
        urdf_path: str,
        position: list[float] | None = None,
        orientation: list[float] | None = None,
        fixed_base: bool = False,
        use_fixed_base: bool = False,
    ) -> int:
        self._ensure_initialized()
        try:
            xml_path = urdf_path
            if urdf_path.endswith(".urdf"):
                xml_path = self._convert_urdf_to_mjcf(urdf_path)
            model = self._mj.MjModel.from_xml_path(xml_path)
            data = self._mj.MjData(model)
            if position is not None:
                data.qpos[:3] = position
            if orientation is not None and len(orientation) >= 4:
                if data.qpos.shape[0] > 3:
                    start = 3
                    end = min(start + len(orientation), data.qpos.shape[0])
                    data.qpos[start:end] = orientation[: end - start]
            self._mj.mj_forward(model, data)
            body_id = self._next_body_id
            self._next_body_id += 1
            self._body_map[body_id] = urdf_path
            self._loaded_models[body_id] = urdf_path
            if self._model is None:
                self._model = model
                self._data = data
            logger.info("URDF/MJCF loaded: %s -> body_id=%d", urdf_path, body_id)
            return body_id
        except Exception as e:
            logger.error("Failed to load URDF %s: %s", urdf_path, e)
            raise

    def _convert_urdf_to_mjcf(self, urdf_path: str) -> str:
        cache_dir = os.path.expanduser("~/Library/Fusion/Simulation/mjcf_cache")
        os.makedirs(cache_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(urdf_path))[0]
        mjcf_path = os.path.join(cache_dir, f"{base}.xml")
        if os.path.exists(mjcf_path):
            logger.debug("Using cached MJCF: %s", mjcf_path)
            return mjcf_path
        try:
            import mujoco

            model = mujoco.MjModel.from_xml_path(urdf_path)
            mujoco.mj_saveLastXML(mjcf_path, model)
            logger.info("Converted URDF to MJCF: %s -> %s", urdf_path, mjcf_path)
            return mjcf_path
        except Exception as e:
            logger.warning("Direct URDF load failed, trying mujoco-menagerie: %s", e)
            return urdf_path

    def load_plane(self, position: list[float] | None = None) -> int:
        self._ensure_initialized()
        try:
            xml = """
            <mujoco model="plane">
              <option gravity="0 0 -9.81" timestep="0.01"/>
              <worldbody>
                <geom type="plane" size="10 10 0.1" rgba="0.5 0.5 0.5 1"/>
              </worldbody>
            </mujoco>
            """
            model = self._mj.MjModel.from_xml_string(xml)
            data = self._mj.MjData(model)
            if position is not None:
                data.qpos[:3] = position
            self._mj.mj_forward(model, data)
            body_id = self._next_body_id
            self._next_body_id += 1
            self._body_map[body_id] = "plane"
            if self._model is None:
                self._model = model
                self._data = data
            logger.info("Plane loaded: body_id=%d", body_id)
            return body_id
        except Exception as e:
            logger.error("Failed to load plane: %s", e)
            raise

    def remove_body(self, body_id: int) -> None:
        self._body_map.pop(body_id, None)
        self._loaded_models.pop(body_id, None)
        logger.debug("Body removed: %d (note: MuJoCo does not support dynamic body removal)", body_id)

    def get_body_state(self, body_id: int) -> BodyState:
        self._ensure_initialized()
        if self._data is None:
            return BodyState(body_id=body_id)
        try:
            pos = np.zeros(3)
            orn = np.array([0.0, 0.0, 0.0, 1.0])
            lin_vel = np.zeros(3)
            ang_vel = np.zeros(3)
            if self._data.qpos.shape[0] >= 3:
                pos = np.array(self._data.qpos[:3])
            if self._data.qpos.shape[0] >= 7:
                orn = np.array(self._data.qpos[3:7])
            if self._data.qvel.shape[0] >= 3:
                lin_vel = np.array(self._data.qvel[:3])
            if self._data.qvel.shape[0] >= 6:
                ang_vel = np.array(self._data.qvel[3:6])
            state = BodyState(
                body_id=body_id,
                position=pos.tolist(),
                orientation=orn.tolist(),
                linear_velocity=lin_vel.tolist(),
                angular_velocity=ang_vel.tolist(),
            )
            nq = self._model.nq if self._model is not None else 0
            nv = self._model.nv if self._model is not None else 0
            if nq > 7:
                state.joint_positions = np.array(self._data.qpos[7:nq]).tolist()
            if nv > 6:
                state.joint_velocities = np.array(self._data.qvel[6:nv]).tolist()
            return state
        except Exception as e:
            logger.error("Failed to get body state for %d: %s", body_id, e)
            return BodyState(body_id=body_id)

    def set_body_position(self, body_id: int, position: list[float], orientation: list[float] | None = None) -> None:
        self._ensure_initialized()
        if self._data is None:
            return
        try:
            pos = np.array(position[:3])
            self._data.qpos[:3] = pos
            if orientation is not None and len(orientation) >= 4:
                if self._data.qpos.shape[0] > 3:
                    self._data.qpos[3:7] = np.array(orientation[:4])
            self._mj.mj_forward(self._model, self._data)
        except Exception as e:
            logger.error("Failed to set position for body %d: %s", body_id, e)

    def apply_force(self, body_id: int, force: list[float], position: list[float] | None = None) -> None:
        self._ensure_initialized()
        if self._data is None:
            return
        try:
            f = np.array(force[:3])
            self._data.qfrc_applied[:3] += f
        except Exception as e:
            logger.error("Failed to apply force to body %d: %s", body_id, e)

    def apply_joint_action(
        self, body_id: int, joint_indices: list[int], values: list[float], mode: str = "position"
    ) -> None:
        self._ensure_initialized()
        if self._data is None:
            return
        try:
            for ji, val in zip(joint_indices, values):
                ctrl_idx = ji
                if ctrl_idx < self._model.nu:
                    if mode == "position" or mode == "velocity":
                        self._data.ctrl[ctrl_idx] = val
                    elif mode == "effort" or mode == "torque":
                        if ji < self._data.qfrc_applied.shape[0]:
                            self._data.qfrc_applied[ji] += val
        except Exception as e:
            logger.error("Failed to apply joint action to body %d: %s", body_id, e)

    def get_joint_info(self, body_id: int) -> list[dict[str, Any]]:
        self._ensure_initialized()
        if self._model is None:
            return []
        try:
            infos = []
            for i in range(self._model.njnt):
                jnt_type = self._model.jnt_type[i]
                jnt_name = self._model.joint(i).name
                range_val = self._model.jnt_range[i] if i < len(self._model.jnt_range) else [0, 0]
                infos.append(
                    {
                        "index": i,
                        "name": jnt_name,
                        "type": int(jnt_type),
                        "lower_limit": float(range_val[0]),
                        "upper_limit": float(range_val[1]),
                        "max_force": 0.0,
                        "max_velocity": 0.0,
                        "link_name": jnt_name,
                    }
                )
            return infos
        except Exception as e:
            logger.error("Failed to get joint info for body %d: %s", body_id, e)
            return []

    def ray_test(self, origin: list[float], direction: list[float], max_dist: float = 100.0) -> dict[str, Any] | None:
        self._ensure_initialized()
        if self._model is None or self._data is None:
            return None
        try:
            pnt = np.array(origin, dtype=np.float64)
            vec = np.array(direction, dtype=np.float64)
            vec = vec / (np.linalg.norm(vec) + 1e-10)
            dist, geom_id = self._mj.mj_ray(
                self._model,
                self._data,
                pnt,
                vec,
                None,
                0,
                -1,
            )
            if dist >= 0 and dist <= max_dist:
                hit_pos = pnt + vec * dist
                return {
                    "object_id": int(geom_id),
                    "link_index": -1,
                    "hit_fraction": float(dist / max_dist),
                    "hit_position": hit_pos.tolist(),
                    "hit_normal": [0.0, 0.0, 1.0],
                }
            return None
        except Exception as e:
            logger.error("Ray test failed: %s", e)
            return None

    def get_contact_points(self, body_id: int) -> list[dict[str, Any]]:
        self._ensure_initialized()
        if self._data is None:
            return []
        try:
            results = []
            for i in range(self._data.ncon):
                contact = self._data.contact[i]
                results.append(
                    {
                        "body_a": int(contact.geom1),
                        "body_b": int(contact.geom2),
                        "link_a": -1,
                        "link_b": -1,
                        "position_on_a": np.array(contact.pos).tolist(),
                        "position_on_b": np.array(contact.pos).tolist(),
                        "contact_normal": np.zeros(3).tolist(),
                        "normal_force": float(contact.dist),
                        "lateral_friction_force_1": 0.0,
                        "lateral_friction_force_2": 0.0,
                    }
                )
            return results
        except Exception as e:
            logger.error("Failed to get contact points for body %d: %s", body_id, e)
            return []

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("MuJoCoEngine not initialized. Call init() first.")
