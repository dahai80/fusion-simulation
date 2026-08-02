from __future__ import annotations

import logging
from typing import Any

from fusion_simulation.physics.base import BodyState, PhysicsConfig, PhysicsEngine

logger = logging.getLogger(__name__)


class PyBulletEngine(PhysicsEngine):
    def __init__(self) -> None:
        self._client: int = -1
        self._config: PhysicsConfig = PhysicsConfig()
        self._initialized: bool = False
        self._body_map: dict[int, str] = {}

    def init(self, config: PhysicsConfig | None = None, headless: bool = True) -> None:
        if self._initialized:
            logger.warning("PyBulletEngine already initialized, closing first")
            self.close()
        self._config = config or PhysicsConfig()
        try:
            import pybullet as p

            mode = p.DIRECT if headless else p.GUI
            self._client = p.connect(mode)
            p.setGravity(*self._config.gravity, physicsClientId=self._client)
            p.setTimeStep(self._config.time_step, physicsClientId=self._client)
            p.setRealTimeSimulation(0, physicsClientId=self._client)
            p.setPhysicsEngineParameter(
                numSolverIterations=self._config.solver_iterations,
                numSubSteps=self._config.num_sub_steps,
                physicsClientId=self._client,
            )
            self._initialized = True
            logger.info(
                "PyBulletEngine initialized: client=%d, headless=%s, dt=%.4f",
                self._client,
                headless,
                self._config.time_step,
            )
        except ImportError:
            logger.error("PyBullet not installed. Install with: pip install pybullet")
            raise
        except Exception as e:
            logger.error("PyBulletEngine init failed: %s", e)
            raise

    def step(self) -> None:
        if not self._initialized:
            return
        try:
            import pybullet as p

            p.stepSimulation(physicsClientId=self._client)
        except Exception as e:
            logger.error("PyBullet step failed: %s", e)

    def reset(self) -> None:
        if not self._initialized:
            return
        try:
            import pybullet as p

            p.resetSimulation(physicsClientId=self._client)
            p.setGravity(*self._config.gravity, physicsClientId=self._client)
            p.setTimeStep(self._config.time_step, physicsClientId=self._client)
            self._body_map.clear()
            logger.info("PyBulletEngine reset")
        except Exception as e:
            logger.error("PyBullet reset failed: %s", e)

    def close(self) -> None:
        if self._client >= 0:
            try:
                import pybullet as p

                p.disconnect(self._client)
            except Exception:
                logger.debug("Error disconnecting PyBullet client %d", self._client)
        self._client = -1
        self._initialized = False
        self._body_map.clear()
        logger.info("PyBulletEngine closed")

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
            import pybullet as p

            pos = position or [0.0, 0.0, 0.0]
            orn = orientation or [0.0, 0.0, 0.0, 1.0]
            body_id = p.loadURDF(
                urdf_path,
                basePosition=pos,
                baseOrientation=orn,
                useFixedBase=fixed_base or use_fixed_base,
                physicsClientId=self._client,
            )
            self._body_map[body_id] = urdf_path
            logger.info("URDF loaded: %s -> body_id=%d", urdf_path, body_id)
            return body_id
        except Exception as e:
            logger.error("Failed to load URDF %s: %s", urdf_path, e)
            raise

    def load_plane(self, position: list[float] | None = None) -> int:
        self._ensure_initialized()
        try:
            import pybullet as p
            import pybullet_data

            pos = position or [0.0, 0.0, 0.0]
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            body_id = p.loadURDF(
                "plane.urdf",
                basePosition=pos,
                physicsClientId=self._client,
            )
            self._body_map[body_id] = "plane"
            logger.info("Plane loaded: body_id=%d", body_id)
            return body_id
        except Exception as e:
            logger.error("Failed to load plane: %s", e)
            raise

    def remove_body(self, body_id: int) -> None:
        if not self._initialized:
            return
        try:
            import pybullet as p

            p.removeBody(body_id, physicsClientId=self._client)
            self._body_map.pop(body_id, None)
            logger.debug("Body removed: %d", body_id)
        except Exception as e:
            logger.error("Failed to remove body %d: %s", body_id, e)

    def get_body_state(self, body_id: int) -> BodyState:
        self._ensure_initialized()
        try:
            import pybullet as p

            pos, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=self._client)
            lin_vel, ang_vel = p.getBaseVelocity(body_id, physicsClientId=self._client)
            state = BodyState(
                body_id=body_id,
                position=list(pos),
                orientation=list(orn),
                linear_velocity=list(lin_vel),
                angular_velocity=list(ang_vel),
            )
            num_joints = p.getNumJoints(body_id, physicsClientId=self._client)
            if num_joints > 0:
                joint_states = p.getJointStates(body_id, range(num_joints), physicsClientId=self._client)
                state.joint_positions = [js[0] for js in joint_states]
                state.joint_velocities = [js[1] for js in joint_states]
                state.joint_efforts = [js[3] for js in joint_states]
            return state
        except Exception as e:
            logger.error("Failed to get body state for %d: %s", body_id, e)
            return BodyState(body_id=body_id)

    def set_body_position(self, body_id: int, position: list[float], orientation: list[float] | None = None) -> None:
        self._ensure_initialized()
        try:
            import pybullet as p

            orn = orientation or [0.0, 0.0, 0.0, 1.0]
            p.resetBasePositionAndOrientation(
                body_id,
                position,
                orn,
                physicsClientId=self._client,
            )
        except Exception as e:
            logger.error("Failed to set position for body %d: %s", body_id, e)

    def apply_force(self, body_id: int, force: list[float], position: list[float] | None = None) -> None:
        self._ensure_initialized()
        try:
            import pybullet as p

            pos = position or [0.0, 0.0, 0.0]
            p.applyExternalForce(
                body_id,
                -1,
                force,
                pos,
                p.WORLD_FRAME,
                physicsClientId=self._client,
            )
        except Exception as e:
            logger.error("Failed to apply force to body %d: %s", body_id, e)

    def apply_joint_action(
        self, body_id: int, joint_indices: list[int], values: list[float], mode: str = "position"
    ) -> None:
        self._ensure_initialized()
        try:
            import pybullet as p

            for ji, val in zip(joint_indices, values):
                if mode == "position":
                    p.setJointMotorControl2(
                        body_id,
                        ji,
                        p.POSITION_CONTROL,
                        targetPosition=val,
                        physicsClientId=self._client,
                    )
                elif mode == "velocity":
                    p.setJointMotorControl2(
                        body_id,
                        ji,
                        p.VELOCITY_CONTROL,
                        targetVelocity=val,
                        physicsClientId=self._client,
                    )
                elif mode == "effort" or mode == "torque":
                    p.setJointMotorControl2(
                        body_id,
                        ji,
                        p.TORQUE_CONTROL,
                        force=val,
                        physicsClientId=self._client,
                    )
        except Exception as e:
            logger.error("Failed to apply joint action to body %d: %s", body_id, e)

    def get_joint_info(self, body_id: int) -> list[dict[str, Any]]:
        self._ensure_initialized()
        try:
            import pybullet as p

            num_joints = p.getNumJoints(body_id, physicsClientId=self._client)
            infos = []
            for i in range(num_joints):
                info = p.getJointInfo(body_id, i, physicsClientId=self._client)
                infos.append(
                    {
                        "index": info[0],
                        "name": info[1].decode("utf-8") if isinstance(info[1], bytes) else info[1],
                        "type": info[2],
                        "lower_limit": info[8],
                        "upper_limit": info[9],
                        "max_force": info[10],
                        "max_velocity": info[11],
                        "link_name": info[12].decode("utf-8") if isinstance(info[12], bytes) else info[12],
                    }
                )
            return infos
        except Exception as e:
            logger.error("Failed to get joint info for body %d: %s", body_id, e)
            return []

    def ray_test(self, origin: list[float], direction: list[float], max_dist: float = 100.0) -> dict[str, Any] | None:
        self._ensure_initialized()
        try:
            import pybullet as p

            end = [origin[i] + direction[i] * max_dist for i in range(3)]
            result = p.rayTest(origin, end, physicsClientId=self._client)
            if result:
                hit = result[0]
                return {
                    "object_id": hit[0],
                    "link_index": hit[1],
                    "hit_fraction": hit[2],
                    "hit_position": list(hit[3]),
                    "hit_normal": list(hit[4]),
                }
            return None
        except Exception as e:
            logger.error("Ray test failed: %s", e)
            return None

    def get_contact_points(self, body_id: int) -> list[dict[str, Any]]:
        self._ensure_initialized()
        try:
            import pybullet as p

            contacts = p.getContactPoints(bodyA=body_id, physicsClientId=self._client)
            results = []
            for c in contacts:
                results.append(
                    {
                        "body_a": c[1],
                        "body_b": c[2],
                        "link_a": c[3],
                        "link_b": c[4],
                        "position_on_a": list(c[5]),
                        "position_on_b": list(c[6]),
                        "contact_normal": list(c[7]),
                        "normal_force": c[9],
                        "lateral_friction_force_1": c[10],
                        "lateral_friction_force_2": c[12],
                    }
                )
            return results
        except Exception as e:
            logger.error("Failed to get contact points for body %d: %s", body_id, e)
            return []

    def get_camera_image(
        self,
        width: int = 640,
        height: int = 480,
        view_matrix: list[float] | None = None,
        proj_matrix: list[float] | None = None,
        target_position: list[float] | None = None,
        distance: float = 1.5,
        yaw: float = 45.0,
        pitch: float = -30.0,
        fov: float = 60.0,
        near: float = 0.1,
        far: float = 100.0,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        try:
            import numpy as np
            import pybullet as p

            if view_matrix is None:
                tgt = target_position or [0.0, 0.0, 0.0]
                view_matrix = p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=tgt,
                    distance=distance,
                    yaw=yaw,
                    pitch=pitch,
                    roll=0,
                    upAxisIndex=2,
                )
            if proj_matrix is None:
                aspect = width / max(height, 1)
                proj_matrix = p.computeProjectionMatrixFOV(
                    fov=fov,
                    aspect=aspect,
                    nearVal=near,
                    farVal=far,
                )
            _, _, rgb, depth, seg = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                physicsClientId=self._client,
            )
            rgb_array = np.array(rgb, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
            depth_array = np.array(depth, dtype=np.float32).reshape(height, width)
            seg_array = np.array(seg, dtype=np.int32).reshape(height, width)
            return {
                "rgb": rgb_array,
                "depth": depth_array,
                "segmentation": seg_array,
                "width": width,
                "height": height,
            }
        except Exception as e:
            logger.error("Failed to get camera image: %s", e)
            return {"rgb": None, "depth": None, "segmentation": None, "width": width, "height": height}

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("PyBulletEngine not initialized. Call init() first.")
