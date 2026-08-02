from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class JointControlMode(Enum):
    POSITION = "position"
    VELOCITY = "velocity"
    EFFORT = "effort"


@dataclass
class JointTarget:
    joint_index: int
    value: float
    mode: JointControlMode = JointControlMode.POSITION
    kp: float = 1.0
    kd: float = 0.1


@dataclass
class JointState:
    position: float = 0.0
    velocity: float = 0.0
    effort: float = 0.0


class JointController:
    def __init__(self, body_id: int, num_joints: int, physics_engine: Any = None) -> None:
        self._body_id = body_id
        self._num_joints = num_joints
        self._physics = physics_engine
        self._mode = JointControlMode.POSITION
        self._targets: list[float] = [0.0] * num_joints
        self._kp: list[float] = [1.0] * num_joints
        self._kd: list[float] = [0.1] * num_joints
        self._effort_limits: list[float] = [100.0] * num_joints
        self._position_limits: list[tuple[float, float]] = [(-3.14, 3.14)] * num_joints
        self._velocity_limits: list[float] = [10.0] * num_joints
        self._joint_states: list[JointState] = [JointState() for _ in range(num_joints)]
        logger.info("JointController created: body=%d joints=%d", body_id, num_joints)

    @property
    def body_id(self) -> int:
        return self._body_id

    @property
    def num_joints(self) -> int:
        return self._num_joints

    @property
    def mode(self) -> JointControlMode:
        return self._mode

    def set_mode(self, mode: JointControlMode) -> None:
        self._mode = mode
        logger.info("Joint control mode set to %s for body %d", mode.value, self._body_id)

    def set_targets(self, targets: list[float], mode: JointControlMode | None = None) -> None:
        if len(targets) != self._num_joints:
            raise ValueError(f"Expected {self._num_joints} targets, got {len(targets)}")
        control_mode = mode or self._mode
        for i, t in enumerate(targets):
            self._targets[i] = self._clamp_target(i, t, control_mode)
        self._apply_targets(control_mode)
        logger.debug("Targets applied: body=%d mode=%s", self._body_id, control_mode.value)

    def set_target(self, joint_index: int, value: float, mode: JointControlMode | None = None) -> None:
        if joint_index < 0 or joint_index >= self._num_joints:
            raise IndexError(f"Joint index {joint_index} out of range [0, {self._num_joints})")
        control_mode = mode or self._mode
        self._targets[joint_index] = self._clamp_target(joint_index, value, control_mode)
        self._apply_single_target(joint_index, control_mode)
        logger.debug("Target applied: joint=%d value=%.4f mode=%s", joint_index, value, control_mode.value)

    def set_gains(self, kp: list[float] | None = None, kd: list[float] | None = None) -> None:
        if kp is not None:
            if len(kp) != self._num_joints:
                raise ValueError(f"Expected {self._num_joints} kp values")
            self._kp = list(kp)
        if kd is not None:
            if len(kd) != self._num_joints:
                raise ValueError(f"Expected {self._num_joints} kd values")
            self._kd = list(kd)
        logger.info("Gains updated for body %d", self._body_id)

    def set_position_limits(self, limits: list[tuple[float, float]]) -> None:
        if len(limits) != self._num_joints:
            raise ValueError(f"Expected {self._num_joints} limit pairs")
        self._position_limits = list(limits)

    def set_velocity_limits(self, limits: list[float]) -> None:
        if len(limits) != self._num_joints:
            raise ValueError(f"Expected {self._num_joints} velocity limits")
        self._velocity_limits = list(limits)

    def set_effort_limits(self, limits: list[float]) -> None:
        if len(limits) != self._num_joints:
            raise ValueError(f"Expected {self._num_joints} effort limits")
        self._effort_limits = list(limits)

    def get_targets(self) -> list[float]:
        return list(self._targets)

    def get_joint_states(self) -> list[JointState]:
        if self._physics is not None:
            self._refresh_states()
        return list(self._joint_states)

    def get_joint_positions(self) -> list[float]:
        return [s.position for s in self.get_joint_states()]

    def get_joint_velocities(self) -> list[float]:
        return [s.velocity for s in self.get_joint_states()]

    def get_joint_efforts(self) -> list[float]:
        return [s.effort for s in self.get_joint_states()]

    def _clamp_target(self, joint_index: int, value: float, mode: JointControlMode) -> float:
        if mode == JointControlMode.POSITION:
            lo, hi = self._position_limits[joint_index]
            return max(lo, min(hi, value))
        elif mode == JointControlMode.VELOCITY:
            limit = self._velocity_limits[joint_index]
            return max(-limit, min(limit, value))
        elif mode == JointControlMode.EFFORT:
            limit = self._effort_limits[joint_index]
            return max(-limit, min(limit, value))
        return value

    def _apply_targets(self, mode: JointControlMode) -> None:
        if self._physics is None:
            return
        indices = list(range(self._num_joints))
        try:
            self._physics.apply_joint_action(
                self._body_id,
                indices,
                self._targets,
                mode=mode.value,
            )
        except Exception as e:
            logger.error("Failed to apply joint actions: %s", e)

    def _apply_single_target(self, joint_index: int, mode: JointControlMode) -> None:
        if self._physics is None:
            return
        try:
            self._physics.apply_joint_action(
                self._body_id,
                [joint_index],
                [self._targets[joint_index]],
                mode=mode.value,
            )
        except Exception as e:
            logger.error("Failed to apply joint action for joint %d: %s", joint_index, e)

    def _refresh_states(self) -> None:
        if self._physics is None:
            return
        try:
            state = self._physics.get_body_state(self._body_id)
            for i in range(min(self._num_joints, len(state.joint_positions))):
                self._joint_states[i].position = state.joint_positions[i]
            for i in range(min(self._num_joints, len(state.joint_velocities))):
                self._joint_states[i].velocity = state.joint_velocities[i]
            for i in range(min(self._num_joints, len(state.joint_efforts))):
                self._joint_states[i].effort = state.joint_efforts[i]
        except Exception as e:
            logger.warning("Failed to refresh joint states: %s", e)

    def reset(self) -> None:
        self._targets = [0.0] * self._num_joints
        self._joint_states = [JointState() for _ in range(self._num_joints)]
        logger.info("JointController reset for body %d", self._body_id)
