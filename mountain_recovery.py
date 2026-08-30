"""Slope-specific fall arrest and learned recovery for fixed-line travel.

The recovery policy is a residual on top of the pinned, pretrained WBC
get-up prior.  The prior supplies a useful whole-body motion vocabulary; the
residual is trained in this repository on the actual 28-degree fixed-line
MuJoCo scene.  During the fall and recovery only actuator torques, contacts,
gravity, and the force-balanced one-way safety line affect the robot.  The
floating base is never prescribed after the episode reset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import onnxruntime as ort
from gymnasium import spaces

from fixed_line_slope_env import G1FixedLineEnv
from getup_controller import (
    ACTION_SCALE,
    DEFAULT_JOINT_POS,
    FLOOR_READY_KD,
    FLOOR_READY_KP,
    MOTOR_TORQUE_LIMIT,
    WBC_KD,
    WBC_KP,
    _quat_inverse_rotate,
)
from scripts.fetch_getup_assets import ensure_getup_assets


POLICY_DT = 0.02
SUBSTEPS = 10
RECOVERY_ACTION_DIM = 4

WBC_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def _ort_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    # One inference thread per MuJoCo worker avoids severe oversubscription in
    # the 16-32 process cloud training configuration.
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


class FixedLineRecoveryController:
    """Operate fall, arrest, WBC-residual recovery, and rope re-grasp phases."""

    def __init__(
        self,
        env: G1FixedLineEnv,
        *,
        fetch_assets: bool = True,
    ) -> None:
        self.env = env
        self.model = env.model
        self.data = env.data
        if not np.isclose(self.model.opt.timestep, 0.002):
            raise ValueError("Mountain recovery expects a 0.002 s MuJoCo step")
        assets = (
            ensure_getup_assets()
            if fetch_assets
            else Path(__file__).resolve().parent / "third_party" / "wbc_g1_getup"
        )
        motion = np.load(assets / "getup_01.npz", allow_pickle=True)
        self.ref_joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64)
        self.ref_body_pos = np.asarray(motion["body_pos_w"], dtype=np.float64)
        self.ref_body_quat = np.asarray(motion["body_quat_w"], dtype=np.float64)
        self.ref_body_lin_vel = np.asarray(
            motion["body_lin_vel_w"], dtype=np.float64
        )
        self.ref_body_ang_vel = np.asarray(
            motion["body_ang_vel_w"], dtype=np.float64
        )
        self.session = _ort_session(assets / "policy.onnx")

        self.actuator_ids = np.asarray(
            [env._actuator_ids[name] for name in WBC_JOINT_NAMES], dtype=np.int32
        )
        self.qpos_addresses = env._actuator_qpos_addresses[self.actuator_ids]
        self.dof_addresses = env._actuator_dof_addresses[self.actuator_ids]
        self._stand_gainprm = self.model.actuator_gainprm.copy()
        self._stand_biasprm = self.model.actuator_biasprm.copy()
        self._stand_forcelimited = self.model.actuator_forcelimited.copy()
        self._stand_forcerange = self.model.actuator_forcerange.copy()

        self.previous_wbc_action = np.zeros(29, dtype=np.float64)
        self.previous_residual = np.zeros(RECOVERY_ACTION_DIM, dtype=np.float64)
        self.prepared_joint_pos = env._nominal_ctrl[self.actuator_ids].copy()
        self._blend_groups = (
            np.arange(0, 6, dtype=np.int32),
            np.arange(6, 12, dtype=np.int32),
            np.arange(12, 15, dtype=np.int32),
            np.arange(15, 29, dtype=np.int32),
        )
        self.floor_ready_start = np.zeros(29, dtype=np.float64)
        self.recovery_frame = 0
        self.stable_frames = 0
        self.peak_line_load_n = 0.0
        self.peak_motor_torque_ratio = 0.0
        self.peak_contact_force_n = 0.0
        self.peak_lateral_guide_load_n = 0.0
        self.maximum_contact_penetration_m = 0.0
        self.rope_core_collision_frames = 0
        self.hand_rope_penetration_frames = 0
        self.root_teleports_after_fall_start = 0
        self.recovery_cam_open = False
        self.recovery_lanyard_slack_m = 0.0
        self._contact_force = np.zeros(6, dtype=np.float64)
        self.phase = "climbing"

    @property
    def fall_frames(self) -> int:
        return 150

    @property
    def push_frames(self) -> int:
        return 20

    @property
    def floor_ready_frames(self) -> int:
        return 100

    @property
    def recovery_frames(self) -> int:
        return len(self.ref_joint_pos) + 150

    def _set_pd_gains(self, kp: np.ndarray, kd: np.ndarray) -> None:
        ids = self.actuator_ids
        self.model.actuator_gainprm[ids] = 0.0
        self.model.actuator_biasprm[ids] = 0.0
        self.model.actuator_gainprm[ids, 0] = kp
        self.model.actuator_biasprm[ids, 1] = -kp
        self.model.actuator_biasprm[ids, 2] = -kd

    def _set_passive_damping(self) -> None:
        ids = self.actuator_ids
        self.model.actuator_gainprm[ids] = 0.0
        self.model.actuator_biasprm[ids] = 0.0
        self.model.actuator_biasprm[ids, 2] = -2.0

    def _enforce_motor_limits(self) -> None:
        ids = self.actuator_ids
        self.model.actuator_forcelimited[ids] = 1
        self.model.actuator_forcerange[ids, 0] = -MOTOR_TORQUE_LIMIT
        self.model.actuator_forcerange[ids, 1] = MOTOR_TORQUE_LIMIT

    def restore_climbing_actuators(self) -> None:
        self.model.actuator_gainprm[:] = self._stand_gainprm
        self.model.actuator_biasprm[:] = self._stand_biasprm
        self.model.actuator_forcelimited[:] = self._stand_forcelimited
        self.model.actuator_forcerange[:] = self._stand_forcerange

    def start_fall(self) -> None:
        """Release active pose support and lock the chest safety cam."""

        self.env._swing_side = None
        self.env._swing_phase = 0.0
        self.env._step_cooldown = 0
        self.env._request_hold_steps = 0
        self.env._arm_pull_command = 0.0
        # A one-way cam may take up slack at the instant before a fall, but it
        # may not pull the climber higher afterward.
        self.env._line_ratchet_progress = max(
            self.env._line_ratchet_progress, self.env._progress()
        )
        self.env._hand_ascender_progress = self.env._line_ratchet_progress + 0.08
        self._enforce_motor_limits()
        self._set_passive_damping()
        self.data.ctrl[self.actuator_ids] = self.data.qpos[self.qpos_addresses]
        self.previous_wbc_action.fill(0.0)
        self.previous_residual.fill(0.0)
        self.recovery_frame = 0
        self.stable_frames = 0
        self.peak_line_load_n = 0.0
        self.peak_motor_torque_ratio = 0.0
        self.peak_contact_force_n = 0.0
        self.peak_lateral_guide_load_n = 0.0
        self.maximum_contact_penetration_m = 0.0
        self.rope_core_collision_frames = 0
        self.hand_rope_penetration_frames = 0
        self.root_teleports_after_fall_start = 0
        self.recovery_cam_open = False
        self.recovery_lanyard_slack_m = 0.0
        self.phase = "fall"

    def _apply_safety_line_only(
        self,
        *,
        pelvis_push_n: float = 0.0,
        torso_push_n: float = 0.0,
        lateral_push_n: float = 0.0,
        torso_torque_y_nm: float = 0.0,
        safety_line_active: bool = True,
    ) -> float:
        """Apply only a causal cam reaction plus an optional finite shove."""

        env = self.env
        self.data.qfrc_applied.fill(0.0)
        self.data.xfrc_applied.fill(0.0)
        progress = env._progress()
        uphill_velocity = float(np.dot(self.data.qvel[:3], env.uphill))
        slip = (
            max(
                env._line_ratchet_progress
                - progress
                - self.recovery_lanyard_slack_m,
                0.0,
            )
            if safety_line_active
            else 0.0
        )
        _point, lower, upper, weight = env._rope_sample(
            env._line_ratchet_progress + 0.05
        )
        rope_velocity = float(
            np.dot(env._rope_velocity(lower, upper, weight), env.uphill)
        )
        relative_velocity = uphill_velocity - rope_velocity
        line_force = float(
            np.clip(
                (
                    env.line_stiffness * slip
                    - env.line_damping * min(relative_velocity, 0.0)
                )
                if slip > 0.0
                else 0.0,
                0.0,
                env.max_line_force,
            )
        )
        pelvis_point = self.data.xpos[env.pelvis_body_id]
        if line_force > 0.0:
            force = line_force * env.uphill
            mujoco.mj_applyFT(
                self.model,
                self.data,
                force,
                np.zeros(3, dtype=np.float64),
                pelvis_point,
                env.pelvis_body_id,
                self.data.qfrc_applied,
            )
            env._apply_rope_force(env._line_ratchet_progress + 0.05, -force)
        if pelvis_push_n or lateral_push_n:
            shove = -pelvis_push_n * env.uphill + np.asarray(
                [0.0, lateral_push_n, 0.0]
            )
            mujoco.mj_applyFT(
                self.model,
                self.data,
                shove,
                np.zeros(3, dtype=np.float64),
                pelvis_point,
                env.pelvis_body_id,
                self.data.qfrc_applied,
            )
        if torso_push_n:
            mujoco.mj_applyFT(
                self.model,
                self.data,
                -torso_push_n * env.uphill,
                np.zeros(3, dtype=np.float64),
                self.data.xpos[env.torso_body_id],
                env.torso_body_id,
                self.data.qfrc_applied,
            )
        if torso_torque_y_nm:
            mujoco.mj_applyFT(
                self.model,
                self.data,
                np.zeros(3, dtype=np.float64),
                np.asarray([0.0, torso_torque_y_nm, 0.0]),
                self.data.xpos[env.torso_body_id],
                env.torso_body_id,
                self.data.qfrc_applied,
            )
        if self.recovery_cam_open:
            # The unloaded cam slides along the rope, while its rigid side
            # spacer still keeps the harness on the climber side of the line.
            # This is a transverse, force-balanced constraint, not an uphill
            # recovery force.
            lateral_error = max(-float(pelvis_point[1]), 0.0)
            lateral_force = (
                float(
                    np.clip(
                        1000.0 * lateral_error
                        - 80.0 * min(float(self.data.qvel[1]), 0.0),
                        0.0,
                        300.0,
                    )
                )
                if lateral_error > 0.0
                else 0.0
            )
            if lateral_force > 0.0:
                spacer_force = np.asarray([0.0, lateral_force, 0.0])
                mujoco.mj_applyFT(
                    self.model,
                    self.data,
                    spacer_force,
                    np.zeros(3, dtype=np.float64),
                    pelvis_point,
                    env.pelvis_body_id,
                    self.data.qfrc_applied,
                )
                env._apply_rope_force(env._progress() + 0.05, -spacer_force)
                self.peak_lateral_guide_load_n = max(
                    self.peak_lateral_guide_load_n, lateral_force
                )
        env._last_line_force = line_force
        env._last_rope_guide_force = 0.0
        env._last_arm_pull_force = 0.0
        self.peak_line_load_n = max(self.peak_line_load_n, line_force)
        return line_force

    def _step_physics(
        self,
        *,
        pelvis_push_n: float = 0.0,
        torso_push_n: float = 0.0,
        lateral_push_n: float = 0.0,
        torso_torque_y_nm: float = 0.0,
        safety_line_active: bool | None = None,
    ) -> None:
        if safety_line_active is None:
            safety_line_active = not self.recovery_cam_open
        for _ in range(SUBSTEPS):
            self._apply_safety_line_only(
                pelvis_push_n=pelvis_push_n,
                torso_push_n=torso_push_n,
                lateral_push_n=lateral_push_n,
                torso_torque_y_nm=torso_torque_y_nm,
                safety_line_active=safety_line_active,
            )
            mujoco.mj_step(self.model, self.data)
            ratio = np.abs(self.data.actuator_force[self.actuator_ids]) / np.maximum(
                MOTOR_TORQUE_LIMIT, 1e-6
            )
            self.peak_motor_torque_ratio = max(
                self.peak_motor_torque_ratio, float(np.max(ratio))
            )
            for contact_index in range(self.data.ncon):
                contact = self.data.contact[contact_index]
                self.maximum_contact_penetration_m = max(
                    self.maximum_contact_penetration_m,
                    max(-float(contact.dist), 0.0),
                )
                mujoco.mj_contactForce(
                    self.model,
                    self.data,
                    contact_index,
                    self._contact_force,
                )
                self.peak_contact_force_n = max(
                    self.peak_contact_force_n,
                    abs(float(self._contact_force[0])),
                )
        self.env._sync_equipment_visuals()
        mujoco.mj_forward(self.model, self.data)
        metrics = self.env._metrics()
        self.rope_core_collision_frames += int(
            metrics["rope_core_collision"] > 0.5
        )
        self.hand_rope_penetration_frames += int(
            metrics["hand_rope_max_penetration_m"] > 8e-4
        )

    def step_fall(
        self,
        frame: int,
        *,
        lateral_bias_n: float = 0.0,
    ) -> dict[str, Any]:
        if self.phase != "fall":
            raise RuntimeError(f"Expected fall phase, got {self.phase}")
        pushing = frame < self.push_frames
        self._step_physics(
            # A crampon slip plus finite angular perturbation produces a
            # repeatable backward loss of balance without setting root pose or
            # velocity.  The line still arrests translation through its
            # equal-and-opposite tension path.
            pelvis_push_n=100.0 if pushing else 0.0,
            lateral_push_n=lateral_bias_n if pushing else 0.0,
            # 86 N m is enough to break double support and produce a visible
            # torso impact, but does not spin the harness through the line or
            # manufacture an adversarial upside-down pose.  This is a finite
            # 0.4 s disturbance; the free base remains fully simulated.
            torso_torque_y_nm=86.0 if pushing else 0.0,
        )
        if frame + 1 >= self.fall_frames:
            self.floor_ready_start[:] = self.data.qpos[self.qpos_addresses]
            self._set_pd_gains(FLOOR_READY_KP, FLOOR_READY_KD)
            self.phase = "floor_ready"
        return self.telemetry()

    def step_floor_ready(self, frame: int) -> dict[str, Any]:
        if self.phase != "floor_ready":
            raise RuntimeError(f"Expected floor-ready phase, got {self.phase}")
        blend = min((frame + 1) / self.floor_ready_frames, 1.0)
        self.data.ctrl[self.actuator_ids] = (
            (1.0 - blend) * self.floor_ready_start
            + blend * self.ref_joint_pos[0]
        )
        self._step_physics()
        if frame + 1 >= self.floor_ready_frames:
            self.start_recovery()
        return self.telemetry()

    def start_recovery(self) -> None:
        self._enforce_motor_limits()
        self._set_pd_gains(WBC_KP, WBC_KD)
        self.previous_wbc_action.fill(0.0)
        self.previous_residual.fill(0.0)
        self.recovery_frame = 0
        self.stable_frames = 0
        # Transfer from the taut chest catch to a 1.1 m energy-absorbing
        # lanyard while grounded.  The rope coordinate stays locked: this is
        # explicit slack, not a moved anchor.  It gives the clip room for its
        # hands-and-knees transition, then catches further downslope motion
        # with an equal-and-opposite deformable-rope reaction.
        self.recovery_cam_open = False
        self.recovery_lanyard_slack_m = 1.1
        self.env._hand_ascender_progress = self.env._progress() + 0.08
        self.phase = "recovery"

    def _wbc_observation(self, reference_frame: int) -> np.ndarray:
        anchor = 15  # torso_link in the upstream motion asset
        reference_quaternion = self.ref_body_quat[reference_frame, anchor]
        reference = np.concatenate(
            (
                [self.ref_body_pos[reference_frame, anchor, 2]],
                _quat_inverse_rotate(
                    reference_quaternion,
                    self.ref_body_lin_vel[reference_frame, anchor],
                ),
                _quat_inverse_rotate(
                    reference_quaternion,
                    self.ref_body_ang_vel[reference_frame, anchor],
                ),
                _quat_inverse_rotate(
                    reference_quaternion, np.asarray([0.0, 0.0, -1.0])
                ),
                self.ref_joint_pos[reference_frame],
            )
        )
        measured = np.concatenate(
            (
                self.data.qvel[3:6],
                _quat_inverse_rotate(
                    self.data.qpos[3:7], np.asarray([0.0, 0.0, -1.0])
                ),
                self.data.qpos[self.qpos_addresses] - DEFAULT_JOINT_POS,
                self.data.qvel[self.dof_addresses],
                self.previous_wbc_action,
            )
        )
        result = np.concatenate((reference, measured)).astype(np.float32)
        if result.shape != (132,):
            raise RuntimeError(f"Unexpected WBC observation shape: {result.shape}")
        return result

    def policy_observation(self) -> np.ndarray:
        reference_frame = min(self.recovery_frame, len(self.ref_joint_pos) - 1)
        base = self._wbc_observation(reference_frame).astype(np.float64)
        metrics = self.env._metrics()
        torso_z = self.data.xmat[self.env.torso_body_id].reshape(3, 3)[:, 2]
        recovery_state = np.concatenate(
            (
                [
                    metrics["pelvis_normal_height"],
                    metrics["line_slip_m"],
                    metrics["line_load_n"] / max(self.env.body_weight, 1e-6),
                ],
                torso_z,
                [
                    metrics["left_boot_contact"],
                    metrics["right_boot_contact"],
                    self.recovery_frame / max(self.recovery_frames, 1),
                ],
                self.previous_residual,
            )
        )
        return np.concatenate((base, recovery_state)).astype(np.float32)

    def step_recovery(self, residual: np.ndarray) -> dict[str, Any]:
        if self.phase != "recovery":
            raise RuntimeError(f"Expected recovery phase, got {self.phase}")
        residual = np.clip(
            np.asarray(residual, dtype=np.float64), -1.0, 1.0
        )
        if residual.shape != (RECOVERY_ACTION_DIM,):
            raise ValueError(
                f"Expected {RECOVERY_ACTION_DIM} recovery actions, got "
                f"{residual.shape}"
            )
        reference_frame = min(self.recovery_frame, len(self.ref_joint_pos) - 1)
        observation = self._wbc_observation(reference_frame)
        wbc_action = self.session.run(
            ["actions"], {"obs": observation[None]}
        )[0][0].astype(np.float64)
        target = self.ref_joint_pos[reference_frame] + ACTION_SCALE * wbc_action
        # PPO controls group-wise braking of the whole-body get-up prior.  A
        # command of one holds the measured joint angle, leaving the physical
        # PD derivative term to dissipate velocity; zero follows the prior.
        # The time gate prevents premature braking before hands and knees have
        # created a viable support polygon.
        handoff_gate = float(
            np.clip((self.recovery_frame - 155) / 20.0, 0.0, 1.0)
        )
        blend_commands = handoff_gate * np.maximum(residual, 0.0)
        measured_joint_pos = self.data.qpos[self.qpos_addresses]
        effective_kd = WBC_KD.copy()
        for blend, group in zip(blend_commands, self._blend_groups):
            target[group] = (
                (1.0 - blend) * target[group]
                + blend * measured_joint_pos[group]
            )
            effective_kd[group] *= 1.0 + 5.0 * blend
        desired_torque = (
            WBC_KP * (target - self.data.qpos[self.qpos_addresses])
            - effective_kd * self.data.qvel[self.dof_addresses]
        )
        clipped_torque = np.clip(
            desired_torque, -MOTOR_TORQUE_LIMIT, MOTOR_TORQUE_LIMIT
        )
        target = self.data.qpos[self.qpos_addresses] + (
            clipped_torque + WBC_KD * self.data.qvel[self.dof_addresses]
        ) / WBC_KP
        self.data.ctrl[self.actuator_ids] = np.clip(
            target,
            self.model.actuator_ctrlrange[self.actuator_ids, 0],
            self.model.actuator_ctrlrange[self.actuator_ids, 1],
        )
        self.previous_wbc_action[:] = wbc_action
        self.previous_residual[:] = residual
        self._step_physics()
        self.recovery_frame += 1
        self._update_stability()
        return self.telemetry()

    def _update_stability(self) -> None:
        metrics = self.env._metrics()
        stable = (
            metrics["pelvis_normal_height"] > 0.60
            and metrics["upright_score"] > 0.90
            and bool(metrics["left_boot_contact"])
            and bool(metrics["right_boot_contact"])
            # This gate marks a viable two-foot transfer into the fixed-line
            # stance controller.  The subsequent physical re-grasp blend must
            # independently settle below the stricter final velocity limits.
            and np.linalg.norm(self.data.qvel[:3]) < 0.90
            and np.linalg.norm(self.data.qvel[3:6]) < 1.00
            and metrics["lateral_offset"] < 0.55
            and self.rope_core_collision_frames == 0
        )
        self.stable_frames = self.stable_frames + 1 if stable else 0

    @property
    def recovered(self) -> bool:
        return self.stable_frames >= 1

    def regrasp_for_climb(self, frames: int = 100) -> None:
        """Blend from the recovered stance to the fixed-line prepared pose."""

        self.start_regrasp(frames)
        while self.phase == "regrasp":
            self.step_regrasp()

    def start_regrasp(self, frames: int = 100) -> None:
        """Initialize the visible post-recovery ascender re-grasp."""

        if not self.recovered:
            raise RuntimeError("Cannot re-grasp before the recovery gate passes")
        self.restore_climbing_actuators()
        self.recovery_cam_open = False
        self.recovery_lanyard_slack_m = 0.0
        self.env._line_ratchet_progress = self.env._progress()
        self._regrasp_start = self.data.qpos[
            self.env._actuator_qpos_addresses
        ].copy()
        self.env._hand_ascender_progress = (
            self.env._progress() + self.env.hand_ascender_reach
        )
        self.env._swing_side = None
        right_target = self.env._solve_arm_ik(
            "right",
            self.env._arm_target_positions()["right"],
            self.env._wrist_target_quaternions["right"],
        )
        self._regrasp_prepared = self.env._nominal_ctrl.copy()
        self._regrasp_prepared[self.env._arm_actuator_ids["right"]] = right_target
        self._regrasp_frames = int(frames)
        self._regrasp_frame = 0
        self.phase = "regrasp"

    def step_regrasp(self) -> dict[str, Any]:
        if self.phase != "regrasp":
            raise RuntimeError(f"Expected regrasp phase, got {self.phase}")
        blend = 0.5 - 0.5 * np.cos(
            np.pi * (self._regrasp_frame + 1) / self._regrasp_frames
        )
        self.data.ctrl[:] = np.clip(
            (1.0 - blend) * self._regrasp_start
            + blend * self._regrasp_prepared,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )
        for _ in range(SUBSTEPS):
            self.env._apply_support_forces()
            mujoco.mj_step(self.model, self.data)
        self.env._sync_equipment_visuals()
        mujoco.mj_forward(self.model, self.data)
        self._regrasp_frame += 1
        if self._regrasp_frame >= self._regrasp_frames:
            self.env._previous_progress = self.env._progress()
            self.env._high_water_progress = max(
                self.env._high_water_progress, self.env._progress()
            )
            self.env._last_action.fill(0.0)
            self.env._arm_pull_command = 0.0
            self.phase = "climbing"
        return self.telemetry()

    def telemetry(self) -> dict[str, Any]:
        metrics = self.env._metrics()
        contacts = self.ground_contact_body_names()
        return {
            "phase": self.phase,
            "recovery_frame": self.recovery_frame,
            "pelvis_normal_height_m": metrics["pelvis_normal_height"],
            "torso_upright": metrics["upright_score"],
            "base_linear_speed_mps": float(np.linalg.norm(self.data.qvel[:3])),
            "base_angular_speed_radps": float(np.linalg.norm(self.data.qvel[3:6])),
            "left_boot_contact": bool(metrics["left_boot_contact"]),
            "right_boot_contact": bool(metrics["right_boot_contact"]),
            "nonfoot_ground_contact": any(
                name not in {"left_ankle_roll_link", "right_ankle_roll_link"}
                for name in contacts
            ),
            "ground_contact_bodies": sorted(contacts),
            "line_load_n": metrics["line_load_n"],
            "peak_line_load_n": self.peak_line_load_n,
            "rope_core_collision_frames": self.rope_core_collision_frames,
            "hand_rope_penetration_frames": self.hand_rope_penetration_frames,
            "maximum_hand_rope_penetration_m": metrics[
                "hand_rope_max_penetration_m"
            ],
            "maximum_contact_penetration_m": self.maximum_contact_penetration_m,
            "peak_contact_force_n": self.peak_contact_force_n,
            "peak_lateral_guide_load_n": self.peak_lateral_guide_load_n,
            "peak_motor_torque_ratio": self.peak_motor_torque_ratio,
            "root_teleports_after_fall_start": self.root_teleports_after_fall_start,
            "policy_inference_active": self.phase == "recovery",
            "recovery_cam_open": self.recovery_cam_open,
            "recovery_lanyard_slack_m": self.recovery_lanyard_slack_m,
            "recovered": self.recovered,
        }

    def ground_contact_body_names(self) -> set[str]:
        names: set[str] = set()
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if self.env.ice_face_geom_id not in (contact.geom1, contact.geom2):
                continue
            other = (
                contact.geom2
                if contact.geom1 == self.env.ice_face_geom_id
                else contact.geom1
            )
            if other < 0:
                continue
            body_id = int(self.model.geom_bodyid[other])
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            if name:
                names.add(name)
        return names

    def close(self) -> None:
        self.session = None


class MountainRecoveryEnv(gym.Env):
    """PPO residual-training task on the real fixed-line mountain model."""

    metadata = {"render_modes": []}

    def __init__(self, rank: int = 0, randomize: bool = True) -> None:
        super().__init__()
        self.rank = int(rank)
        self.randomize = bool(randomize)
        self.fixed_line = G1FixedLineEnv(randomize_reset=False)
        self.controller = FixedLineRecoveryController(self.fixed_line)
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(RECOVERY_ACTION_DIM,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(145,), dtype=np.float32
        )
        self.max_episode_steps = self.controller.recovery_frames
        self._episode_step = 0
        self._previous_score = 0.0
        self._build_grounded_cache()

    def _build_grounded_cache(self) -> None:
        observation, _ = self.fixed_line.reset(
            seed=10_000 + self.rank, options={"randomize": False}
        )
        # Train on falls from multiple real points in the saved climbing PPO's
        # gait, not only the pristine reset pose.  This closes the distribution
        # gap between an isolated get-up clip and the integrated mountain demo.
        climb_checkpoint = (
            Path(__file__).resolve().parent
            / "models"
            / "ppo_fixed_line_slope"
            / "g1_fixed_line_final.zip"
        )
        if climb_checkpoint.exists():
            from stable_baselines3 import PPO

            climbing_policy = PPO.load(climb_checkpoint, device="cpu")
            target_steps = (0, 20, 43, 66, 112, 158)[self.rank % 6]
            for _ in range(target_steps):
                action, _state = climbing_policy.predict(
                    observation, deterministic=True
                )
                observation, _reward, terminated, truncated, _info = (
                    self.fixed_line.step(action)
                )
                if terminated or truncated:
                    break
            # Begin a fall from double support rather than cutting a gait
            # stroke in midair.  The extra wait is still generated by PPO.
            for _ in range(40):
                if self.fixed_line._swing_side is None:
                    break
                action, _state = climbing_policy.predict(
                    observation, deterministic=True
                )
                observation, _reward, terminated, truncated, _info = (
                    self.fixed_line.step(action)
                )
                if terminated or truncated:
                    break
            del climbing_policy
        self.controller.start_fall()
        lateral = ((self.rank % 5) - 2) * 2.0
        for frame in range(self.controller.fall_frames):
            self.controller.step_fall(frame, lateral_bias_n=lateral)
        for frame in range(self.controller.floor_ready_frames):
            self.controller.step_floor_ready(frame)
        state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self._state_spec = state_spec
        self._cached_state = np.empty(
            mujoco.mj_stateSize(self.fixed_line.model, state_spec), dtype=np.float64
        )
        mujoco.mj_getState(
            self.fixed_line.model,
            self.fixed_line.data,
            self._cached_state,
            state_spec,
        )
        self._cached_ctrl = self.fixed_line.data.ctrl.copy()
        self._cached_qacc_warmstart = self.fixed_line.data.qacc_warmstart.copy()
        self._cached_mocap_pos = self.fixed_line.data.mocap_pos.copy()
        self._cached_mocap_quat = self.fixed_line.data.mocap_quat.copy()
        self._cached_line_ratchet = self.fixed_line._line_ratchet_progress
        self._cached_hand_ascender = self.fixed_line._hand_ascender_progress

    def _stand_score(self) -> float:
        telemetry = self.controller.telemetry()
        height = float(
            np.clip((telemetry["pelvis_normal_height_m"] - 0.08) / 0.58, 0.0, 1.0)
        )
        upright = float(np.clip((telemetry["torso_upright"] + 0.1) / 1.0, 0.0, 1.0))
        feet = 0.5 * float(telemetry["left_boot_contact"]) + 0.5 * float(
            telemetry["right_boot_contact"]
        )
        speed = np.exp(
            -0.65 * telemetry["base_linear_speed_mps"] ** 2
            -0.18 * telemetry["base_angular_speed_radps"] ** 2
        )
        # Multiplication prevents a fast ballistic pass through an upright
        # pose from looking like a successful recovery.
        support = 0.25 + 0.75 * feet
        return height * upright * support * speed

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        # mjSTATE_FULLPHYSICS intentionally excludes applied forces and solver
        # warm-start buffers.  Clear the full data object first so a terminal
        # contact impulse from the previous episode cannot leak into the next
        # cached grounded state.
        mujoco.mj_resetData(self.fixed_line.model, self.fixed_line.data)
        mujoco.mj_setState(
            self.fixed_line.model,
            self.fixed_line.data,
            self._cached_state,
            self._state_spec,
        )
        self.fixed_line.data.ctrl[:] = self._cached_ctrl
        self.fixed_line.data.mocap_pos[:] = self._cached_mocap_pos
        self.fixed_line.data.mocap_quat[:] = self._cached_mocap_quat
        self.fixed_line.data.qfrc_applied.fill(0.0)
        self.fixed_line.data.xfrc_applied.fill(0.0)
        self.fixed_line.data.qacc_warmstart[:] = self._cached_qacc_warmstart
        self.fixed_line._line_ratchet_progress = self._cached_line_ratchet
        self.fixed_line._hand_ascender_progress = self._cached_hand_ascender
        if self.randomize:
            self.fixed_line.data.qpos[self.controller.qpos_addresses] += (
                self.np_random.normal(0.0, 0.008, 29)
            )
            self.fixed_line.data.qvel[:6] += self.np_random.normal(0.0, 0.015, 6)
            self.fixed_line.data.qvel[self.controller.dof_addresses] += (
                self.np_random.normal(0.0, 0.015, 29)
            )
            friction_scale = float(self.np_random.uniform(0.85, 1.15))
            self.fixed_line.model.geom_friction[self.fixed_line.ice_face_geom_id] = (
                self.fixed_line._slope_friction * friction_scale
            )
        else:
            self.fixed_line.model.geom_friction[self.fixed_line.ice_face_geom_id] = (
                self.fixed_line._slope_friction
            )
        mujoco.mj_forward(self.fixed_line.model, self.fixed_line.data)
        self.controller.start_recovery()
        self.controller.peak_line_load_n = 0.0
        self.controller.peak_motor_torque_ratio = 0.0
        self.controller.peak_contact_force_n = 0.0
        self.controller.peak_lateral_guide_load_n = 0.0
        self.controller.maximum_contact_penetration_m = 0.0
        self.controller.rope_core_collision_frames = 0
        self.controller.hand_rope_penetration_frames = 0
        self._episode_step = 0
        self._previous_score = self._stand_score()
        return self.controller.policy_observation(), self.controller.telemetry()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        telemetry = self.controller.step_recovery(action)
        self._episode_step += 1
        score = self._stand_score()
        progress_reward = 24.0 * (score - self._previous_score)
        posture_reward = 0.22 * score
        reference_frame = min(
            max(self.controller.recovery_frame - 1, 0),
            len(self.controller.ref_joint_pos) - 1,
        )
        joint_error = self.fixed_line.data.qpos[
            self.controller.qpos_addresses
        ] - self.controller.ref_joint_pos[reference_frame]
        tracking_reward = 0.025 * float(
            np.exp(-2.0 * np.mean(np.square(joint_error)))
        )
        velocity_cost = 0.004 * min(
            telemetry["base_linear_speed_mps"] ** 2
            + 0.20 * telemetry["base_angular_speed_radps"] ** 2,
            50.0,
        )
        action_cost = 0.002 * float(np.mean(np.square(action)))
        line_cost = 0.002 * max(
            telemetry["line_load_n"] / max(self.fixed_line.body_weight, 1e-6) - 0.7,
            0.0,
        )
        collision = telemetry["rope_core_collision_frames"] > 0
        reward = (
            progress_reward
            + posture_reward
            + tracking_reward
            - velocity_cost
            - action_cost
            - line_cost
        )
        if collision:
            reward -= 25.0
        if telemetry["recovered"]:
            reward += 120.0
        finite = bool(
            np.isfinite(self.fixed_line.data.qpos).all()
            and np.isfinite(self.fixed_line.data.qvel).all()
        )
        failed = bool(
            collision
            or not finite
            or self.fixed_line._metrics()["lateral_offset"] > 1.0
        )
        terminated = bool(telemetry["recovered"] or failed)
        truncated = self._episode_step >= self.max_episode_steps
        telemetry.update(
            {
                "success": bool(telemetry["recovered"]),
                "failure": failed,
                "stand_score": score,
                "episode_step": self._episode_step,
            }
        )
        self._previous_score = score
        return (
            self.controller.policy_observation(),
            float(reward),
            terminated,
            truncated,
            telemetry,
        )

    def close(self) -> None:
        self.controller.close()
        self.fixed_line.close()
