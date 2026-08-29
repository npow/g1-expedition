"""Ground-contact fixed-line travel on a 28-degree alpine slope.

The robot walks uphill with alternating boot contacts while a one-way rope
attachment catches downslope slip.  The rope is protection, not a vertical
hoist: actuator-driven leg motion becomes uphill travel only through the
stock boot geoms' physical contact with the inclined MuJoCo surface.
"""

from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class G1FixedLineSlopeEnv(gym.Env):
    """High-level alternating-step policy for inclined fixed-line travel."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(
        self,
        model_path: str | None = None,
        frame_skip: int = 10,
        max_episode_steps: int = 1100,
        target_ascent: float = 1.50,
        render_mode: str | None = None,
        randomize_reset: bool = True,
        action_filter: float = 0.35,
    ) -> None:
        super().__init__()
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__),
                "assets",
                "unitree_g1",
                "scene_fixed_line_slope.xml",
            )
        self.model_path = os.path.abspath(model_path)
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)
        self.frame_skip = int(frame_skip)
        self.max_episode_steps = int(max_episode_steps)
        self.target_ascent = float(target_ascent)
        self.render_mode = render_mode
        self.randomize_reset = bool(randomize_reset)
        self.action_filter = float(action_filter)
        self.policy_dt = self.frame_skip * self.model.opt.timestep
        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.action_dim = 2

        self.slope_angle = np.deg2rad(28.0)
        self.uphill = np.asarray(
            [np.cos(self.slope_angle), 0.0, np.sin(self.slope_angle)],
            dtype=np.float64,
        )
        self.slope_normal = np.asarray(
            [-np.sin(self.slope_angle), 0.0, np.cos(self.slope_angle)],
            dtype=np.float64,
        )
        self.rope_y = -0.10
        # A taut alpine handline sits about knee-to-waist high above the local
        # surface at this pitch; it follows the slope instead of hanging plumb.
        self.rope_normal_offset = 0.55
        half_rope_rotation = 0.5 * (0.5 * np.pi - self.slope_angle)
        self.rope_quaternion = np.asarray(
            [np.cos(half_rope_rotation), 0.0, np.sin(half_rope_rotation), 0.0]
        )

        self.pelvis_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.left_foot_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        self.right_foot_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        self.left_wrist_body_id = self._id(
            mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"
        )
        self.right_wrist_body_id = self._id(
            mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"
        )
        self.ice_face_geom_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "ice_face")
        self._slope_friction = self.model.geom_friction[
            self.ice_face_geom_id
        ].copy()
        self._mocap_ids = {
            name: int(
                self.model.body_mocapid[
                    self._id(mujoco.mjtObj.mjOBJ_BODY, name)
                ]
            )
            for name in (
                "harness_visual",
                "chest_ascender_visual",
                "hand_ascender_visual",
                "chest_lanyard_visual",
                "left_crampon_visual",
                "right_crampon_visual",
            )
        }

        self._actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            for i in range(self.nu)
        ]
        self._actuator_ids = {
            name: index for index, name in enumerate(self._actuator_names)
        }
        self._actuator_qpos_addresses = self.model.jnt_qposadr[
            self.model.actuator_trnid[:, 0]
        ].astype(np.int32)
        self._nominal_ctrl = np.zeros(self.nu, dtype=np.float64)
        self._configure_prepared_pose()

        arm_joints = (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
        self._arm_actuator_ids = {
            side: np.asarray(
                [self._actuator_ids[f"{side}_{joint}_joint"] for joint in arm_joints],
                dtype=np.int32,
            )
            for side in ("left", "right")
        }
        self._arm_qpos_addresses = {
            side: self._actuator_qpos_addresses[actuators]
            for side, actuators in self._arm_actuator_ids.items()
        }
        self._arm_dof_addresses = {
            side: self.model.jnt_dofadr[
                self.model.actuator_trnid[actuators, 0]
            ].astype(np.int32)
            for side, actuators in self._arm_actuator_ids.items()
        }
        self._distal_digit_body_ids = {
            side: np.asarray(
                [
                    self._id(
                        mujoco.mjtObj.mjOBJ_BODY,
                        f"{side}_hand_{digit}_link",
                    )
                    for digit in ("thumb_2", "index_1", "middle_1")
                ],
                dtype=np.int32,
            )
            for side in ("left", "right")
        }
        self._foot_body_ids = {
            side: self._id(
                mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link"
            )
            for side in ("left", "right")
        }

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        # Rotation, base velocity, actuator error/velocity, 10 gait/contact
        # scalars, and the previous bilateral step command.
        self.obs_dim = 9 + 3 + 3 + 2 * self.nu + 10 + self.action_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.total_mass = float(mujoco.mj_getTotalmass(self.model))
        self.body_weight = self.total_mass * 9.81
        self.line_stiffness = 18_000.0
        self.line_damping = 520.0
        self.max_line_force = 700.0
        self.lateral_stiffness = 700.0
        self.lateral_damping = 100.0
        self.orientation_stiffness = 420.0
        self.orientation_damping = 46.0
        self.normal_balance_stiffness = 900.0
        self.normal_balance_damping = 90.0
        self.max_normal_balance_force = 100.0
        self.step_duration = 18
        self.step_request_hold = 4
        self._desired_base_quaternion = np.asarray(
            [np.cos(0.045), 0.0, np.sin(0.045), 0.0]
        )

        self._renderer: mujoco.Renderer | None = None
        self._line_enabled = True
        self._traction_enabled = True
        self._step_count = 0
        self._completed_steps = 0
        self._expected_side = 0
        self._swing_side: int | None = None
        self._swing_phase = 0.0
        self._step_power = 0.0
        self._step_cooldown = 0
        self._request_hold_steps = 0
        self._start_progress = 0.0
        self._previous_progress = 0.0
        self._high_water_progress = 0.0
        self._line_ratchet_progress = 0.0
        self._stable_success_steps = 0
        self._line_loaded_steps = 0
        self._grounded_steps = 0
        self._left_contact_steps = 0
        self._right_contact_steps = 0
        self._double_support_steps = 0
        self._airborne_streak = 0
        self._maximum_airborne_streak = 0
        self._last_action = np.zeros(self.action_dim, dtype=np.float64)
        self._last_line_force = 0.0
        self._last_ground_load = 0.0
        self._wrist_target_quaternions = {
            "left": np.asarray([1.0, 0.0, 0.0, 0.0]),
            "right": np.asarray([1.0, 0.0, 0.0, 0.0]),
        }

    def _id(self, object_type: mujoco.mjtObj, name: str) -> int:
        result = mujoco.mj_name2id(self.model, object_type, name)
        if result < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return result

    def _set_named_targets(self, values: dict[str, float]) -> None:
        for name, value in values.items():
            self._nominal_ctrl[self._actuator_ids[name]] = value

    def _configure_prepared_pose(self) -> None:
        for side in ("left", "right"):
            self._set_named_targets(
                {
                    f"{side}_hip_pitch_joint": -0.18,
                    f"{side}_knee_joint": 0.38,
                    f"{side}_ankle_pitch_joint": -0.52,
                    f"{side}_hip_roll_joint": 0.0,
                    f"{side}_hip_yaw_joint": 0.0,
                    f"{side}_ankle_roll_joint": 0.0,
                }
            )
        arms = {
            # The left arm stays bent and clear for balance; forcing it across
            # the torso to a second rope device looked unlike real handline
            # travel at this pitch.
            "left": (-0.42, 0.34, 0.05, 0.88, 0.0, 0.02, 0.0),
            "right": (-0.95, -0.08, -0.10, 0.62, 0.0, 0.08, 0.0),
        }
        joints = (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
        for side, values in arms.items():
            self._set_named_targets(
                {
                    f"{side}_{joint}_joint": value
                    for joint, value in zip(joints, values)
                }
            )
        self._set_named_targets(
            {
                "left_hand_thumb_0_joint": 0.05,
                "left_hand_thumb_1_joint": 0.22,
                "left_hand_thumb_2_joint": 0.20,
                "left_hand_middle_0_joint": -0.25,
                "left_hand_middle_1_joint": -0.35,
                "left_hand_index_0_joint": -0.25,
                "left_hand_index_1_joint": -0.35,
                "right_hand_thumb_0_joint": -0.10,
                "right_hand_thumb_1_joint": -0.70,
                "right_hand_thumb_2_joint": -0.85,
                "right_hand_middle_0_joint": 0.85,
                "right_hand_middle_1_joint": 1.10,
                "right_hand_index_0_joint": 0.85,
                "right_hand_index_1_joint": 1.10,
            }
        )

    def _rope_point(self, progress: float) -> np.ndarray:
        point = progress * self.uphill + self.rope_normal_offset * self.slope_normal
        point[1] = self.rope_y
        return point

    def _progress(self) -> float:
        return float(np.dot(self.data.xpos[self.pelvis_body_id], self.uphill))

    def _normal_height(self, position: np.ndarray) -> float:
        return float(np.dot(position, self.slope_normal))

    def _foot_contacts(self) -> tuple[bool, bool, float, float]:
        contacts = {"left": False, "right": False}
        loads = {"left": 0.0, "right": 0.0}
        contact_force = np.zeros(6, dtype=np.float64)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self.ice_face_geom_id not in (contact.geom1, contact.geom2):
                continue
            other_geom = (
                contact.geom2
                if contact.geom1 == self.ice_face_geom_id
                else contact.geom1
            )
            other_body = int(self.model.geom_bodyid[other_geom])
            side = next(
                (
                    candidate
                    for candidate, body_id in self._foot_body_ids.items()
                    if other_body == body_id
                ),
                None,
            )
            if side is None:
                continue
            mujoco.mj_contactForce(self.model, self.data, index, contact_force)
            contacts[side] = True
            loads[side] += max(float(contact_force[0]), 0.0)
        return contacts["left"], contacts["right"], loads["left"], loads["right"]

    def _apply_support_forces(self) -> tuple[float, float]:
        self.data.qfrc_applied.fill(0.0)
        pelvis = self.data.xpos[self.pelvis_body_id]
        velocity = self.data.qvel[:3]
        progress = float(np.dot(pelvis, self.uphill))
        uphill_velocity = float(np.dot(velocity, self.uphill))
        line_force = 0.0
        if self._line_enabled:
            self._line_ratchet_progress = max(self._line_ratchet_progress, progress)
            slip = max(self._line_ratchet_progress - progress, 0.0)
            line_force = float(
                np.clip(
                    self.line_stiffness * slip
                    - self.line_damping * min(uphill_velocity, 0.0),
                    0.0,
                    self.max_line_force,
                )
            )
        lateral_force = (
            -self.lateral_stiffness * float(pelvis[1])
            - self.lateral_damping * float(velocity[1])
        )
        support_force = line_force * self.uphill
        support_force[1] += lateral_force
        left_contact, right_contact, _left_load, _right_load = self._foot_contacts()
        normal_height = self._normal_height(pelvis)
        normal_velocity = float(np.dot(velocity, self.slope_normal))
        normal_balance = 0.0
        if left_contact or right_contact:
            normal_balance = float(
                np.clip(
                    self.normal_balance_stiffness * (0.67 - normal_height)
                    - self.normal_balance_damping * normal_velocity,
                    0.0,
                    self.max_normal_balance_force,
                )
            )
            support_force += normal_balance * self.slope_normal
        orientation_error = np.zeros(3, dtype=np.float64)
        mujoco.mju_subQuat(
            orientation_error,
            self._desired_base_quaternion,
            self.data.qpos[3:7],
        )
        torque = (
            self.orientation_stiffness * orientation_error
            - self.orientation_damping * self.data.qvel[3:6]
        )
        mujoco.mj_applyFT(
            self.model,
            self.data,
            support_force,
            torque,
            pelvis,
            self.pelvis_body_id,
            self.data.qfrc_applied,
        )

        # No auxiliary uphill force is applied here. Progress comes from the
        # position-actuated leg stroke resolving through boot/slope contact.
        return line_force, 0.0

    def _start_step_if_requested(self, action: np.ndarray) -> None:
        if not self._traction_enabled:
            self._request_hold_steps = 0
            return
        if self._swing_side is not None or self._step_cooldown > 0:
            self._request_hold_steps = 0
            return
        expected = self._expected_side
        other = 1 - expected
        deliberate_request = (
            action[expected] > 0.22
            and action[expected] - action[other] > 0.20
        )
        self._request_hold_steps = (
            self._request_hold_steps + 1 if deliberate_request else 0
        )
        if self._request_hold_steps >= self.step_request_hold:
            self._swing_side = expected
            self._swing_phase = 0.0
            self._step_power = float(np.clip(0.55 + 0.45 * action[expected], 0.45, 1.0))
            self._request_hold_steps = 0

    def _leg_targets(self, target: np.ndarray) -> None:
        for side_index, side in enumerate(("left", "right")):
            hip, knee, ankle = -0.18, 0.38, -0.52
            if self._swing_side is not None:
                if side_index == self._swing_side:
                    lift = float(np.sin(np.pi * self._swing_phase))
                    hip = -0.18 - 0.52 * lift
                    knee = 0.38 + 1.00 * lift
                    ankle = -0.52 - 0.24 * lift
                else:
                    hip, knee, ankle = -0.08, 0.22, -0.50
            for joint, value in zip(
                ("hip_pitch", "knee", "ankle_pitch"),
                (hip, knee, ankle),
            ):
                target[self._actuator_ids[f"{side}_{joint}_joint"]] = value

    def _solve_arm_ik(
        self,
        side: str,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
    ) -> np.ndarray:
        ik_data = self._ik_data
        ik_data.qpos[:] = self.data.qpos
        actuator_ids = self._arm_actuator_ids[side]
        qpos_addresses = self._arm_qpos_addresses[side]
        dof_addresses = self._arm_dof_addresses[side]
        joint_ids = self.model.actuator_trnid[actuator_ids, 0]
        lower = self.model.jnt_range[joint_ids, 0] + 1e-4
        upper = self.model.jnt_range[joint_ids, 1] - 1e-4
        wrist_body_id = (
            self.left_wrist_body_id if side == "left" else self.right_wrist_body_id
        )
        jacobian_position = np.zeros((3, self.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.nv), dtype=np.float64)
        orientation_error = np.zeros(3, dtype=np.float64)
        for _ in range(8):
            mujoco.mj_forward(self.model, ik_data)
            mujoco.mju_subQuat(
                orientation_error,
                target_quaternion,
                ik_data.xquat[wrist_body_id],
            )
            error = np.concatenate(
                [
                    target_position - ik_data.xpos[wrist_body_id],
                    # Position takes priority so the digits stay on the
                    # inclined hardware; a light orientation term prevents
                    # implausible wrist roll without blocking cross-body reach.
                    0.05 * orientation_error,
                ]
            )
            if float(np.linalg.norm(error)) < 8e-4:
                break
            mujoco.mj_jacBody(
                self.model,
                ik_data,
                jacobian_position,
                jacobian_rotation,
                wrist_body_id,
            )
            jacobian = np.vstack(
                [
                    jacobian_position[:, dof_addresses],
                    jacobian_rotation[:, dof_addresses],
                ]
            )
            damping = 2e-3
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(6), error
            )
            ik_data.qpos[qpos_addresses] = np.clip(
                ik_data.qpos[qpos_addresses] + np.clip(delta, -0.18, 0.18),
                lower,
                upper,
            )
        return ik_data.qpos[qpos_addresses].copy()

    def _device_points(self) -> tuple[np.ndarray, np.ndarray]:
        progress = self._progress()
        return self._rope_point(progress + 0.05), self._rope_point(progress + 0.36)

    def _arm_target_positions(self) -> dict[str, np.ndarray]:
        _chest, hand = self._device_points()
        return {
            "right": hand + 0.050 * self.slope_normal + np.asarray([0.0, -0.040, 0.0]),
        }

    def _sync_equipment_visuals(self) -> None:
        pelvis = self.data.xpos[self.pelvis_body_id]
        chest, hand = self._device_points()
        positions = {
            "harness_visual": pelvis + np.asarray([0.0, 0.0, -0.08]),
            "chest_ascender_visual": chest,
            "hand_ascender_visual": hand,
            "chest_lanyard_visual": pelvis + np.asarray([0.0, 0.0, -0.04]),
        }
        for name, position in positions.items():
            mocap_id = self._mocap_ids[name]
            self.data.mocap_pos[mocap_id] = position
            self.data.mocap_quat[mocap_id] = (
                self.rope_quaternion
                if name in ("chest_ascender_visual", "hand_ascender_visual")
                else np.asarray([1.0, 0.0, 0.0, 0.0])
            )
        for side in ("left", "right"):
            mocap_id = self._mocap_ids[f"{side}_crampon_visual"]
            body_id = self._foot_body_ids[side]
            self.data.mocap_pos[mocap_id] = self.data.xpos[body_id]
            self.data.mocap_quat[mocap_id] = self.data.xquat[body_id]

    def _collision_flags(self) -> tuple[bool, bool]:
        cross_hand = False
        slope_hand = False
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            name1 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body1
            ) or ""
            name2 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body2
            ) or ""
            cross_hand = cross_hand or (
                name1.startswith("left_hand_") and name2.startswith("right_hand_")
            ) or (
                name1.startswith("right_hand_") and name2.startswith("left_hand_")
            )
            if self.ice_face_geom_id in (contact.geom1, contact.geom2):
                other = name2 if contact.geom1 == self.ice_face_geom_id else name1
                slope_hand = slope_hand or other.startswith(("left_hand_", "right_hand_"))
        return cross_hand, slope_hand

    def _metrics(self) -> dict[str, float]:
        pelvis = self.data.xpos[self.pelvis_body_id]
        torso_rotation = self.data.xmat[self.torso_body_id].reshape(3, 3)
        progress = self._progress() - self._start_progress
        left_contact, right_contact, left_load, right_load = self._foot_contacts()
        left_wrist = self.data.xpos[self.left_wrist_body_id]
        right_wrist = self.data.xpos[self.right_wrist_body_id]
        left_digit_positions = self.data.xpos[self._distal_digit_body_ids["left"]]
        right_digit_positions = self.data.xpos[self._distal_digit_body_ids["right"]]
        left_digits = np.mean(left_digit_positions, axis=0)
        right_digits = np.mean(right_digit_positions, axis=0)
        chest, hand = self._device_points()
        left_grip_error = float(np.linalg.norm(left_digits - chest))
        right_grip_error = float(np.linalg.norm(right_digits - hand))
        # A true wrap means articulated distal digits span both sides of the
        # rope/handle centerline. Using their mean would incorrectly reject a
        # visibly wrapped hand whenever the palm and mean lie on one side.
        left_wrap = float(
            np.min(left_digit_positions[:, 1])
            < self.rope_y
            < np.max(left_digit_positions[:, 1])
        )
        right_wrap = float(
            np.min(right_digit_positions[:, 1])
            < self.rope_y
            < np.max(right_digit_positions[:, 1])
        )
        # At this slope angle the handled ascender is operated one-handed. The
        # left hand is intentionally free for balance, so only the right grip
        # is scored as a required equipment contact.
        grasp_score = float(
            0.25 * right_wrap
            + 0.75 * np.exp(-np.square(right_grip_error / 0.15))
        )
        cross_hand, slope_hand = self._collision_flags()
        left_foot = self.data.site_xpos[self.left_foot_site_id]
        right_foot = self.data.site_xpos[self.right_foot_site_id]
        ground_load = left_load + right_load
        return {
            "ascent": progress,
            "height_m": progress,
            "vertical_gain_m": float(pelvis[2] - self._start_pelvis_z),
            "uphill_speed": float(np.dot(self.data.qvel[:3], self.uphill)),
            "high_water_ascent": float(self._high_water_progress - self._start_progress),
            "descent_from_high_water": float(self._high_water_progress - self._progress()),
            "upright_score": float(torso_rotation[2, 2]),
            "lateral_offset": abs(float(pelvis[1])),
            "pelvis_normal_height": self._normal_height(pelvis),
            "left_boot_contact": float(left_contact),
            "right_boot_contact": float(right_contact),
            "any_boot_contact": float(left_contact or right_contact),
            "double_support": float(left_contact and right_contact),
            "left_boot_load_n": left_load,
            "right_boot_load_n": right_load,
            "ground_load_n": ground_load,
            "ground_load_bodyweight": ground_load / max(self.body_weight, 1e-6),
            "left_boot_clearance_m": self._normal_height(left_foot),
            "right_boot_clearance_m": self._normal_height(right_foot),
            "line_load_n": self._last_line_force,
            "line_slip_m": max(self._line_ratchet_progress - self._progress(), 0.0),
            "completed_cycles": float(self._completed_steps),
            "expected_side": float(self._expected_side),
            "swing_side": float(-1 if self._swing_side is None else self._swing_side),
            "swing_phase": self._swing_phase,
            "left_grip_error": left_grip_error,
            "right_grip_error": right_grip_error,
            "left_wrap_score": left_wrap,
            "right_wrap_score": right_wrap,
            "grasp_score": grasp_score,
            "hand_separation": float(np.linalg.norm(left_wrist - right_wrist)),
            "cross_hand_collision": float(cross_hand),
            "wall_hand_collision": float(slope_hand),
        }

    def _get_obs(self) -> np.ndarray:
        metrics = self._metrics()
        rotation = self.data.xmat[self.pelvis_body_id].reshape(3, 3)
        joint_errors = self.data.qpos[self._actuator_qpos_addresses] - self.data.ctrl
        gait = np.asarray(
            [
                metrics["ascent"] / max(self.target_ascent, 1e-6),
                metrics["uphill_speed"] * 0.4,
                metrics["descent_from_high_water"] * 5.0,
                -1.0 if self._expected_side == 0 else 1.0,
                metrics["swing_side"],
                metrics["swing_phase"],
                metrics["left_boot_contact"],
                metrics["right_boot_contact"],
                metrics["ground_load_bodyweight"],
                metrics["completed_cycles"] / 10.0,
            ]
        )
        observation = np.concatenate(
            [
                rotation.ravel(),
                self.data.qvel[3:6] * 0.10,
                self.data.qvel[:3] * 0.20,
                joint_errors,
                self.data.qvel[6:] * 0.05,
                gait,
                self._last_action,
            ]
        )
        return observation.astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        randomized = bool(options.get("randomize", self.randomize_reset))
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        lateral = float(self.np_random.uniform(-0.012, 0.012)) if randomized else 0.0
        self.data.qpos[:3] = [0.0, lateral, 0.84]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self._actuator_qpos_addresses] = self._nominal_ctrl
        self.data.ctrl[:] = self._nominal_ctrl
        if randomized:
            self.data.qpos[self._actuator_qpos_addresses] += self.np_random.normal(
                0.0, 0.004, self.nu
            )
            self.data.qvel[:] = self.np_random.normal(0.0, 0.004, self.nv)
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._completed_steps = 0
        self._expected_side = 0
        self._swing_side = None
        self._swing_phase = 0.0
        self._step_power = 0.0
        self._step_cooldown = 0
        self._request_hold_steps = 0
        initial_progress = self._progress()
        self._start_progress = initial_progress
        self._previous_progress = initial_progress
        self._high_water_progress = initial_progress
        self._line_ratchet_progress = initial_progress
        self._stable_success_steps = 0
        self._line_loaded_steps = 0
        self._grounded_steps = 0
        self._left_contact_steps = 0
        self._right_contact_steps = 0
        self._double_support_steps = 0
        self._airborne_streak = 0
        self._maximum_airborne_streak = 0
        self._last_action.fill(0.0)
        self._last_line_force = 0.0
        self._last_ground_load = 0.0
        self._wrist_target_quaternions = {
            "left": self.data.xquat[self.left_wrist_body_id].copy(),
            "right": self.data.xquat[self.right_wrist_body_id].copy(),
        }
        for side, position in self._arm_target_positions().items():
            arm_targets = self._solve_arm_ik(
                side, position, self._wrist_target_quaternions[side]
            )
            self.data.qpos[self._arm_qpos_addresses[side]] = arm_targets
            self.data.ctrl[self._arm_actuator_ids[side]] = arm_targets
        self._sync_equipment_visuals()
        mujoco.mj_forward(self.model, self.data)

        # Settle into real boot contacts before measuring episode progress.
        for _ in range(160):
            line_force, _drive = self._apply_support_forces()
            self._last_line_force = line_force
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        settled_progress = self._progress()
        self._start_progress = settled_progress
        self._previous_progress = settled_progress
        self._high_water_progress = settled_progress
        self._line_ratchet_progress = settled_progress
        self._start_pelvis_z = float(self.data.xpos[self.pelvis_body_id, 2])
        self._last_line_force = 0.0
        self._sync_equipment_visuals()
        mujoco.mj_forward(self.model, self.data)
        metrics = self._metrics()
        metrics.update(
            {
                "success": False,
                "line_enabled": self._line_enabled,
                "traction_enabled": self._traction_enabled,
            }
        )
        return self._get_obs(), metrics

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        command = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        filtered = (
            command
            if self._step_count == 1
            else self.action_filter * self._last_action
            + (1.0 - self.action_filter) * command
        )
        if self._step_cooldown > 0:
            self._step_cooldown -= 1
        self._start_step_if_requested(filtered)

        target = self._nominal_ctrl.copy()
        self._leg_targets(target)
        for side, position in self._arm_target_positions().items():
            target[self._arm_actuator_ids[side]] = self._solve_arm_ik(
                side, position, self._wrist_target_quaternions[side]
            )
        self.data.ctrl[:] = np.clip(
            target,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )

        line_force_peak = 0.0
        ground_load_peak = 0.0
        for _ in range(self.frame_skip):
            line_force, _drive = self._apply_support_forces()
            line_force_peak = max(line_force_peak, line_force)
            _lc, _rc, left_load, right_load = self._foot_contacts()
            ground_load_peak = max(ground_load_peak, left_load + right_load)
            mujoco.mj_step(self.model, self.data)
        self._last_line_force = line_force_peak
        self._last_ground_load = ground_load_peak

        completed_step = False
        if self._swing_side is not None:
            self._swing_phase += 1.0 / self.step_duration
            if self._swing_phase >= 1.0:
                finished_side = self._swing_side
                self._swing_side = None
                self._swing_phase = 0.0
                self._step_cooldown = 3
                self._expected_side = 1 - finished_side
                self._completed_steps += 1
                completed_step = True

        self._sync_equipment_visuals()
        mujoco.mj_forward(self.model, self.data)
        progress_absolute = self._progress()
        self._high_water_progress = max(self._high_water_progress, progress_absolute)
        progress_delta = progress_absolute - self._previous_progress
        metrics = self._metrics()
        left_contact = bool(metrics["left_boot_contact"])
        right_contact = bool(metrics["right_boot_contact"])
        if left_contact or right_contact:
            self._grounded_steps += 1
            self._airborne_streak = 0
        else:
            self._airborne_streak += 1
        self._maximum_airborne_streak = max(
            self._maximum_airborne_streak, self._airborne_streak
        )
        self._left_contact_steps += int(left_contact)
        self._right_contact_steps += int(right_contact)
        self._double_support_steps += int(left_contact and right_contact)
        self._line_loaded_steps += int(line_force_peak > 40.0)

        action_cost = float(np.mean(np.square(filtered)))
        smoothness_cost = float(np.mean(np.square(filtered - self._last_action)))
        reward = (
            35.0 * max(progress_delta, 0.0)
            - 45.0 * max(-progress_delta, 0.0)
            + 0.70 * float(completed_step)
            + 0.028 * metrics["any_boot_contact"]
            + 0.010 * metrics["double_support"]
            + 0.018 * float(np.clip(metrics["upright_score"], 0.0, 1.0))
            + 0.018 * metrics["grasp_score"]
            - 0.06 * float(not (left_contact or right_contact))
            - 0.003 * action_cost
            - 0.005 * smoothness_cost
        )

        elapsed = max(self._step_count, 1)
        left_contact_fraction = self._left_contact_steps / elapsed
        right_contact_fraction = self._right_contact_steps / elapsed
        grounded_fraction = self._grounded_steps / elapsed
        success_candidate = (
            metrics["ascent"] >= self.target_ascent
            and self._completed_steps >= 8
            and metrics["descent_from_high_water"] < 0.10
            and metrics["upright_score"] > 0.78
            and metrics["lateral_offset"] < 0.25
            and metrics["grasp_score"] > 0.35
            and metrics["hand_separation"] > 0.12
            and metrics["cross_hand_collision"] < 0.5
            and metrics["wall_hand_collision"] < 0.5
            and grounded_fraction > 0.90
            and left_contact_fraction > 0.30
            and right_contact_fraction > 0.30
            and self._maximum_airborne_streak <= 3
            and self._line_loaded_steps >= 5
        )
        self._stable_success_steps = (
            self._stable_success_steps + 1 if success_candidate else 0
        )
        success = self._stable_success_steps >= 8
        failure = bool(
            metrics["ascent"] < -0.55
            or metrics["pelvis_normal_height"] < 0.42
            or metrics["lateral_offset"] > 0.70
            or metrics["upright_score"] < 0.05
            or self._airborne_streak > 20
            or not np.isfinite(self.data.qpos).all()
        )
        terminated = success or failure
        truncated = self._step_count >= self.max_episode_steps
        if success:
            reward += 100.0
        if failure:
            reward -= 30.0

        metrics.update(
            {
                "success": success,
                "failure": failure,
                "stable_success_steps": self._stable_success_steps,
                "completed_cycles": self._completed_steps,
                "grounded_fraction": grounded_fraction,
                "left_boot_contact_fraction": left_contact_fraction,
                "right_boot_contact_fraction": right_contact_fraction,
                "double_support_fraction": self._double_support_steps / elapsed,
                "maximum_airborne_streak": self._maximum_airborne_streak,
                "line_load_fraction": self._line_loaded_steps / elapsed,
                "chest_load_fraction": self._line_loaded_steps / elapsed,
                "foot_loop_load_fraction": grounded_fraction,
                "line_enabled": self._line_enabled,
                "traction_enabled": self._traction_enabled,
                "foot_ascender_enabled": self._traction_enabled,
            }
        )
        self._previous_progress = progress_absolute
        self._last_action = filtered.copy()
        return self._get_obs(), float(reward), terminated, truncated, metrics

    def set_line_enabled(self, enabled: bool) -> None:
        self._line_enabled = bool(enabled)

    def set_traction_enabled(self, enabled: bool) -> None:
        self._traction_enabled = bool(enabled)
        self.model.geom_friction[self.ice_face_geom_id] = self._slope_friction

    def set_foot_ascender_enabled(self, enabled: bool) -> None:
        """Compatibility alias: disables boot-derived uphill traction."""
        self.set_traction_enabled(enabled)

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=720, width=1280)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = self.pelvis_body_id
        camera.distance = 2.8
        camera.azimuth = 118
        camera.elevation = -10
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# The existing training/evaluation entry points import this conventional name.
G1FixedLineEnv = G1FixedLineSlopeEnv
