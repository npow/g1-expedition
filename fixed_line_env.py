"""Gymnasium task for learning fixed-line ascent with a Unitree G1.

The first curriculum stage starts from a prepared, harnessed position.  A
one-way chest ascender prevents downward travel and a hand ascender supports a
real foot-loop reaction at the feet.  The policy must repeatedly crouch, slide
the unloaded hand ascender, and stand in the loop.  All robot motion is MuJoCo
dynamics; the custom force law only models the directional rope hardware that
a rigid contact solver cannot represent.
"""

from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class G1FixedLineEnv(gym.Env):
    """Climb a vertical fixed rope using an alternating ascender cycle."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(
        self,
        model_path: str | None = None,
        frame_skip: int = 10,
        max_episode_steps: int = 650,
        target_ascent: float = 1.50,
        render_mode: str | None = None,
        randomize_reset: bool = True,
        action_filter: float = 0.55,
    ) -> None:
        super().__init__()
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__),
                "assets",
                "unitree_g1",
                "scene_fixed_line.xml",
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

        self._mocap_ids = {
            name: int(
                self.model.body_mocapid[
                    self._id(mujoco.mjtObj.mjOBJ_BODY, name)
                ]
            )
            for name in (
                "harness_visual",
                "chest_ascender_visual",
                "chest_lanyard_visual",
                "hand_ascender_visual",
                "foot_stirrup_visual",
            )
        }

        self._actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            for i in range(self.nu)
        ]
        self._actuator_ids = {
            name: i for i, name in enumerate(self._actuator_names)
        }
        self._actuator_qpos_addresses = self.model.jnt_qposadr[
            self.model.actuator_trnid[:, 0]
        ].astype(np.int32)

        self._crouch_targets = np.asarray([-0.80, 1.60, -0.80])
        self._stand_targets = np.asarray([0.00, 0.05, -0.03])
        self._leg_actuator_ids = np.asarray(
            [
                self._actuator_ids[f"{side}_{joint}_joint"]
                for side in ("left", "right")
                for joint in ("hip_pitch", "knee", "ankle_pitch")
            ],
            dtype=np.int32,
        )
        self._nominal_ctrl = np.zeros(self.nu, dtype=np.float64)
        self._configure_prepared_pose()
        self._arm_actuator_ids = {
            side: np.asarray(
                [
                    self._actuator_ids[f"{side}_{joint}_joint"]
                    for joint in (
                        "shoulder_pitch",
                        "shoulder_roll",
                        "shoulder_yaw",
                        "elbow",
                        "wrist_roll",
                        "wrist_pitch",
                        "wrist_yaw",
                    )
                ],
                dtype=np.int32,
            )
            for side in ("left", "right")
        }
        self._arm_qpos_addresses = {
            side: self._actuator_qpos_addresses[actuator_ids]
            for side, actuator_ids in self._arm_actuator_ids.items()
        }
        self._arm_dof_addresses = {
            side: self.model.jnt_dofadr[
                self.model.actuator_trnid[actuator_ids, 0]
            ].astype(np.int32)
            for side, actuator_ids in self._arm_actuator_ids.items()
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
        self.ice_face_geom_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "ice_face")

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        # Pelvis/torso orientation, body velocity, all actuator joint errors and
        # velocities, ascender geometry/load state, and the previous action.
        self.obs_dim = 9 + 3 + 3 + 2 * self.nu + 9 + self.action_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.rope_x = 0.36
        self.rope_y = 0.0
        # A 1.24 m etrier places the hand ascender at the dominant hand while
        # the climber is standing.  The previous 1.10 m loop made both hands
        # overlap above a waist-high chest device.
        self.foot_loop_length = 1.24
        self.max_hand_height_from_pelvis = 0.60
        self.foot_advance_rate = 0.62
        self.chest_stiffness = 22_000.0
        self.chest_damping = 520.0
        self.foot_stiffness = 18_000.0
        self.foot_damping = 260.0
        self.max_chest_force = 850.0
        self.max_foot_force = 520.0
        self.horizontal_stiffness = 1_250.0
        self.horizontal_damping = 170.0
        self.orientation_stiffness = 180.0
        self.orientation_damping = 28.0

        self._renderer: mujoco.Renderer | None = None
        self._line_enabled = True
        self._foot_ascender_enabled = True
        self._step_count = 0
        self._stable_success_steps = 0
        self._chest_loaded_steps = 0
        self._foot_loaded_steps = 0
        self._completed_cycles = 0
        self._start_z = 0.0
        self._previous_z = 0.0
        self._high_water_z = 0.0
        self._chest_ratchet_z = 0.0
        self._foot_support_z = 0.0
        self._cycle_start_z = 0.0
        self._cycle_support_z = 0.0
        # 0: waiting for an unloaded advance, 1: waiting for a loaded stand,
        # 2: waiting for the foot loop to unload before another cycle.
        self._cycle_stage = 0
        self._last_action = np.zeros(self.action_dim, dtype=np.float64)
        self._last_chest_force = 0.0
        self._last_foot_force = 0.0
        self._last_support_advance = 0.0
        self._wrist_target_quaternions = {
            "left": np.asarray([1.0, 0.0, 0.0, 0.0]),
            "right": np.asarray([1.0, 0.0, 0.0, 0.0]),
        }

    def _id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return object_id

    def _set_named_targets(self, values: dict[str, float]) -> None:
        for name, value in values.items():
            self._nominal_ctrl[self._actuator_ids[name]] = value

    def _configure_prepared_pose(self) -> None:
        """Set an ergonomic two-level rope grasp and a long-leg start stance."""
        # The dominant (right) hand closes around the high hand-ascender
        # handle.  The left hand is deliberately lower and crosses the fixed
        # line at the chest cam, as it would while tending rope.  These targets
        # were fitted against the articulated distal digit frames; a symmetric
        # pose made the two hands occupy the same volume.
        arms = {
            "left": {
                "shoulder_pitch": -0.6587,
                "shoulder_roll": 0.1770,
                "shoulder_yaw": -0.5470,
                "elbow": 0.1238,
                "wrist_roll": 0.0003,
                "wrist_pitch": 0.0794,
                "wrist_yaw": 0.0,
            },
            "right": {
                "shoulder_pitch": -1.3064,
                "shoulder_roll": 0.0241,
                "shoulder_yaw": 0.0002,
                "elbow": 0.3283,
                "wrist_roll": -0.0003,
                "wrist_pitch": 0.0794,
                "wrist_yaw": 0.0,
            },
        }
        for side, arm in arms.items():
            self._set_named_targets(
                {
                    f"{side}_{joint}_joint": value
                    for joint, value in arm.items()
                }
            )

        self._set_named_targets(
            {
                "left_hand_thumb_0_joint": 0.10,
                "left_hand_thumb_1_joint": 0.70,
                "left_hand_thumb_2_joint": 0.85,
                "left_hand_middle_0_joint": -0.85,
                "left_hand_middle_1_joint": -1.10,
                "left_hand_index_0_joint": -0.85,
                "left_hand_index_1_joint": -1.10,
                "right_hand_thumb_0_joint": -0.10,
                "right_hand_thumb_1_joint": -0.70,
                "right_hand_thumb_2_joint": -0.85,
                "right_hand_middle_0_joint": 0.85,
                "right_hand_middle_1_joint": 1.10,
                "right_hand_index_0_joint": 0.85,
                "right_hand_index_1_joint": 1.10,
            }
        )
        # Reset with the legs long.  The first useful action is therefore a
        # crouch that lifts and unloads the stirrup before it is advanced.
        self._set_leg_extension(1.0, self._nominal_ctrl)

    def _set_leg_extension(self, extension: float, target: np.ndarray) -> None:
        extension = float(np.clip(extension, 0.0, 1.0))
        values = self._crouch_targets + extension * (
            self._stand_targets - self._crouch_targets
        )
        for side in ("left", "right"):
            for joint, value in zip(
                ("hip_pitch", "knee", "ankle_pitch"), values
            ):
                target[self._actuator_ids[f"{side}_{joint}_joint"]] = value

    def _feet(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.data.site_xpos[self.left_foot_site_id].copy(),
            self.data.site_xpos[self.right_foot_site_id].copy(),
        )

    def _site_velocity(self, site_id: int) -> np.ndarray:
        jacobian = np.zeros((3, self.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacobian, None, site_id)
        return jacobian @ self.data.qvel

    def _apply_force_at_site(
        self, force: np.ndarray, site_id: int, body_id: int
    ) -> None:
        mujoco.mj_applyFT(
            self.model,
            self.data,
            force,
            np.zeros(3, dtype=np.float64),
            self.data.site_xpos[site_id],
            body_id,
            self.data.qfrc_applied,
        )

    def _apply_fixed_line_forces(self) -> tuple[float, float]:
        self.data.qfrc_applied.fill(0.0)
        if not self._line_enabled:
            return 0.0, 0.0

        pelvis_position = self.data.xpos[self.pelvis_body_id]
        pelvis_velocity = self.data.qvel[:3]
        pelvis_z = float(pelvis_position[2])
        if pelvis_z > self._chest_ratchet_z:
            self._chest_ratchet_z = pelvis_z
        self._high_water_z = max(self._high_water_z, pelvis_z)

        chest_compression = max(self._chest_ratchet_z - pelvis_z, 0.0)
        chest_force = float(
            np.clip(
                self.chest_stiffness * chest_compression
                - self.chest_damping * min(float(pelvis_velocity[2]), 0.0),
                0.0,
                self.max_chest_force,
            )
        )

        horizontal_force = np.asarray(
            [
                -self.horizontal_stiffness * float(pelvis_position[0])
                - self.horizontal_damping * float(pelvis_velocity[0]),
                -self.horizontal_stiffness * float(pelvis_position[1])
                - self.horizontal_damping * float(pelvis_velocity[1]),
                chest_force,
            ],
            dtype=np.float64,
        )
        base_quaternion = self.data.qpos[3:7]
        quaternion_sign = 1.0 if base_quaternion[0] >= 0.0 else -1.0
        rotation_error = 2.0 * quaternion_sign * base_quaternion[1:4]
        stabilizing_torque = (
            -self.orientation_stiffness * rotation_error
            - self.orientation_damping * self.data.qvel[3:6]
        )
        mujoco.mj_applyFT(
            self.model,
            self.data,
            horizontal_force,
            stabilizing_torque,
            pelvis_position,
            self.pelvis_body_id,
            self.data.qfrc_applied,
        )

        total_foot_force = 0.0
        if self._foot_ascender_enabled:
            for site_id in (self.left_foot_site_id, self.right_foot_site_id):
                foot_z = float(self.data.site_xpos[site_id, 2])
                foot_vz = float(self._site_velocity(site_id)[2])
                compression = max(self._foot_support_z - foot_z, 0.0)
                force = float(
                    np.clip(
                        self.foot_stiffness * compression
                        - self.foot_damping * min(foot_vz, 0.0),
                        0.0,
                        self.max_foot_force / 2.0,
                    )
                )
                if force > 0.0:
                    body_id = int(self.model.site_bodyid[site_id])
                    self._apply_force_at_site(
                        np.asarray([0.0, 0.0, force]), site_id, body_id
                    )
                total_foot_force += force
        return chest_force, total_foot_force

    def _advance_foot_support(self, advance_command: float) -> float:
        if not (self._line_enabled and self._foot_ascender_enabled):
            return 0.0
        left_foot, right_foot = self._feet()
        mean_foot_z = 0.5 * float(left_foot[2] + right_foot[2])
        pelvis_z = float(self.data.xpos[self.pelvis_body_id, 2])
        foot_slack = mean_foot_z - self._foot_support_z - 0.004
        hand_reach_slack = (
            pelvis_z
            + self.max_hand_height_from_pelvis
            - self.foot_loop_length
            - self._foot_support_z
        )
        available = max(min(foot_slack, hand_reach_slack), 0.0)
        requested = (
            self.foot_advance_rate
            * self.policy_dt
            * max(float(advance_command), 0.0)
        )
        advance = min(available, requested)
        self._foot_support_z += advance
        return float(advance)

    def _solve_arm_ik(
        self,
        side: str,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
    ) -> np.ndarray:
        """Fit a seven-DoF arm to fixed hardware in world coordinates.

        Solving both wrist position and orientation prevents a torso pitch or
        leg stroke from visually peeling the fingers off the rope.  This is a
        low-level grasp-retention controller; PPO still chooses the leg and
        ascender cycle.
        """
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
        for _ in range(7):
            mujoco.mj_forward(self.model, ik_data)
            mujoco.mju_subQuat(
                orientation_error,
                target_quaternion,
                ik_data.xquat[wrist_body_id],
            )
            error = np.concatenate(
                [
                    target_position - ik_data.xpos[wrist_body_id],
                    0.55 * orientation_error,
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
                [jacobian_position[:, dof_addresses], jacobian_rotation[:, dof_addresses]]
            )
            damping = 2e-3
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(6), error
            )
            delta = np.clip(delta, -0.18, 0.18)
            ik_data.qpos[qpos_addresses] = np.clip(
                ik_data.qpos[qpos_addresses] + delta,
                lower,
                upper,
            )
        return ik_data.qpos[qpos_addresses].copy()

    def _sync_equipment_visuals(self) -> None:
        pelvis = self.data.xpos[self.pelvis_body_id]
        chest_z = self._chest_ratchet_z + 0.28
        hand_z = self._foot_support_z + self.foot_loop_length
        positions = {
            "harness_visual": np.asarray([pelvis[0], pelvis[1], pelvis[2] - 0.08]),
            "chest_ascender_visual": np.asarray([self.rope_x, self.rope_y, chest_z]),
            "chest_lanyard_visual": np.asarray(
                [pelvis[0], self.rope_y, pelvis[2] - 0.02]
            ),
            "hand_ascender_visual": np.asarray([self.rope_x, self.rope_y, hand_z]),
            "foot_stirrup_visual": np.asarray([0.10, self.rope_y, self._foot_support_z]),
        }
        for name, position in positions.items():
            self.data.mocap_pos[self._mocap_ids[name]] = position

    def _metrics(self) -> dict[str, float]:
        pelvis_position = self.data.xpos[self.pelvis_body_id]
        torso_rotation = self.data.xmat[self.torso_body_id].reshape(3, 3)
        left_foot, right_foot = self._feet()
        mean_foot_z = 0.5 * float(left_foot[2] + right_foot[2])
        left_wrist = self.data.xpos[self.left_wrist_body_id]
        right_wrist = self.data.xpos[self.right_wrist_body_id]
        wrist_midpoint = 0.5 * (left_wrist + right_wrist)
        left_digits = self.data.xpos[self._distal_digit_body_ids["left"]]
        right_digits = self.data.xpos[self._distal_digit_body_ids["right"]]
        left_digit_center = np.mean(left_digits, axis=0)
        right_digit_center = np.mean(right_digits, axis=0)
        chest_device_z = self._chest_ratchet_z + 0.28
        hand_device_z = self._foot_support_z + self.foot_loop_length
        left_grip_error = float(
            np.linalg.norm(
                left_digit_center[[0, 2]]
                - np.asarray([self.rope_x - 0.008, chest_device_z - 0.068])
            )
        )
        right_grip_error = float(
            np.linalg.norm(
                right_digit_center[[0, 2]]
                - np.asarray([self.rope_x + 0.008, hand_device_z])
            )
        )
        left_wrap = float(
            (left_wrist[1] - self.rope_y)
            * (left_digit_center[1] - self.rope_y)
            < 0.0
        )
        right_wrap = float(
            (right_wrist[1] - self.rope_y)
            * (right_digit_center[1] - self.rope_y)
            < 0.0
        )
        grasp_score = float(
            0.25 * left_wrap
            + 0.25 * right_wrap
            + 0.25 * np.exp(-np.square(left_grip_error / 0.075))
            + 0.25 * np.exp(-np.square(right_grip_error / 0.075))
        )
        cross_hand_collision = False
        wall_hand_collision = False
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            name1 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body1
            ) or ""
            name2 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body2
            ) or ""
            cross_hand_collision = cross_hand_collision or (
                name1.startswith("left_hand_")
                and name2.startswith("right_hand_")
            ) or (
                name1.startswith("right_hand_")
                and name2.startswith("left_hand_")
            )
            if self.ice_face_geom_id in (contact.geom1, contact.geom2):
                other_name = name2 if contact.geom1 == self.ice_face_geom_id else name1
                wall_hand_collision = wall_hand_collision or other_name.startswith(
                    ("left_hand_", "right_hand_")
                )
        ascent = float(pelvis_position[2] - self._start_z)
        return {
            "ascent": ascent,
            "height_m": ascent,
            "vertical_speed": float(self.data.qvel[2]),
            "high_water_ascent": float(self._high_water_z - self._start_z),
            "descent_from_high_water": float(self._high_water_z - pelvis_position[2]),
            "upright_score": float(torso_rotation[2, 2]),
            "rope_offset": float(np.linalg.norm(pelvis_position[:2])),
            "mean_foot_z": mean_foot_z,
            "foot_support_gap": float(mean_foot_z - self._foot_support_z),
            "chest_ratchet_z": float(self._chest_ratchet_z),
            "foot_support_z": float(self._foot_support_z),
            "hand_ascender_z": float(self._foot_support_z + self.foot_loop_length),
            "hand_rope_distance": float(
                np.linalg.norm(wrist_midpoint[:2] - [self.rope_x, self.rope_y])
            ),
            "left_grip_error": left_grip_error,
            "right_grip_error": right_grip_error,
            "left_wrap_score": left_wrap,
            "right_wrap_score": right_wrap,
            "grasp_score": grasp_score,
            "hand_separation": float(np.linalg.norm(left_wrist - right_wrist)),
            "cross_hand_collision": float(cross_hand_collision),
            "wall_hand_collision": float(wall_hand_collision),
            "chest_load_n": self._last_chest_force,
            "foot_loop_load_n": self._last_foot_force,
            "completed_cycles": float(self._completed_cycles),
        }

    def _get_obs(self) -> np.ndarray:
        metrics = self._metrics()
        rotation = self.data.xmat[self.pelvis_body_id].reshape(3, 3)
        joint_errors = (
            self.data.qpos[self._actuator_qpos_addresses] - self.data.ctrl
        )
        equipment = np.asarray(
            [
                metrics["ascent"] / max(self.target_ascent, 1e-6),
                metrics["vertical_speed"] * 0.25,
                metrics["descent_from_high_water"] * 5.0,
                metrics["foot_support_gap"] * 5.0,
                (metrics["chest_ratchet_z"] - self.data.qpos[2]) * 5.0,
                metrics["chest_load_n"] / 350.0,
                metrics["foot_loop_load_n"] / 350.0,
                metrics["upright_score"],
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
                equipment,
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
        mujoco.mj_resetData(self.model, self.data)

        start_z = float(options.get("start_z", 1.02))
        lateral = (
            float(self.np_random.uniform(-0.015, 0.015)) if randomized else 0.0
        )
        self.data.qpos[:3] = [0.0, lateral, start_z]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.ctrl[:] = self._nominal_ctrl
        self.data.qpos[self._actuator_qpos_addresses] = self._nominal_ctrl
        if randomized:
            self.data.qpos[self._actuator_qpos_addresses] += self.np_random.normal(
                0.0, 0.006, size=self.nu
            )
            self.data.qvel[:] = self.np_random.normal(0.0, 0.008, size=self.nv)
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._stable_success_steps = 0
        self._chest_loaded_steps = 0
        self._foot_loaded_steps = 0
        self._completed_cycles = 0
        self._start_z = float(self.data.xpos[self.pelvis_body_id, 2])
        self._previous_z = self._start_z
        self._high_water_z = self._start_z
        self._chest_ratchet_z = self._start_z
        left_foot, right_foot = self._feet()
        self._foot_support_z = 0.5 * float(left_foot[2] + right_foot[2]) - 0.004
        self._cycle_start_z = self._start_z
        self._cycle_support_z = self._foot_support_z
        self._cycle_stage = 0
        self._last_action.fill(0.0)
        self._last_chest_force = 0.0
        self._last_foot_force = 0.0
        self._last_support_advance = 0.0
        self._wrist_target_quaternions = {
            "left": self.data.xquat[self.left_wrist_body_id].copy(),
            "right": self.data.xquat[self.right_wrist_body_id].copy(),
        }
        self._sync_equipment_visuals()
        mujoco.mj_forward(self.model, self.data)
        metrics = self._metrics()
        metrics.update(
            {
                "success": False,
                "line_enabled": self._line_enabled,
                "foot_ascender_enabled": self._foot_ascender_enabled,
            }
        )
        return self._get_obs(), metrics

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        command = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        filtered_action = (
            command
            if self._step_count == 1
            else self.action_filter * self._last_action
            + (1.0 - self.action_filter) * command
        )
        support_advance = self._advance_foot_support(filtered_action[1])
        target = self._nominal_ctrl.copy()
        extension = 0.5 * (filtered_action[0] + 1.0)
        self._set_leg_extension(extension, target)
        chest_device_z = self._chest_ratchet_z + 0.28
        hand_device_z = self._foot_support_z + self.foot_loop_length
        arm_targets = {
            "left": np.asarray(
                [self.rope_x - 0.09, self.rope_y + 0.045, chest_device_z - 0.08]
            ),
            "right": np.asarray(
                [self.rope_x - 0.06, self.rope_y - 0.040, hand_device_z - 0.090]
            ),
        }
        for side, position in arm_targets.items():
            target[self._arm_actuator_ids[side]] = self._solve_arm_ik(
                side, position, self._wrist_target_quaternions[side]
            )
        target = np.clip(
            target,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )
        self.data.ctrl[:] = target

        chest_force_peak = 0.0
        foot_force_peak = 0.0
        for _ in range(self.frame_skip):
            chest_force, foot_force = self._apply_fixed_line_forces()
            chest_force_peak = max(chest_force_peak, chest_force)
            foot_force_peak = max(foot_force_peak, foot_force)
            mujoco.mj_step(self.model, self.data)
        self._last_chest_force = chest_force_peak
        self._last_foot_force = foot_force_peak
        self._last_support_advance = support_advance
        self._sync_equipment_visuals()
        mujoco.mj_forward(self.model, self.data)

        pelvis_z = float(self.data.xpos[self.pelvis_body_id, 2])
        self._high_water_z = max(self._high_water_z, pelvis_z)
        if pelvis_z > self._chest_ratchet_z:
            self._chest_ratchet_z = pelvis_z
        ascent_delta = pelvis_z - self._previous_z
        metrics = self._metrics()
        if chest_force_peak > 50.0:
            self._chest_loaded_steps += 1
        if foot_force_peak > 50.0:
            self._foot_loaded_steps += 1

        if (
            self._cycle_stage == 0
            and self._foot_support_z - self._cycle_support_z >= 0.075
            and foot_force_peak < 50.0
        ):
            self._cycle_stage = 1
            self._cycle_start_z = pelvis_z
        elif (
            self._cycle_stage == 1
            and pelvis_z - self._cycle_start_z >= 0.10
            and foot_force_peak > 50.0
        ):
            self._completed_cycles += 1
            self._cycle_stage = 2
        elif self._cycle_stage == 2 and foot_force_peak < 20.0:
            self._cycle_stage = 0
            self._cycle_support_z = self._foot_support_z

        descent = max(-ascent_delta, 0.0)
        upward_progress = max(ascent_delta, 0.0)
        action_cost = float(np.mean(np.square(filtered_action)))
        smoothness_cost = float(
            np.mean(np.square(filtered_action - self._last_action))
        )
        reward = (
            42.0 * upward_progress
            - 34.0 * descent
            + 7.0 * support_advance
            + 0.012 * float(chest_force_peak > 50.0)
            + 0.018 * float(foot_force_peak > 50.0)
            + 0.015 * float(np.clip(metrics["upright_score"], 0.0, 1.0))
            + 0.020 * metrics["grasp_score"]
            - 0.003 * action_cost
            - 0.006 * smoothness_cost
        )

        success_candidate = (
            metrics["ascent"] >= self.target_ascent
            and self._completed_cycles >= 7
            and metrics["descent_from_high_water"] < 0.08
            and metrics["upright_score"] > 0.82
            and metrics["rope_offset"] < 0.20
            and metrics["grasp_score"] > 0.75
            and metrics["hand_separation"] > 0.12
            and metrics["cross_hand_collision"] < 0.5
            and metrics["wall_hand_collision"] < 0.5
            and self._chest_loaded_steps >= 5
            and self._foot_loaded_steps >= 5
        )
        self._stable_success_steps = (
            self._stable_success_steps + 1 if success_candidate else 0
        )
        success = self._stable_success_steps >= 8
        failure = bool(
            pelvis_z < self._start_z - 0.55
            or metrics["rope_offset"] > 0.65
            or metrics["upright_score"] < 0.0
            or not np.isfinite(self.data.qpos).all()
        )
        terminated = success or failure
        truncated = self._step_count >= self.max_episode_steps
        if success:
            reward += 100.0
        if failure:
            reward -= 25.0

        metrics.update(
            {
                "success": success,
                "failure": failure,
                "stable_success_steps": self._stable_success_steps,
                "completed_cycles": self._completed_cycles,
                "chest_load_fraction": self._chest_loaded_steps / self._step_count,
                "foot_loop_load_fraction": self._foot_loaded_steps / self._step_count,
                "support_advance": support_advance,
                "line_enabled": self._line_enabled,
                "foot_ascender_enabled": self._foot_ascender_enabled,
            }
        )
        self._previous_z = pelvis_z
        self._last_action = filtered_action.copy()
        return self._get_obs(), float(reward), terminated, truncated, metrics

    def set_line_enabled(self, enabled: bool) -> None:
        """Enable or disable all fixed-line forces for a causal ablation."""
        self._line_enabled = bool(enabled)

    def set_foot_ascender_enabled(self, enabled: bool) -> None:
        """Enable or disable the hand-ascender/foot-loop reaction only."""
        self._foot_ascender_enabled = bool(enabled)

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=720, width=1280
            )
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = self.pelvis_body_id
        camera.distance = 2.6
        camera.azimuth = 105
        camera.elevation = -8
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
