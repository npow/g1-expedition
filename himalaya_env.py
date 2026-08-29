"""Gymnasium task for learning a Unitree G1 ice-axe self-arrest."""

from __future__ import annotations

from collections import deque
import os
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class G1SelfArrestEnv(gym.Env):
    """Learn to stop a prone downhill slide by loading the ice-axe pick.

    The fingers and support (left) arm are position-controlled in a verified
    two-handed grasp. PPO controls all fourteen arm joints and must learn how
    to engage and load the pick. This hierarchical curriculum stage isolates
    the arrest skill instead of asking one policy to rediscover a 43-DoF grasp,
    whole-body pose, and snow interaction simultaneously.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    LEFT_ARM_IDS = np.arange(15, 22, dtype=np.int32)
    RIGHT_ARM_IDS = np.arange(29, 36, dtype=np.int32)
    POLICY_ACTUATOR_IDS = np.concatenate((LEFT_ARM_IDS, RIGHT_ARM_IDS))
    RIGHT_FINGER_IDS = np.arange(36, 43, dtype=np.int32)
    ACTION_SCALE = np.asarray(
        [
            0.20,
            0.20,
            0.50,
            0.30,
            0.20,
            0.25,
            0.20,
            0.70,
            0.35,
            0.45,
            0.60,
            0.30,
            0.55,
            0.30,
        ]
    )
    AXE_BLADE_VECTOR = np.asarray([0.135, 0.0, -0.085], dtype=np.float64)

    def __init__(
        self,
        model_path: str | None = None,
        frame_skip: int = 5,
        max_episode_steps: int = 700,
        render_mode: str | None = None,
        randomize_reset: bool = True,
        initial_speed_range: tuple[float, float] = (4.0, 5.0),
        heading_range_degrees: tuple[float, float] = (0.0, 0.0),
        lateral_speed_range: tuple[float, float] = (0.0, 0.0),
        roll_range_degrees: tuple[float, float] = (0.0, 0.0),
        anchor_resets: tuple[tuple[float, float, float, float], ...] | None = None,
        anchor_probability: float = 0.0,
        action_filter: float = 0.95,
    ) -> None:
        super().__init__()

        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "assets", "scene_self_arrest.xml")
        self.model_path = os.path.abspath(model_path)
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.randomize_reset = randomize_reset
        self.initial_speed_range = initial_speed_range
        self.heading_range_degrees = heading_range_degrees
        self.lateral_speed_range = lateral_speed_range
        self.roll_range_degrees = roll_range_degrees
        self.anchor_resets = tuple(anchor_resets or ())
        self.anchor_probability = float(anchor_probability)
        self.action_filter = float(action_filter)
        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.action_dim = len(self.POLICY_ACTUATOR_IDS)

        self.pelvis_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.left_wrist_body_id = self._id(
            mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"
        )
        self.right_wrist_body_id = self._id(
            mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"
        )
        self.axe_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "ice_axe")
        self.axe_pick_geom_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "axe_pick_tip")
        self.axe_shaft_geom_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "axe_shaft")
        self.axe_adze_geom_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "axe_adze_tip")
        self._axe_geom_ids = {
            self.axe_pick_geom_id,
            self.axe_shaft_geom_id,
            self.axe_adze_geom_id,
        }
        self.slope_geom_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "snow_mountain_slope")
        self.axe_pick_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "axe_pick_site")
        self.axe_touch_sensor_id = self._id(mujoco.mjtObj.mjOBJ_SENSOR, "axe_pick_touch")
        self.chest_touch_sensor_id = self._id(mujoco.mjtObj.mjOBJ_SENSOR, "chest_touch")

        angle = np.deg2rad(35.0)
        self.slope_normal = np.asarray([-np.sin(angle), 0.0, np.cos(angle)], dtype=np.float64)
        self.slope_downhill = np.asarray([-np.cos(angle), 0.0, -np.sin(angle)], dtype=np.float64)
        self.slope_uphill = -self.slope_downhill

        # The policy begins with the axe visibly raised: the pick is roughly
        # 11 cm clear of the slope and is not in solver contact. Both arms are
        # learned so the ensuing two-handed planting stroke is not supplied by
        # a low-level arm trajectory.
        self.ready_qpos = np.zeros(self.nu, dtype=np.float64)
        self.ready_qpos[3] = 0.55
        self.ready_qpos[4] = -0.55
        self.ready_qpos[9] = 0.55
        self.ready_qpos[10] = -0.55
        self.ready_qpos[14] = -0.20
        self.ready_qpos[15:22] = [
            0.2724142,
            0.7543950,
            0.3611323,
            -0.4373010,
            -0.1707974,
            -1.2211052,
            -0.9872638,
        ]
        self.ready_qpos[22:29] = [
            0.42338478,
            -0.26784778,
            1.70000000,
            # Close the middle and index opposition around the lower shaft.
            # The earlier targets let these two digits roll off during longer
            # strict-contact arrests even though the thumb stayed planted.
            -0.75000000,
            -1.70000000,
            -1.20000000,
            -1.68000000,
        ]
        self.ready_qpos[29:36] = [
            -0.394132,
            0.243729,
            0.067443,
            -0.895466,
            -0.034593,
            -0.858455,
            -0.037823,
        ]
        self.ready_qpos[36:43] = [
            -0.815,
            0.537,
            -0.717,
            1.365,
            1.532,
            1.229,
            0.246,
        ]

        # Kinematically fitted reference for a correctly planted pick. This is
        # used only for diagnostics and curriculum shaping; it is never played
        # back as an arm trajectory. It places the shortened pick at the snow
        # surface with the blade 25.6 degrees into the slope.
        self.engaged_qpos = self.ready_qpos.copy()
        self.engaged_qpos[15:22] = [
            0.3374819,
            0.8272450,
            -0.0841265,
            -0.2702070,
            -0.2293778,
            -1.1468587,
            -1.0932391,
        ]
        self.engaged_qpos[29:36] = [
            0.205868,
            0.243729,
            0.067443,
            -0.895466,
            -0.034593,
            -0.858455,
            -0.037823,
        ]

        # Controls are stored in actuator order, while qpos is stored in joint
        # tree order.  Those orders differ for the right index/middle fingers.
        # Keep an explicit mapping so reset and observation code cannot silently
        # swap those digits again.
        self._actuator_qpos_addresses = self.model.jnt_qposadr[
            self.model.actuator_trnid[:, 0]
        ].astype(np.int32)
        self._joint_target_qpos = np.zeros(self.nu, dtype=np.float64)
        self._joint_target_qpos[
            self._actuator_qpos_addresses - 7
        ] = self.ready_qpos

        # PPO controls both complete arms through bounded joint-position
        # residuals. The fingers remain low-level secure-grasp controllers.
        self.action_scale = self.ACTION_SCALE.copy()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.action_dim,), dtype=np.float32)

        # Base R, angular velocity, slope velocity, all joint errors/velocities,
        # pick/chest/hand force, pick/ventral/stroke geometry, and previous action.
        self.obs_dim = 9 + 3 + 3 + 2 * self.nu + 5 + 5 + self.action_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        self._renderer: mujoco.Renderer | None = None
        self._step_count = 0
        self._stable_arrest_steps = 0
        self._invalid_arrest_steps = 0
        self._pick_contact_steps = 0
        self._left_grasp_contact_steps = 0
        self._right_grasp_contact_steps = 0
        self._last_action = np.zeros(self.action_dim, dtype=np.float64)
        self._last_snow_drag_force = 0.0
        self._pick_contact_active = False
        self._peak_rigid_pick_force = 0.0
        self._rigid_pick_contact_seen = False
        self._rigid_pick_contact_substeps = 0
        self._physics_substeps = 0
        self._contact_substeps_this_step = 0
        self._snow_drag_sum_this_step = 0.0
        self._contact_fraction_window: deque[float] = deque(maxlen=50)
        self._snow_drag_window: deque[float] = deque(maxlen=50)
        self._blade_angle_window: deque[float] = deque(maxlen=50)
        self._initial_pick_torso_position = np.zeros(3, dtype=np.float64)
        self._initial_pick_height = 0.0
        self._first_rigid_contact_step = -1
        self._stroke_at_first_contact = 0.0
        self._lowering_at_first_contact = 0.0
        self._blade_angle_at_first_contact = 0.0
        self._previous_speed = 0.0
        self._initial_speed = 0.0
        self._reset_heading_degrees = 0.0
        self._reset_lateral_speed = 0.0
        self._reset_roll_degrees = 0.0
        self._initial_position = np.zeros(3, dtype=np.float64)
        self._pick_default_contype = int(self.model.geom_contype[self.axe_pick_geom_id])
        self._pick_default_conaffinity = int(self.model.geom_conaffinity[self.axe_pick_geom_id])
        self._pick_enabled = True
        self._body_slope_friction_enabled = True
        # A rigid Coulomb plane cannot represent a sharp pick cutting and
        # plowing through snow.  This local contact law adds tangential snow
        # resistance only while the real pick geom is penetrating the plane.
        self.snow_surface_compliance = 0.004  # 4 mm compliant contact margin
        self.snow_drag_stiffness = 100_000.0  # N/m of visible pick compression
        self.snow_drag_damping = 15.0  # N per m/s while contact is active
        self.snow_cohesion_force = 100.0  # N once the pick cuts cohesive snow
        self.snow_normal_load_gain = 15.0  # shear resistance per N of pick load
        self.snow_drag_force_limit = 650.0
        finger_names = [
            "left_hand_thumb_0_joint",
            "left_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint",
            "left_hand_middle_1_joint",
            "left_hand_index_0_joint",
            "left_hand_index_1_joint",
            "right_hand_thumb_0_joint",
            "right_hand_thumb_1_joint",
            "right_hand_thumb_2_joint",
            "right_hand_middle_0_joint",
            "right_hand_middle_1_joint",
            "right_hand_index_0_joint",
            "right_hand_index_1_joint",
        ]
        self._finger_qpos_addresses = np.asarray(
            [
                self.model.jnt_qposadr[self._id(mujoco.mjtObj.mjOBJ_JOINT, name)]
                for name in finger_names
            ],
            dtype=np.int32,
        )
        self._finger_target_qpos = self._joint_target_qpos[
            self._finger_qpos_addresses - 7
        ].copy()

    def _id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return object_id

    def _sensor(self, sensor_id: int) -> float:
        return float(self.data.sensordata[self.model.sensor_adr[sensor_id]])

    def _pick_contact_force(self) -> tuple[float, bool]:
        force = np.zeros(6, dtype=np.float64)
        total_normal_force = 0.0
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if self.axe_pick_geom_id not in (contact.geom1, contact.geom2):
                continue
            if self.slope_geom_id not in (contact.geom1, contact.geom2):
                continue
            mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            total_normal_force += abs(float(force[0]))
        return total_normal_force, total_normal_force > 0.1

    def _hand_axe_contacts(self) -> dict[str, float]:
        """Measure opposing-digit contacts, not merely hand proximity.

        A secure grasp requires a thumb contact and an index-or-middle-finger
        contact on each hand.  Palm or wrist intersections do not count.
        """
        force = np.zeros(6, dtype=np.float64)
        contacts = {
            "left_thumb": False,
            "left_opposition": False,
            "right_thumb": False,
            "right_opposition": False,
        }
        forces = {"left": 0.0, "right": 0.0}
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if contact.geom1 in self._axe_geom_ids:
                other_geom = contact.geom2
            elif contact.geom2 in self._axe_geom_ids:
                other_geom = contact.geom1
            else:
                continue
            body_id = self.model.geom_bodyid[other_geom]
            body_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
            ) or ""
            if not body_name.startswith(("left_hand_", "right_hand_")):
                continue
            mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            normal_force = abs(float(force[0]))
            side = "left" if body_name.startswith("left_hand_") else "right"
            group = "thumb" if "_thumb_" in body_name else "opposition"
            contacts[f"{side}_{group}"] = True
            forces[side] += normal_force
        left_secure = contacts["left_thumb"] and contacts["left_opposition"]
        right_secure = contacts["right_thumb"] and contacts["right_opposition"]
        return {
            "left_grasp_contact": float(left_secure),
            "right_grasp_contact": float(right_secure),
            "left_thumb_contact": float(contacts["left_thumb"]),
            "left_opposition_contact": float(contacts["left_opposition"]),
            "right_thumb_contact": float(contacts["right_thumb"]),
            "right_opposition_contact": float(contacts["right_opposition"]),
            "left_grasp_force": forces["left"],
            "right_grasp_force": forces["right"],
        }

    def _blade_into_slope_angle_degrees(self) -> float:
        """Return blade angle below the slope surface (positive points in)."""
        axe_rotation = self.data.xmat[self.axe_body_id].reshape(3, 3)
        blade_direction = axe_rotation @ self.AXE_BLADE_VECTOR
        blade_direction /= max(float(np.linalg.norm(blade_direction)), 1e-9)
        into_normal_component = float(
            np.clip(np.dot(blade_direction, -self.slope_normal), -1.0, 1.0)
        )
        return float(np.rad2deg(np.arcsin(into_normal_component)))

    def _metrics(self) -> dict[str, float]:
        velocity = self.data.qvel[:3]
        normal_velocity = np.dot(velocity, self.slope_normal) * self.slope_normal
        slope_speed = float(np.linalg.norm(velocity - normal_velocity))
        downhill_speed = float(np.dot(velocity, self.slope_downhill))
        torso_rotation = self.data.xmat[self.torso_body_id].reshape(3, 3)
        chest_down = float(-np.dot(torso_rotation[:, 0], self.slope_normal))
        pick_height = float(np.dot(self.data.site_xpos[self.axe_pick_site_id], self.slope_normal))
        finger_error = (
            self.data.qpos[self._finger_qpos_addresses] - self._finger_target_qpos
        )
        grip_score = float(np.exp(-np.mean(np.square(finger_error / 0.35))))
        travel = float(np.dot(self.data.qpos[:3] - self._initial_position, self.slope_downhill))
        pick_force, rigid_pick_contact = self._pick_contact_force()
        grasp = self._hand_axe_contacts()
        torso_origin = self.data.xpos[self.torso_body_id]
        torso_local_rotation = torso_rotation.T
        head_local = torso_local_rotation @ (
            self.data.xpos[self.axe_body_id] - torso_origin
        )
        left_wrist_local = torso_local_rotation @ (
            self.data.xpos[self.left_wrist_body_id] - torso_origin
        )
        right_wrist_local = torso_local_rotation @ (
            self.data.xpos[self.right_wrist_body_id] - torso_origin
        )
        pick_local = torso_local_rotation @ (
            self.data.site_xpos[self.axe_pick_site_id] - torso_origin
        )
        pick_stroke_displacement = float(
            np.linalg.norm(pick_local - self._initial_pick_torso_position)
        )
        pick_lowering = float(self._initial_pick_height - pick_height)
        blade_angle = self._blade_into_slope_angle_degrees()
        ventral_margin = float(
            min(head_local[0], left_wrist_local[0], right_wrist_local[0])
        )
        result = {
            "v_slope": slope_speed,
            "v_downhill": downhill_speed,
            "f_pick": pick_force,
            "pick_contact": float(self._pick_contact_active),
            "rigid_pick_contact": float(rigid_pick_contact),
            "rigid_pick_contact_seen": float(self._rigid_pick_contact_seen),
            "peak_rigid_pick_force": self._peak_rigid_pick_force,
            "rigid_pick_contact_substep_fraction": (
                self._rigid_pick_contact_substeps / max(self._physics_substeps, 1)
            ),
            "rolling_rigid_pick_contact_fraction": float(
                np.mean(self._contact_fraction_window)
                if self._contact_fraction_window
                else 0.0
            ),
            "rolling_mean_snow_drag_force": float(
                np.mean(self._snow_drag_window) if self._snow_drag_window else 0.0
            ),
            "contact_window_steps": float(len(self._contact_fraction_window)),
            "f_chest": self._sensor(self.chest_touch_sensor_id),
            "snow_drag_force": self._last_snow_drag_force,
            "pick_height": pick_height,
            "pick_stroke_displacement": pick_stroke_displacement,
            "pick_lowering": pick_lowering,
            "pick_blade_into_slope_angle_deg": blade_angle,
            "rolling_pick_blade_into_slope_angle_deg": float(
                np.mean(self._blade_angle_window)
                if self._blade_angle_window
                else blade_angle
            ),
            "first_rigid_contact_step": float(self._first_rigid_contact_step),
            "stroke_at_first_contact": self._stroke_at_first_contact,
            "lowering_at_first_contact": self._lowering_at_first_contact,
            "blade_angle_at_first_contact_deg": self._blade_angle_at_first_contact,
            "chest_down": chest_down,
            "grip_score": grip_score,
            "axe_head_torso_x": float(head_local[0]),
            "axe_head_torso_z": float(head_local[2]),
            "left_wrist_torso_x": float(left_wrist_local[0]),
            "right_wrist_torso_x": float(right_wrist_local[0]),
            "ventral_placement_margin": ventral_margin,
            "stopping_distance": travel,
            "reset_heading_degrees": self._reset_heading_degrees,
            "reset_lateral_speed_mps": self._reset_lateral_speed,
            "reset_roll_degrees": self._reset_roll_degrees,
        }
        result.update(grasp)
        return result

    def _get_obs(self) -> np.ndarray:
        rotation = self.data.xmat[self.pelvis_body_id].reshape(3, 3)
        velocity = self.data.qvel[:3]
        projected_velocity = np.asarray(
            [
                np.dot(velocity, self.slope_downhill),
                velocity[1],
                np.dot(velocity, self.slope_normal),
            ]
        )
        metrics = self._metrics()
        return np.concatenate(
            [
                rotation.ravel(),
                self.data.qvel[3:6] * 0.10,
                projected_velocity * 0.20,
                self.data.qpos[7:] - self._joint_target_qpos,
                self.data.qvel[6:] * 0.05,
                [
                    metrics["f_pick"] / 100.0,
                    metrics["f_chest"] / 100.0,
                    np.clip(metrics["left_grasp_force"] / 50.0, 0.0, 5.0),
                    np.clip(metrics["right_grasp_force"] / 50.0, 0.0, 5.0),
                    metrics["snow_drag_force"] / 500.0,
                ],
                [
                    metrics["pick_height"] * 10.0,
                    metrics["ventral_placement_margin"] * 5.0,
                    metrics["axe_head_torso_z"] * 2.0,
                    metrics["pick_stroke_displacement"] * 5.0,
                    metrics["pick_blade_into_slope_angle_deg"] / 45.0,
                ],
                self._last_action,
            ]
        ).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        randomized = bool(options.get("randomize", self.randomize_reset))
        anchor_reset: np.ndarray | None = None
        explicit_oblique_reset = any(
            key in options
            for key in (
                "speed",
                "heading_degrees",
                "lateral_speed",
                "roll_degrees",
            )
        )
        if (
            randomized
            and not explicit_oblique_reset
            and self.anchor_resets
            and self.np_random.random() < self.anchor_probability
        ):
            anchor_reset = np.asarray(
                self.anchor_resets[
                    int(self.np_random.integers(0, len(self.anchor_resets)))
                ],
                dtype=np.float64,
            ).copy()
            # Train neighborhoods, not seven memorized points. Components are
            # downhill speed, heading, lateral velocity, and body roll.
            anchor_reset += self.np_random.uniform(
                [-0.10, -2.0, -0.10, -1.0],
                [0.10, 2.0, 0.10, 1.0],
            )
        self._step_count = 0
        self._stable_arrest_steps = 0
        self._invalid_arrest_steps = 0
        self._pick_contact_steps = 0
        self._left_grasp_contact_steps = 0
        self._right_grasp_contact_steps = 0
        self._last_action.fill(0.0)
        self._last_snow_drag_force = 0.0
        self._pick_contact_active = False
        self._peak_rigid_pick_force = 0.0
        self._rigid_pick_contact_seen = False
        self._rigid_pick_contact_substeps = 0
        self._physics_substeps = 0
        self._contact_substeps_this_step = 0
        self._snow_drag_sum_this_step = 0.0
        self._contact_fraction_window.clear()
        self._snow_drag_window.clear()
        self._blade_angle_window.clear()
        self._initial_pick_torso_position.fill(0.0)
        self._initial_pick_height = 0.0
        self._first_rigid_contact_step = -1
        self._stroke_at_first_contact = 0.0
        self._lowering_at_first_contact = 0.0
        self._blade_angle_at_first_contact = 0.0
        mujoco.mj_resetData(self.model, self.data)

        angle = np.deg2rad(35.0)
        x_position = float(options.get("x_position", 4.0))
        surface_position = np.asarray([x_position, 0.0, x_position * np.tan(angle)])
        base_height = float(options.get("base_height", 0.10))
        if randomized:
            surface_position[1] += float(self.np_random.uniform(-0.025, 0.025))
        self.data.qpos[:3] = surface_position + self.slope_normal * base_height
        base_quaternion = np.asarray(
            [0.8870108, 0.0, 0.4617486, 0.0], dtype=np.float64
        )
        heading_degrees = float(
            options.get(
                "heading_degrees",
                anchor_reset[1]
                if anchor_reset is not None
                else (
                    self.np_random.uniform(*self.heading_range_degrees)
                    if randomized
                    else 0.0
                ),
            )
        )
        roll_degrees = float(
            options.get(
                "roll_degrees",
                anchor_reset[3]
                if anchor_reset is not None
                else (
                    self.np_random.uniform(*self.roll_range_degrees)
                    if randomized
                    else 0.0
                ),
            )
        )
        heading_half_angle = 0.5 * np.deg2rad(heading_degrees)
        roll_half_angle = 0.5 * np.deg2rad(roll_degrees)
        heading_quaternion = np.concatenate(
            ([np.cos(heading_half_angle)], self.slope_normal * np.sin(heading_half_angle))
        )
        roll_quaternion = np.concatenate(
            ([np.cos(roll_half_angle)], self.slope_downhill * np.sin(roll_half_angle))
        )
        oriented_quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mulQuat(oriented_quaternion, heading_quaternion, base_quaternion)
        mujoco.mju_mulQuat(
            self.data.qpos[3:7], roll_quaternion, oriented_quaternion
        )

        self.data.qpos[7:] = self._joint_target_qpos
        self.data.ctrl[:] = self.ready_qpos

        initial_speed = float(
            options.get(
                "speed",
                anchor_reset[0]
                if anchor_reset is not None
                else (
                    self.np_random.uniform(*self.initial_speed_range)
                    if randomized
                    else 4.5
                ),
            )
        )
        lateral_speed = float(
            options.get(
                "lateral_speed",
                anchor_reset[2]
                if anchor_reset is not None
                else (
                    self.np_random.uniform(*self.lateral_speed_range)
                    if randomized
                    else 0.0
                ),
            )
        )
        self._reset_heading_degrees = heading_degrees
        self._reset_lateral_speed = lateral_speed
        self._reset_roll_degrees = roll_degrees
        self.data.qvel[:3] = self.slope_downhill * initial_speed + np.asarray(
            [0.0, lateral_speed, 0.0]
        )
        mujoco.mj_forward(self.model, self.data)

        self._initial_position = self.data.qpos[:3].copy()
        self._initial_speed = initial_speed
        torso_rotation = self.data.xmat[self.torso_body_id].reshape(3, 3)
        self._initial_pick_torso_position = torso_rotation.T @ (
            self.data.site_xpos[self.axe_pick_site_id]
            - self.data.xpos[self.torso_body_id]
        )
        self._initial_pick_height = float(
            np.dot(
                self.data.site_xpos[self.axe_pick_site_id], self.slope_normal
            )
        )
        self._previous_speed = self._metrics()["v_slope"]
        return self._get_obs(), self._metrics()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        command = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        # The real arm cannot reverse a joint target at 100 Hz. A short motor
        # command filter makes exploration physically meaningful and prevents
        # stochastic PPO samples from chattering the pick out of the snow.
        action = self.action_filter * self._last_action + (
            1.0 - self.action_filter
        ) * command
        target = self.ready_qpos.copy()
        target[self.POLICY_ACTUATOR_IDS] += action * self.action_scale
        target = np.clip(target, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        self.data.ctrl[:] = target
        self._contact_substeps_this_step = 0
        self._snow_drag_sum_this_step = 0.0
        for _ in range(self.frame_skip):
            # Split stepping exposes the current substep's collision solution
            # before forces are integrated.  Snow resistance is therefore
            # based on current physical pick contact, never a previous-step
            # latch.  It also permits a body-friction-only causal ablation.
            mujoco.mj_step1(self.model, self.data)
            self._apply_snow_pick_drag()
            self._suppress_body_slope_friction_if_disabled()
            mujoco.mj_step2(self.model, self.data)
        self._contact_fraction_window.append(
            self._contact_substeps_this_step / self.frame_skip
        )
        self._snow_drag_window.append(
            self._snow_drag_sum_this_step / self.frame_skip
        )
        self._blade_angle_window.append(
            self._blade_into_slope_angle_degrees()
        )

        metrics = self._metrics()
        speed = metrics["v_slope"]
        if metrics["pick_contact"] > 0.5:
            self._pick_contact_steps += 1
        if metrics["left_grasp_contact"] > 0.5:
            self._left_grasp_contact_steps += 1
        if metrics["right_grasp_contact"] > 0.5:
            self._right_grasp_contact_steps += 1
        left_grasp_fraction = self._left_grasp_contact_steps / self._step_count
        right_grasp_fraction = self._right_grasp_contact_steps / self._step_count
        deceleration = float(np.clip((self._previous_speed - speed) / 0.05, -1.0, 1.0))
        progress = float(np.clip((self._initial_speed - speed) / self._initial_speed, -1.0, 1.0))
        slow_reward = float(np.exp(-np.square(speed / 0.55)))
        # The pick tip is an 8 mm sphere.  Reward a center height below that
        # radius so PPO learns to engage the snow, rather than hover just above
        # contact (the former 10 mm target accidentally encouraged hovering).
        pick_height_reward = float(
            np.exp(-np.square((metrics["pick_height"] - 0.004) / 0.008))
        )
        pick_force_reward = float(np.clip(metrics["f_pick"] / 80.0, 0.0, 1.0))
        snow_drag_reward = float(
            np.clip(
                metrics["snow_drag_force"]
                / max(self.snow_drag_force_limit, 1.0),
                0.0,
                1.0,
            )
        )
        rolling_contact_reward = metrics["rolling_rigid_pick_contact_fraction"]
        rolling_drag_reward = float(
            np.clip(
                metrics["rolling_mean_snow_drag_force"]
                / max(self.snow_drag_force_limit, 1.0),
                0.0,
                1.0,
            )
        )
        posture_reward = float(np.clip(metrics["chest_down"], 0.0, 1.0))
        stroke_progress = float(
            np.clip(metrics["pick_lowering"] / 0.10, 0.0, 1.0)
        )
        blade_angle_reward = float(
            np.exp(
                -np.square(
                    (metrics["pick_blade_into_slope_angle_deg"] - 27.5)
                    / 10.0
                )
            )
        )
        rolling_blade_angle_reward = float(
            np.exp(
                -np.square(
                    (
                        metrics["rolling_pick_blade_into_slope_angle_deg"]
                        - 27.5
                    )
                    / 6.0
                )
            )
        )
        arm_qpos = self.data.qpos[
            self._actuator_qpos_addresses[self.POLICY_ACTUATOR_IDS]
        ]
        engaged_arm_qpos = self.engaged_qpos[self.POLICY_ACTUATOR_IDS]
        engaged_pose_reward = float(
            np.exp(
                -np.mean(
                    np.square(
                        (arm_qpos - engaged_arm_qpos)
                        / np.maximum(self.action_scale, 0.20)
                    )
                )
            )
        )
        action_cost = float(np.mean(np.square(action)))
        smoothness_cost = float(np.mean(np.square(action - self._last_action)))

        # This is the technique/load portion of the terminal predicate.  It is
        # also used for dense shaping once the robot becomes slow, preventing
        # PPO from farming reward in a friction stop with a shallow pick or an
        # open supporting hand.
        arrest_form = (
            metrics["contact_window_steps"] >= 50
            and metrics["rolling_rigid_pick_contact_fraction"] > 0.50
            and metrics["rolling_mean_snow_drag_force"] > 100.0
            and self._first_rigid_contact_step >= 20
            and self._stroke_at_first_contact > 0.08
            and self._lowering_at_first_contact > 0.05
            and self._blade_angle_at_first_contact > 18.0
            and 22.0
            < metrics["rolling_pick_blade_into_slope_angle_deg"]
            < 42.0
            and metrics["pick_height"]
            < (
                float(self.model.geom_size[self.axe_pick_geom_id, 0])
                + self.snow_surface_compliance
                + 0.004
            )
            and metrics["chest_down"] > 0.60
            and metrics["grip_score"] > 0.85
            and left_grasp_fraction > 0.70
            and right_grasp_fraction > 0.90
            and metrics["ventral_placement_margin"] > 0.03
            and 0.15 < metrics["axe_head_torso_z"] < 0.34
        )

        reward = (
            2.00 * deceleration
            + 0.25 * slow_reward
            + 0.20 * pick_height_reward
            + 0.50 * metrics["pick_contact"]
            + 0.20 * pick_force_reward
            + 0.50 * snow_drag_reward
            + 0.40 * rolling_contact_reward
            + 0.30 * rolling_drag_reward
            + 0.35 * stroke_progress
            + 0.60 * blade_angle_reward
            + 0.85 * rolling_blade_angle_reward
            + 0.30 * engaged_pose_reward
            + 0.10 * posture_reward
            + 0.05 * metrics["grip_score"]
            + 1.25 * metrics["left_grasp_contact"]
            + 0.75 * left_grasp_fraction
            + 0.30 * metrics["right_grasp_contact"]
            + 0.20 * right_grasp_fraction
            + 0.05 * float(np.clip(metrics["ventral_placement_margin"] / 0.10, 0.0, 1.0))
            - 0.050 * speed
            - 0.050 * action_cost
            - 0.020 * smoothness_cost
            - 1.250
        )

        if metrics["contact_window_steps"] >= 50:
            if arrest_form:
                reward += 1.50 * slow_reward
            elif speed < 0.20:
                reward -= 3.00

        premature_or_shallow_contact = (
            metrics["rigid_pick_contact"] > 0.5
            and self._first_rigid_contact_step >= 0
            and (
                self._first_rigid_contact_step < 20
                or self._stroke_at_first_contact < 0.08
                or self._lowering_at_first_contact < 0.05
                or self._blade_angle_at_first_contact < 18.0
            )
        )
        if premature_or_shallow_contact:
            # A shallow first touch can still generate enough incidental drag
            # to look useful during training, even though it is not a valid
            # self-arrest plant.  Make that local optimum clearly worse than
            # delaying contact until the pick is properly rotated and lowered.
            reward -= 3.0

        sufficient_deceleration = speed < 0.25 * self._initial_speed
        arrest_candidate = (
            speed < 0.20
            and sufficient_deceleration
            and arrest_form
        )
        self._stable_arrest_steps = self._stable_arrest_steps + 1 if arrest_candidate else 0
        success = self._stable_arrest_steps >= 25
        invalid_slow_stop = (
            speed < 0.20
            and metrics["contact_window_steps"] >= 50
            and not arrest_form
        )
        self._invalid_arrest_steps = (
            self._invalid_arrest_steps + 1 if invalid_slow_stop else 0
        )
        stalled_with_invalid_technique = self._invalid_arrest_steps >= 150

        terminated = bool(success)
        failure = (
            speed > 13.5
            # The old fall-line-only task used a narrow two-metre lane. An
            # oblique fall can legitimately travel several metres across the
            # 200 m-wide slope before arresting, so retain only a broad runaway
            # guard here.
            or abs(float(self.data.qpos[1])) > 8.0
            or not np.isfinite(self.data.qpos).all()
            or stalled_with_invalid_technique
        )
        if failure:
            terminated = True
            reward -= 250.0
        if success:
            reward += 1_500.0
        truncated = self._step_count >= self.max_episode_steps
        if truncated and not success:
            reward -= 500.0

        metrics.update(
            {
                "success": success,
                "is_arrested": success,
                "failure": failure,
                "stalled_with_invalid_technique": stalled_with_invalid_technique,
                "stable_arrest_steps": self._stable_arrest_steps,
                "pick_contact_fraction": self._pick_contact_steps / self._step_count,
                "left_grasp_contact_fraction": left_grasp_fraction,
                "right_grasp_contact_fraction": right_grasp_fraction,
                "initial_speed": self._initial_speed,
                "initial_pick_height": self._initial_pick_height,
                "valid_learned_plant_motion": bool(
                    self._first_rigid_contact_step >= 20
                    and self._stroke_at_first_contact > 0.08
                    and self._lowering_at_first_contact > 0.05
                    and self._blade_angle_at_first_contact > 18.0
                ),
                "reward_deceleration": deceleration,
                "reward_progress": progress,
                "reward_stroke_progress": stroke_progress,
                "reward_blade_angle": blade_angle_reward,
                "reward_rolling_blade_angle": rolling_blade_angle_reward,
                "reward_engaged_pose": engaged_pose_reward,
                "arrest_form": bool(arrest_form),
            }
        )
        self._previous_speed = speed
        self._last_action = action.copy()
        return self._get_obs(), float(reward), terminated, truncated, metrics

    def set_pick_enabled(self, enabled: bool) -> None:
        """Enable/disable only the pick contact for causal evaluation."""
        self._pick_enabled = bool(enabled)
        self.model.geom_contype[self.axe_pick_geom_id] = self._pick_default_contype if enabled else 0
        self.model.geom_conaffinity[self.axe_pick_geom_id] = (
            self._pick_default_conaffinity if enabled else 0
        )

    def _apply_snow_pick_drag(self) -> None:
        """Apply snow resistance only during current, visible pick contact.

        The pick has an 8 mm collision sphere and a 4 mm compliant snow
        margin. Resistance is a function of compression inside that 12 mm
        interaction band plus tip velocity. There is deliberately no contact
        hysteresis: if the solver loses pick contact, axe resistance is zero
        on that same substep.
        """
        self.data.qfrc_applied.fill(0.0)
        self._last_snow_drag_force = 0.0
        self._physics_substeps += 1
        if not self._pick_enabled:
            self._pick_contact_active = False
            return
        contact_force, in_contact = self._pick_contact_force()
        if in_contact:
            self._rigid_pick_contact_seen = True
            self._rigid_pick_contact_substeps += 1
            self._contact_substeps_this_step += 1
            self._peak_rigid_pick_force = max(
                self._peak_rigid_pick_force, contact_force
            )
            if self._first_rigid_contact_step < 0:
                torso_rotation = self.data.xmat[self.torso_body_id].reshape(
                    3, 3
                )
                pick_local = torso_rotation.T @ (
                    self.data.site_xpos[self.axe_pick_site_id]
                    - self.data.xpos[self.torso_body_id]
                )
                pick_height_now = float(
                    np.dot(
                        self.data.site_xpos[self.axe_pick_site_id],
                        self.slope_normal,
                    )
                )
                self._first_rigid_contact_step = self._step_count
                self._stroke_at_first_contact = float(
                    np.linalg.norm(
                        pick_local - self._initial_pick_torso_position
                    )
                )
                self._lowering_at_first_contact = float(
                    self._initial_pick_height - pick_height_now
                )
                self._blade_angle_at_first_contact = (
                    self._blade_into_slope_angle_degrees()
                )
        pick_radius = float(self.model.geom_size[self.axe_pick_geom_id, 0])
        pick_height = float(
            np.dot(self.data.site_xpos[self.axe_pick_site_id], self.slope_normal)
        )
        interaction_height = pick_radius + self.snow_surface_compliance
        self._pick_contact_active = bool(
            in_contact and pick_height <= interaction_height + 1e-6
        )
        if not self._pick_contact_active:
            return
        blade_angle = self._blade_into_slope_angle_degrees()
        angle_efficiency = float(np.clip((blade_angle - 15.0) / 10.0, 0.0, 1.0))
        if angle_efficiency <= 0.0:
            return
        compression = max(interaction_height - pick_height, 0.0)
        jacobian = np.zeros((3, self.nv), dtype=np.float64)
        mujoco.mj_jacSite(
            self.model, self.data, jacobian, None, self.axe_pick_site_id
        )
        pick_velocity = jacobian @ self.data.qvel
        tangential_velocity = pick_velocity - self.slope_normal * np.dot(
            pick_velocity, self.slope_normal
        )
        tangential_speed = float(np.linalg.norm(tangential_velocity))
        if tangential_speed < 1e-5:
            return
        force_magnitude = angle_efficiency * min(
            self.snow_drag_force_limit,
            self.snow_cohesion_force
            + self.snow_normal_load_gain * contact_force
            + self.snow_drag_stiffness * compression
            + self.snow_drag_damping * tangential_speed,
        )
        force = -force_magnitude * tangential_velocity / tangential_speed
        mujoco.mj_applyFT(
            self.model,
            self.data,
            force,
            np.zeros(3, dtype=np.float64),
            self.data.site_xpos[self.axe_pick_site_id],
            self.axe_body_id,
            self.data.qfrc_applied,
        )
        self._last_snow_drag_force = float(force_magnitude)
        self._snow_drag_sum_this_step += float(force_magnitude)

    def _suppress_body_slope_friction_if_disabled(self) -> None:
        """Remove tangential body/slope friction while retaining normal support.

        This operates on current contacts after ``mj_step1`` and before
        ``mj_step2``. Pick/slope and hand/axe contacts are untouched.
        """
        if self._body_slope_friction_enabled:
            return
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if self.slope_geom_id not in (contact.geom1, contact.geom2):
                continue
            if contact.geom1 in self._axe_geom_ids or contact.geom2 in self._axe_geom_ids:
                continue
            contact.friction[:] = 1e-6

    def set_initial_speed_range(self, speed_range: tuple[float, float]) -> None:
        self.initial_speed_range = tuple(speed_range)

    def set_reset_diversity(
        self,
        heading_range_degrees: tuple[float, float],
        lateral_speed_range: tuple[float, float],
        roll_range_degrees: tuple[float, float],
    ) -> None:
        """Update oblique-fall randomization without rebuilding the model."""
        self.heading_range_degrees = tuple(
            float(value) for value in heading_range_degrees
        )
        self.lateral_speed_range = tuple(
            float(value) for value in lateral_speed_range
        )
        self.roll_range_degrees = tuple(
            float(value) for value in roll_range_degrees
        )

    def set_body_slope_friction_enabled(self, enabled: bool) -> None:
        """Enable/disable only non-axe tangential contacts with the slope."""
        self._body_slope_friction_enabled = bool(enabled)

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=720, width=1280)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = self.pelvis_body_id
        camera.distance = 2.2
        camera.azimuth = 115
        camera.elevation = -20
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
