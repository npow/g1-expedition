"""Physical fall and pretrained RL whole-body get-up controller for Unitree G1.

The WBC observation/action contract and constants are adapted from
wbc-mjlab/wbc-g1-deploy at commit 6dabf86 (Apache-2.0). The robot's floating
base is never prescribed after reset: a finite push, gravity, contacts, and
joint torques determine the full fall and recovery trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Iterator

import mujoco
import numpy as np
import onnxruntime as ort

from scripts.fetch_getup_assets import ensure_getup_assets


POLICY_DT = 0.02
SIM_DT = 0.002
SUBSTEPS = int(round(POLICY_DT / SIM_DT))

DEFAULT_JOINT_POS = np.asarray(
    [
        -0.1, 0, 0, 0.3, -0.2, 0,
        -0.1, 0, 0, 0.3, -0.2, 0,
        0, 0, 0,
        0.35, 0.18, 0, 0.87, 0, 0, 0,
        0.35, -0.18, 0, 0.87, 0, 0, 0,
    ],
    dtype=np.float64,
)
WBC_KP = np.asarray(
    [
        40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
        40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
        40.179, 28.501, 28.501,
        14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778,
        14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778,
    ],
    dtype=np.float64,
)
WBC_KD = np.asarray(
    [
        2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
        2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
        2.558, 1.814, 1.814,
        0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068,
        0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068,
    ],
    dtype=np.float64,
)
ACTION_SCALE = np.asarray(
    [
        0.51830, 0.33048, 0.51830, 0.33048, 0.55962, 0.55962,
        0.51830, 0.33048, 0.51830, 0.33048, 0.55962, 0.55962,
        0.51830, 0.55962, 0.55962,
        0.55962, 0.55962, 0.55962, 0.55962, 0.55962, 0.12814, 0.12814,
        0.55962, 0.55962, 0.55962, 0.55962, 0.55962, 0.12814, 0.12814,
    ],
    dtype=np.float64,
)
MOTOR_TORQUE_LIMIT = 0.98 * np.asarray(
    [
        83.3, 131, 83.3, 131, 63.8, 63.8,
        83.3, 131, 83.3, 131, 63.8, 63.8,
        83.3, 63.8, 63.8,
        31.9, 31.9, 31.9, 31.9, 31.9, 8.6, 8.6,
        31.9, 31.9, 31.9, 31.9, 31.9, 8.6, 8.6,
    ],
    dtype=np.float64,
)
FLOOR_READY_KP = np.asarray(
    [
        100, 100, 100, 150, 40, 40,
        100, 100, 100, 150, 40, 40,
        200, 200, 200,
        *([40] * 14),
    ],
    dtype=np.float64,
)
FLOOR_READY_KD = np.asarray(
    [
        2, 2, 2, 4, 2, 2,
        2, 2, 2, 4, 2, 2,
        5, 5, 5,
        *([10] * 14),
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class FallCondition:
    """Finite world-frame disturbance used to produce a repeatable fall."""

    label: str
    force_xyz_n: tuple[float, float, float]
    push_seconds: float = 0.20
    settle_seconds: float = 3.00


DEFAULT_FALL = FallCondition("backward shove", (-100.0, 0.0, 0.0))


def _quat_inverse_rotate(quaternion_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a world vector into the quaternion's body frame."""

    w, x, y, z = quaternion_wxyz
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return rotation.T @ np.asarray(vector, dtype=np.float64)


class G1PhysicalGetup:
    """Run a no-teleport fall, grounded alignment, and learned WBC recovery."""

    BODY_NAMES = (
        "pelvis", "left_hip_pitch_link", "left_hip_roll_link",
        "left_hip_yaw_link", "left_knee_link", "left_ankle_pitch_link",
        "left_ankle_roll_link", "right_hip_pitch_link", "right_hip_roll_link",
        "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link",
        "right_ankle_roll_link", "waist_yaw_link", "waist_roll_link",
        "torso_link", "left_shoulder_pitch_link", "left_shoulder_roll_link",
        "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_link",
        "left_wrist_pitch_link", "left_wrist_yaw_link",
        "right_shoulder_pitch_link", "right_shoulder_roll_link",
        "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link",
        "right_wrist_pitch_link", "right_wrist_yaw_link",
    )
    ANCHOR_INDEX = BODY_NAMES.index("torso_link")

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        *,
        fetch_assets: bool = True,
    ) -> None:
        root = Path(__file__).resolve().parent
        if model_path is None:
            model_path = root / "assets" / "unitree_g1" / "scene_getup.xml"
        asset_dir = (
            ensure_getup_assets()
            if fetch_assets
            else root / "third_party" / "wbc_g1_getup"
        )
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        if not np.isclose(self.model.opt.timestep, SIM_DT):
            raise ValueError(f"Expected a {SIM_DT}s simulation step")
        if self.model.nu != 29 or self.model.nq != 36:
            raise ValueError("The WBC adapter requires the 29-DoF G1 model")
        # Enforce the upstream per-motor catalog continuously inside MuJoCo,
        # including between 50 Hz policy updates.
        self.model.actuator_forcelimited[:] = 1
        self.model.actuator_forcerange[:, 0] = -MOTOR_TORQUE_LIMIT
        self.model.actuator_forcerange[:, 1] = MOTOR_TORQUE_LIMIT
        self._stand_gainprm = self.model.actuator_gainprm.copy()
        self._stand_biasprm = self.model.actuator_biasprm.copy()

        motion = np.load(asset_dir / "getup_01.npz", allow_pickle=True)
        self.ref_joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64)
        self.ref_body_pos = np.asarray(motion["body_pos_w"], dtype=np.float64)
        self.ref_body_quat = np.asarray(motion["body_quat_w"], dtype=np.float64)
        self.ref_body_lin_vel = np.asarray(motion["body_lin_vel_w"], dtype=np.float64)
        self.ref_body_ang_vel = np.asarray(motion["body_ang_vel_w"], dtype=np.float64)
        self.session = ort.InferenceSession(
            str(asset_dir / "policy.onnx"), providers=["CPUExecutionProvider"]
        )
        input_shape = self.session.get_inputs()[0].shape
        if input_shape != [1, 132]:
            raise ValueError(f"Unexpected WBC policy observation shape: {input_shape}")

        self.torso_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.ground_geom_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.left_foot_body_id = self._id(
            mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"
        )
        self.right_foot_body_id = self._id(
            mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link"
        )
        self.foot_body_ids = {self.left_foot_body_id, self.right_foot_body_id}

        self.phase = "standing"
        self.frame = 0
        self.recovery_frame = 0
        self.previous_action = np.zeros(29, dtype=np.float64)
        self.floor_ready_start = np.zeros(29, dtype=np.float64)
        self.condition = DEFAULT_FALL
        self.mode = "policy"
        self.root_teleports_after_fall_start = 0
        self.peak_contact_force_n = 0.0
        self.peak_actuator_torque_nm = 0.0
        self.peak_motor_torque_ratio = 0.0
        self.maximum_penetration_m = 0.0
        self.grounded_frames = 0
        self._stable_frames = 0
        self._success = False
        self.fall_end: dict[str, Any] = {}
        self._contact_force = np.zeros(6, dtype=np.float64)

    def _id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"Missing MuJoCo object: {name}")
        return object_id

    def _set_pd_gains(self, kp: np.ndarray, kd: np.ndarray) -> None:
        self.model.actuator_gainprm[:, 0] = kp
        self.model.actuator_biasprm[:, 1] = -kp
        self.model.actuator_biasprm[:, 2] = -kd

    def _set_passive_damping(self) -> None:
        self.model.actuator_gainprm[:] = 0.0
        self.model.actuator_biasprm[:] = 0.0
        self.model.actuator_biasprm[:, 2] = -2.0

    def reset(
        self,
        *,
        condition: FallCondition = DEFAULT_FALL,
        mode: str = "policy",
    ) -> dict[str, Any]:
        if mode not in {"policy", "reference_only", "motors_off"}:
            raise ValueError(f"Unknown recovery mode: {mode}")
        self.model.actuator_gainprm[:] = self._stand_gainprm
        self.model.actuator_biasprm[:] = self._stand_biasprm
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        # Settle the visible initial stand before the fall begins.
        for _ in range(250):
            mujoco.mj_step(self.model, self.data)
        self.condition = condition
        self.mode = mode
        self.phase = "pre_fall"
        self.frame = 0
        self.recovery_frame = 0
        self.previous_action.fill(0.0)
        self.root_teleports_after_fall_start = 0
        self.peak_contact_force_n = 0.0
        self.peak_actuator_torque_nm = 0.0
        self.peak_motor_torque_ratio = 0.0
        self.maximum_penetration_m = 0.0
        self.grounded_frames = 0
        self._stable_frames = 0
        self._success = False
        self.fall_end = {}
        return self.telemetry()

    @property
    def pre_fall_frames(self) -> int:
        return int(round(0.5 / POLICY_DT))

    @property
    def fall_frames(self) -> int:
        return int(round(self.condition.settle_seconds / POLICY_DT))

    @property
    def push_frames(self) -> int:
        return int(round(self.condition.push_seconds / POLICY_DT))

    @property
    def floor_ready_frames(self) -> int:
        return int(round(2.0 / POLICY_DT))

    @property
    def recovery_frames(self) -> int:
        return len(self.ref_joint_pos) + 100

    @property
    def done(self) -> bool:
        return self.phase == "done"

    def step(self) -> dict[str, Any]:
        """Advance one 50 Hz controller frame and return causal telemetry."""

        if self.phase == "pre_fall":
            self._step_pre_fall()
        elif self.phase == "fall":
            self._step_fall()
        elif self.phase == "floor_ready":
            self._step_floor_ready()
        elif self.phase == "recovery":
            self._step_recovery()
        elif self.phase == "done":
            self._step_physics()
        else:
            raise RuntimeError(f"Invalid phase: {self.phase}")
        self._update_frame_counters()
        return self.telemetry()

    def rollout(self) -> Iterator[dict[str, Any]]:
        while not self.done:
            yield self.step()

    def _step_pre_fall(self) -> None:
        self._step_physics()
        self.frame += 1
        if self.frame >= self.pre_fall_frames:
            self.phase = "fall"
            self.frame = 0
            self._set_passive_damping()

    def _step_fall(self) -> None:
        if self.frame < self.push_frames:
            self.data.xfrc_applied[self.torso_id, :3] = self.condition.force_xyz_n
        else:
            self.data.xfrc_applied[self.torso_id, :3] = 0.0
        self._step_physics()
        self.frame += 1
        if self.frame >= self.fall_frames:
            self.data.xfrc_applied[:] = 0.0
            torso_rotation = self.data.xmat[self.torso_id].reshape(3, 3)
            contacts = self.ground_contact_bodies()
            self.fall_end = {
                "pelvis_height_m": float(self.data.qpos[2]),
                "torso_upright": float(torso_rotation[2, 2]),
                "base_linear_speed_mps": float(np.linalg.norm(self.data.qvel[:3])),
                "base_angular_speed_radps": float(np.linalg.norm(self.data.qvel[3:6])),
                "nonfoot_ground_contact": bool(
                    contacts - self.foot_body_ids - {0}
                ),
                "ground_contact_bodies": sorted(
                    self.model.body(body_id).name
                    for body_id in contacts
                    if body_id != 0
                ),
                "peak_impact_force_n": self.peak_contact_force_n,
            }
            self.phase = "floor_ready"
            self.frame = 0
            self.floor_ready_start[:] = self.data.qpos[7:]
            self._set_pd_gains(FLOOR_READY_KP, FLOOR_READY_KD)

    def _step_floor_ready(self) -> None:
        blend = min((self.frame + 1) / self.floor_ready_frames, 1.0)
        self.data.ctrl[:] = (
            (1.0 - blend) * self.floor_ready_start
            + blend * self.ref_joint_pos[0]
        )
        self._step_physics()
        self.frame += 1
        if self.frame >= self.floor_ready_frames:
            self.phase = "recovery"
            self.frame = 0
            self.recovery_frame = 0
            if self.mode == "motors_off":
                self._set_passive_damping()
            else:
                self._set_pd_gains(WBC_KP, WBC_KD)

    def _step_recovery(self) -> None:
        reference_frame = min(self.recovery_frame, len(self.ref_joint_pos) - 1)
        reference_position = self.ref_joint_pos[reference_frame]
        if self.mode == "motors_off":
            self.data.ctrl[:] = self.data.qpos[7:]
            self.previous_action.fill(0.0)
        else:
            observation = self._policy_observation(reference_frame)
            if self.mode == "policy":
                action = self.session.run(
                    ["actions"], {"obs": observation[None]}
                )[0][0].astype(np.float64)
            else:
                action = np.zeros(29, dtype=np.float64)
            target = reference_position + ACTION_SCALE * action
            desired_torque = (
                WBC_KP * (target - self.data.qpos[7:])
                - WBC_KD * self.data.qvel[6:]
            )
            clipped_torque = np.clip(
                desired_torque, -MOTOR_TORQUE_LIMIT, MOTOR_TORQUE_LIMIT
            )
            # Back-solve a position command whose PD torque respects the catalog.
            target = self.data.qpos[7:] + (
                clipped_torque + WBC_KD * self.data.qvel[6:]
            ) / WBC_KP
            self.data.ctrl[:] = np.clip(
                target,
                self.model.actuator_ctrlrange[:, 0],
                self.model.actuator_ctrlrange[:, 1],
            )
            self.previous_action[:] = action
        self._step_physics()
        self.recovery_frame += 1
        if self.recovery_frame >= self.recovery_frames:
            self.phase = "done"

    def _policy_observation(self, reference_frame: int) -> np.ndarray:
        anchor = self.ANCHOR_INDEX
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
                _quat_inverse_rotate(reference_quaternion, np.asarray([0, 0, -1])),
                self.ref_joint_pos[reference_frame],
            )
        )
        measured = np.concatenate(
            (
                self.data.qvel[3:6],
                _quat_inverse_rotate(self.data.qpos[3:7], np.asarray([0, 0, -1])),
                self.data.qpos[7:] - DEFAULT_JOINT_POS,
                self.data.qvel[6:],
                self.previous_action,
            )
        )
        observation = np.concatenate((reference, measured)).astype(np.float32)
        if observation.shape != (132,):
            raise RuntimeError(f"Bad policy observation shape: {observation.shape}")
        return observation

    def _step_physics(self) -> None:
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
            self.peak_actuator_torque_nm = max(
                self.peak_actuator_torque_nm,
                float(np.max(np.abs(self.data.actuator_force))),
            )
            self.peak_motor_torque_ratio = max(
                self.peak_motor_torque_ratio,
                float(np.max(np.abs(self.data.actuator_force) / MOTOR_TORQUE_LIMIT)),
            )
            for contact_index in range(self.data.ncon):
                contact = self.data.contact[contact_index]
                self.maximum_penetration_m = max(
                    self.maximum_penetration_m, max(0.0, -float(contact.dist))
                )
                mujoco.mj_contactForce(
                    self.model, self.data, contact_index, self._contact_force
                )
                self.peak_contact_force_n = max(
                    self.peak_contact_force_n, abs(float(self._contact_force[0]))
                )

    def ground_contact_bodies(self) -> set[int]:
        bodies: set[int] = set()
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if contact.geom1 == self.ground_geom_id:
                bodies.add(int(self.model.geom_bodyid[contact.geom2]))
            elif contact.geom2 == self.ground_geom_id:
                bodies.add(int(self.model.geom_bodyid[contact.geom1]))
        return bodies

    def _update_frame_counters(self) -> None:
        torso_rotation = self.data.xmat[self.torso_id].reshape(3, 3)
        contacts = self.ground_contact_bodies()
        nonfoot_contacts = contacts - self.foot_body_ids - {0}
        if nonfoot_contacts:
            self.grounded_frames += 1
        stable = (
            self.phase in {"recovery", "done"}
            and self.data.qpos[2] > 0.72
            and torso_rotation[2, 2] > 0.95
            and np.linalg.norm(self.data.qvel[:3]) < 0.15
            and np.linalg.norm(self.data.qvel[3:6]) < 0.35
            and self.foot_body_ids.issubset(contacts)
        )
        self._stable_frames = self._stable_frames + 1 if stable else 0
        self._success = self._success or self._stable_frames >= 25

    def telemetry(self) -> dict[str, Any]:
        torso_rotation = self.data.xmat[self.torso_id].reshape(3, 3)
        contacts = self.ground_contact_bodies()
        nonfoot_contacts = contacts - self.foot_body_ids - {0}
        both_feet = self.foot_body_ids.issubset(contacts)
        base_linear_speed = float(np.linalg.norm(self.data.qvel[:3]))
        base_angular_speed = float(np.linalg.norm(self.data.qvel[3:6]))
        pelvis_height = float(self.data.qpos[2])
        upright = float(torso_rotation[2, 2])
        grounded = bool(nonfoot_contacts)
        return {
            "phase": self.phase,
            "controller_frame": self.frame,
            "recovery_frame": self.recovery_frame,
            "sim_time_s": float(self.data.time),
            "pelvis_height_m": pelvis_height,
            "torso_upright": upright,
            "base_linear_speed_mps": base_linear_speed,
            "base_angular_speed_radps": base_angular_speed,
            "grounded_nonfoot_contact": grounded,
            "ground_contact_bodies": sorted(
                self.model.body(body_id).name for body_id in contacts if body_id != 0
            ),
            "both_feet_contact": both_feet,
            "contact_count": int(self.data.ncon),
            "applied_push_force_n": float(
                np.linalg.norm(self.data.xfrc_applied[self.torso_id, :3])
            ),
            "peak_contact_force_n": self.peak_contact_force_n,
            "peak_actuator_torque_nm": self.peak_actuator_torque_nm,
            "peak_motor_torque_ratio": self.peak_motor_torque_ratio,
            "maximum_contact_penetration_m": self.maximum_penetration_m,
            "root_teleports_after_fall_start": self.root_teleports_after_fall_start,
            "policy_inference_active": self.mode == "policy" and self.phase == "recovery",
            "recovery_mode": self.mode,
            "success": self._success,
        }

    def report(self) -> dict[str, Any]:
        info = self.telemetry()
        return {
            "success": bool(info["success"]),
            "recovery_mode": self.mode,
            "fall_condition": {
                "label": self.condition.label,
                "force_xyz_n": list(self.condition.force_xyz_n),
                "push_duration_s": self.condition.push_seconds,
                "fall_settle_duration_s": self.condition.settle_seconds,
            },
            "fall_end": self.fall_end,
            "final_pelvis_height_m": info["pelvis_height_m"],
            "final_torso_upright": info["torso_upright"],
            "final_base_linear_speed_mps": info["base_linear_speed_mps"],
            "final_base_angular_speed_radps": info["base_angular_speed_radps"],
            "final_both_feet_contact": info["both_feet_contact"],
            "grounded_duration_s": self.grounded_frames * POLICY_DT,
            "peak_contact_force_n": self.peak_contact_force_n,
            "peak_actuator_torque_nm": self.peak_actuator_torque_nm,
            "peak_motor_torque_ratio": self.peak_motor_torque_ratio,
            "motor_torque_catalog_max_nm": float(np.max(MOTOR_TORQUE_LIMIT)),
            "maximum_contact_penetration_m": self.maximum_penetration_m,
            "root_teleports_after_fall_start": self.root_teleports_after_fall_start,
            "simulated_duration_s": float(self.data.time - 0.5),
            "physics": {
                "gravity_mps2": self.model.opt.gravity.tolist(),
                "timestep_s": float(self.model.opt.timestep),
                "solver": "Newton",
                "integrator": "implicitfast",
                "policy_rate_hz": int(round(1.0 / POLICY_DT)),
                "substeps_per_policy_action": SUBSTEPS,
            },
        }

    def close(self) -> None:
        self.session = None
