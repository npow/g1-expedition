"""Name -> id resolution for the composed two-robot scene.

Attaching two copies of ``g1_mjx.xml`` prefixes every name, and the payload
adds a free joint in front of both robots, so nothing about the index layout
should be hard-coded. This module resolves everything once at load time and
hands back plain numpy index arrays that the controller and the environment
can slice with.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .scene import FALL_GEOMS, GROUND_GEOMS, ROBOTS

# Actuated joints, in the order g1_mjx.xml declares its actuators.
LEG_JOINTS = [
    "hip_pitch_joint", "hip_roll_joint", "hip_yaw_joint",
    "knee_joint", "ankle_pitch_joint", "ankle_roll_joint",
]
ARM_JOINTS = [
    "shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint",
    "elbow_joint", "wrist_roll_joint", "wrist_pitch_joint", "wrist_yaw_joint",
]
JOINT_ORDER = (
    [f"left_{j}" for j in LEG_JOINTS]
    + [f"right_{j}" for j in LEG_JOINTS]
    + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    + [f"left_{j}" for j in ARM_JOINTS]
    + [f"right_{j}" for j in ARM_JOINTS]
)
NJ = len(JOINT_ORDER)  # 29

# Index of each group inside the 29-vector.
LEG_L = np.arange(0, 6)
LEG_R = np.arange(6, 12)
WAIST = np.arange(12, 15)
ARM_L = np.arange(15, 22)
ARM_R = np.arange(22, 29)
LEGS = np.concatenate([LEG_L, LEG_R])
ARMS = np.concatenate([ARM_L, ARM_R])


def _id(model, objtype, name) -> int:
    i = mujoco.mj_name2id(model, objtype, name)
    if i < 0:
        raise KeyError(f"{objtype.name} '{name}' not found in model")
    return i


@dataclass
class RobotIndex:
    """Everything the controller needs to address one G1 in the scene."""

    prefix: str
    joint_ids: np.ndarray      # (29,) mjOBJ_JOINT ids
    qpos_adr: np.ndarray       # (29,) indices into d.qpos
    dof_adr: np.ndarray        # (29,) indices into d.qvel / jacobian columns
    act_ids: np.ndarray        # (29,) indices into d.ctrl
    base_body: int
    base_qpos_adr: int         # 7 entries: xyz + wxyz
    base_dof_adr: int          # 6 entries: linear then angular
    torso_body: int
    palm_site: dict[str, int]
    foot_site: dict[str, int]
    hand_geom: dict[str, int]  # palm collision geom, for contact lookup
    fall_geoms: np.ndarray
    ctrl_range: np.ndarray     # (29, 2)
    jnt_range: np.ndarray      # (29, 2)

    @property
    def dof_slice(self) -> slice:
        """Contiguous [base(6) + joints(29)] column block of the Jacobian."""
        return slice(self.base_dof_adr, self.base_dof_adr + 6 + NJ)


@dataclass
class SceneIndex:
    model: mujoco.MjModel
    robots: dict[str, RobotIndex]
    log_body: int
    log_qpos_adr: int
    log_dof_adr: int
    log_geom: int
    ground_geoms: np.ndarray
    sensor_adr: dict[str, tuple[int, int]]

    def robot(self, prefix: str) -> RobotIndex:
        return self.robots[prefix]

    def sensor(self, data: mujoco.MjData, name: str) -> np.ndarray:
        adr, dim = self.sensor_adr[name]
        return data.sensordata[adr:adr + dim]


def build_index(model: mujoco.MjModel) -> SceneIndex:
    OBJ = mujoco.mjtObj
    robots: dict[str, RobotIndex] = {}

    for p in ROBOTS:
        jids, qadr, dadr, aids = [], [], [], []
        for jn in JOINT_ORDER:
            j = _id(model, OBJ.mjOBJ_JOINT, p + jn)
            jids.append(j)
            qadr.append(model.jnt_qposadr[j])
            dadr.append(model.jnt_dofadr[j])
            aids.append(_id(model, OBJ.mjOBJ_ACTUATOR, p + jn))

        base_body = _id(model, OBJ.mjOBJ_BODY, p + "pelvis")
        base_joint = model.body_jntadr[base_body]
        assert model.jnt_type[base_joint] == mujoco.mjtJoint.mjJNT_FREE

        fall = np.array(
            [_id(model, OBJ.mjOBJ_GEOM, p + g) for g in FALL_GEOMS], dtype=np.int32
        )
        act = np.array(aids, dtype=np.int32)
        jnt = np.array(jids, dtype=np.int32)

        robots[p] = RobotIndex(
            prefix=p,
            joint_ids=jnt,
            qpos_adr=np.array(qadr, dtype=np.int32),
            dof_adr=np.array(dadr, dtype=np.int32),
            act_ids=act,
            base_body=base_body,
            base_qpos_adr=int(model.jnt_qposadr[base_joint]),
            base_dof_adr=int(model.jnt_dofadr[base_joint]),
            torso_body=_id(model, OBJ.mjOBJ_BODY, p + "torso_link"),
            palm_site={s: _id(model, OBJ.mjOBJ_SITE, f"{p}{s}_palm") for s in ("left", "right")},
            foot_site={s: _id(model, OBJ.mjOBJ_SITE, f"{p}{s}_foot") for s in ("left", "right")},
            hand_geom={
                s: _id(model, OBJ.mjOBJ_GEOM, f"{p}{s}_hand_collision")
                for s in ("left", "right")
            },
            fall_geoms=fall,
            ctrl_range=model.actuator_ctrlrange[act].copy(),
            jnt_range=model.jnt_range[jnt].copy(),
        )

    log_body = _id(model, OBJ.mjOBJ_BODY, "log")
    log_joint = model.body_jntadr[log_body]

    sensor_adr = {}
    for i in range(model.nsensor):
        nm = mujoco.mj_id2name(model, OBJ.mjOBJ_SENSOR, i)
        sensor_adr[nm] = (int(model.sensor_adr[i]), int(model.sensor_dim[i]))

    return SceneIndex(
        model=model,
        robots=robots,
        log_body=log_body,
        log_qpos_adr=int(model.jnt_qposadr[log_joint]),
        log_dof_adr=int(model.jnt_dofadr[log_joint]),
        log_geom=_id(model, OBJ.mjOBJ_GEOM, "log_core"),
        ground_geoms=np.array(
            [_id(model, OBJ.mjOBJ_GEOM, g) for g in GROUND_GEOMS], dtype=np.int32
        ),
        sensor_adr=sensor_adr,
    )
