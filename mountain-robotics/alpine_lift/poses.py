"""Nominal G1 configurations and scene reset.

The joint vectors are menagerie's ``home`` and ``knees_bent`` keyframes from
``scene_mjx.xml``, re-expressed as 29-vectors in actuator order. ``<attach>``
does not carry keyframes across, so the composed scene sets its initial
state programmatically instead.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .indexing import ARM_L, ARM_R, LEG_L, LEG_R, SceneIndex
from .scene import A, B, ROBOTS, SceneConfig, log_rest_z

# menagerie "home": standing tall, arms hanging.
HOME = np.array([
    -0.10, 0.0, 0.0, 0.30, -0.20, 0.0,        # left leg
    -0.10, 0.0, 0.0, 0.30, -0.20, 0.0,        # right leg
    0.0, 0.0, 0.0,                             # waist yaw/roll/pitch
    0.20, 0.20, 0.0, 1.28, 0.0, 0.0, 0.0,      # left arm
    0.20, -0.20, 0.0, 1.28, 0.0, 0.0, 0.0,     # right arm
])
HOME_HEIGHT = 0.783675

# menagerie "knees_bent": the athletic stance the lift starts from.
READY = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.073,
    0.20, 0.22, 0.0, 1.00, 0.0, 0.0, 0.0,
    0.20, -0.22, 0.0, 1.00, 0.0, 0.0, 0.0,
])
READY_HEIGHT = 0.755

# Posture bias used as the IK nullspace target while carrying: elbows tucked
# in and down so the load hangs close to the chest rather than out in front,
# which is both what a person does and what keeps the CoM inside the feet.
CARRY_POSTURE = READY.copy()
CARRY_POSTURE[ARM_L] = [0.05, 0.30, -0.10, 1.15, 0.0, 0.10, 0.0]
CARRY_POSTURE[ARM_R] = [0.05, -0.30, 0.10, 1.15, 0.0, 0.10, 0.0]

def stagger_pose(delta: float, base: np.ndarray | None = None) -> np.ndarray:
    """A braced stance: one foot back, one forward.

    This is what a person does before shoving something heavy, and it is not
    cosmetic. Feet side by side give the G1 about 0.30 m of fore-aft support,
    and a horizontal push at chest height walks the centre of pressure out of
    that in a fraction of a second -- measured here as the difference between
    clearing the trail and falling over immediately after. Staggering the
    feet roughly doubles the depth of the support polygon in the direction
    the push reacts.

    The stagger goes *backwards only*: one thigh swings back, the other
    stays put, with the moved ankle counter-rotated so both soles stay flat.
    Splitting the difference and advancing the other foot as well is the
    obvious construction and the wrong one here -- it buys support in front,
    where nothing is pushing, and spends the clearance the hands need by
    putting a boot under the trunk. The reaction to shoving a log downhill
    pushes the robot *uphill*, so the support that matters is behind it.

    ``delta`` is the swing angle in radians.
    """
    q = (READY if base is None else base).copy()
    # leg layout: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    q[LEG_L[0]] += delta      # left leg swings back to brace
    q[LEG_L[4]] -= delta
    return q


# Both robots stand uphill of the trunk and face the fall line, so they
# share a heading -- unlike a carry, where they would face each other.
YAW = {A: -np.pi / 2, B: -np.pi / 2}


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def yaw_mat(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def pitch_mat(pitch: float) -> np.ndarray:
    """Rotation about the body's own +y: positive pitches the chest forward."""
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def torso_frame(yaw: float, lean: float) -> np.ndarray:
    """Stance yaw with a forward lean -- the posture a person lifts from."""
    return yaw_mat(yaw) @ pitch_mat(lean)


def reset(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ix: SceneIndex,
    cfg: SceneConfig,
    pose: np.ndarray | None = None,
    height: float | None = None,
    log_yaw: float = 0.0,
    jitter: float = 0.0,
    rng: np.random.Generator | None = None,
) -> None:
    """Put the scene into its start-of-mission state."""
    pose = READY if pose is None else pose
    height = READY_HEIGHT if height is None else height
    rng = rng or np.random.default_rng(0)

    mujoco.mj_resetData(model, data)

    # the trunk lying across the tread
    pa = ix.log_qpos_adr
    data.qpos[pa:pa + 3] = [0.0, cfg.log_y, log_rest_z(cfg)]
    data.qpos[pa + 3:pa + 7] = yaw_quat(log_yaw)

    for p in ROBOTS:
        r = ix.robot(p)
        sx = -1.0 if p == A else 1.0
        base = np.array([sx * cfg.robot_x, cfg.robot_y, height])
        yaw = YAW[p]
        if jitter > 0.0:
            base[:2] += rng.normal(0.0, jitter, 2)
            yaw += rng.normal(0.0, jitter * 0.6)
        data.qpos[r.base_qpos_adr:r.base_qpos_adr + 3] = base
        data.qpos[r.base_qpos_adr + 3:r.base_qpos_adr + 7] = yaw_quat(yaw)

        q = pose.copy()
        if jitter > 0.0:
            q += rng.normal(0.0, jitter * 0.35, q.shape)
        q = np.clip(q, r.jnt_range[:, 0], r.jnt_range[:, 1])
        data.qpos[r.qpos_adr] = q
        data.ctrl[r.act_ids] = q

    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def settle(model: mujoco.MjModel, data: mujoco.MjData, seconds: float = 0.6) -> None:
    """Let the stance converge onto the ground before the mission clock starts."""
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
