"""Reading the trunk loads out of the contact solver.

There are no slings and no equality constraints in this task: the robots put
their palms on the bark and push. That means every newton the team applies
arrives through contact, and ``mj_contactForce`` is the whole instrument. It
is also the honest one -- a real machine would read the same quantity from
joint-torque estimation at the wrist.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .indexing import SceneIndex
from .scene import ROBOTS


def contact_loads(
    model: mujoco.MjModel, data: mujoco.MjData, ix: SceneIndex
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """World-frame force each robot, and each hand, puts into the trunk.

    Returns ``(per_robot, per_hand)``; hand keys are ``"A_left"`` etc. A
    contact is attributed to a hand when the robot-side geom is that hand's
    palm, and to the robot regardless of which of its geoms is touching --
    a shin braced against the trunk is still that robot pushing.
    """
    per_robot = {p: np.zeros(3) for p in ROBOTS}
    per_hand = {f"{p}{s}": np.zeros(3) for p in ROBOTS for s in ("left", "right")}
    hand_of = {
        int(ix.robot(p).hand_geom[s]): f"{p}{s}"
        for p in ROBOTS for s in ("left", "right")
    }
    log = ix.log_geom
    buf = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if log not in (g1, g2):
            continue
        other = g2 if g1 == log else g1
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
        prefix = next((p for p in ROBOTS if name.startswith(p)), None)
        if prefix is None:
            continue
        mujoco.mj_contactForce(model, data, i, buf)
        # mj_contactForce returns the force on geom2 from geom1, in the
        # contact frame; rotate to world and flip if the log is geom1.
        world = c.frame.reshape(3, 3).T @ buf[:3]
        f = world if g2 == log else -world
        per_robot[prefix] += f
        key = hand_of.get(other)
        if key is not None:
            per_hand[key] += f
    return per_robot, per_hand


def push_share(loads: dict[str, np.ndarray]) -> float:
    """Fraction of the *downhill* push supplied by robot A.

    0.5 is balanced. Returns 0.5 when nobody is pushing, so the skew
    controller has a benign value to regulate against before contact.
    """
    fa = max(-loads["A_"][1], 0.0)
    fb = max(-loads["B_"][1], 0.0)
    tot = fa + fb
    return 0.5 if tot < 1e-6 else float(fa / tot)


def log_skew(model: mujoco.MjModel, data: mujoco.MjData, ix: SceneIndex) -> float:
    """Yaw of the trunk away from square to the trail, in radians.

    This is the quantity that makes the task cooperative. Two robots pushing
    a 2.2 m trunk at different rates do not move it crookedly by a little --
    they wind it round until one end digs into the bank and it stops. Square
    is not cosmetic; it is the difference between the log going over the edge
    and the log jamming.
    """
    ax = ix.sensor(data, "log_xaxis")
    return float(np.arctan2(ax[1], ax[0]))


def log_speed(data: mujoco.MjData, ix: SceneIndex) -> float:
    """Downhill speed of the trunk, m/s (positive = moving toward the edge)."""
    return float(-data.qvel[ix.log_dof_adr + 1])


def support_margin(
    model: mujoco.MjModel, data: mujoco.MjData, ix: SceneIndex, prefix: str
) -> float:
    """Kept for callers that want a cheap axis-aligned stance margin."""
    r = ix.robot(prefix)
    feet = np.stack([data.site_xpos[r.foot_site[s]] for s in ("left", "right")])
    com = data.subtree_com[r.base_body]
    lo = feet.min(axis=0)[:2] - 0.10
    hi = feet.max(axis=0)[:2] + 0.10
    return float(np.minimum(com[:2] - lo, hi - com[:2]).min())
