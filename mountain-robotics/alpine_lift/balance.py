"""Standing balance for a position-controlled G1.

Menagerie's G1 falls over in about three seconds under its own keyframes
with no controller -- position servos on the joints do not stabilise an
underactuated biped, which is exactly why the Playground models are paired
with learned policies. The lift controller therefore needs its own
stabiliser underneath the whole-body IK.

This is a capture-point regulator. For a linear inverted pendulum of height
h the divergent component of motion is

    xi = com + comvel / omega,      omega = sqrt(g / h)

and ``xi`` is the point the robot would have to put its foot on to come to
rest. While the feet are planted the only authority available is to shift
the centre of pressure under them, so the controller drives ``xi`` back to
the middle of the support polygon using an ankle strategy, with the hips
and knees added as a second tier for errors the ankles cannot absorb on
their own.

Signs were measured on the model rather than derived: a positive
``ankle_pitch`` offset drives the CoM backwards, a positive ``ankle_roll``
offset drives it to the robot's left, and ``hip_pitch`` and ``knee`` share
the sagittal sign with the ankle.

The correction is returned as a shift of the *IK's centre-of-mass target*,
not as joint offsets bolted onto the IK's answer. That distinction matters.
Offsetting the ankles after the fact rotates the whole body about the feet,
which moves the shoulders, which moves the palms -- and the IK has no idea
it happened, so the hands sit centimetres off the sling loops and the arm
servos end up fighting the constraint that is holding the load. Feeding the
correction in as a task target instead lets one solve satisfy balance and
grip together, and the palms stay where the plan put them.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .indexing import LEG_L, LEG_R, NJ, SceneIndex
from .poses import YAW
from .scene import ROBOTS

G = 9.81

# Offsets within the 6-vector for one leg: hip_pitch, hip_roll, hip_yaw,
# knee, ankle_pitch, ankle_roll.
HIP_PITCH, HIP_ROLL, HIP_YAW, KNEE, ANK_PITCH, ANK_ROLL = range(6)


@dataclass
class BalanceConfig:
    # Slow tier: reshape the posture the IK plans.
    k_cp: float = 0.60           # CoM-target shift per metre of capture-point error
    max_shift: float = 0.07      # m of authority over the CoM target
    # Fast tier: direct joint offsets for disturbances the posture loop
    # cannot answer in time. Kept modest -- large offsets rotate the body
    # about the feet and drag the palms off the sling loops.
    k_ankle_pitch: float = 0.70
    k_ankle_roll: float = 0.60
    k_hip_pitch: float = 0.26
    k_hip_roll: float = 0.22
    k_knee: float = 0.18
    max_ankle: float = 0.22      # rad
    max_hip: float = 0.14
    max_knee: float = 0.14
    vel_tau: float = 0.04        # s, filter on the CoM velocity estimate
    enabled: bool = True


class BalanceStabilizer:
    """Capture-point ankle/hip/knee regulator, one instance per scene."""

    def __init__(self, ix: SceneIndex, cfg: BalanceConfig | None = None):
        self.ix = ix
        self.cfg = cfg or BalanceConfig()
        self.ref: dict[str, np.ndarray] = {}
        self.hull: dict[str, np.ndarray] = {}
        self._vel: dict[str, np.ndarray] = {}
        self._int: dict[str, float] = {}
        self._last: dict[str, np.ndarray] = {}
        self._mass: dict[str, float] = {}
        # Carried load per robot: (vertical newtons, world hand midpoint).
        self._load: dict[str, tuple[float, np.ndarray]] = {}

    def set_load(
        self, prefix: str, force_on_object: np.ndarray, hand_point: np.ndarray
    ) -> None:
        """Tell the stabiliser what this robot is doing to the world, and where.

        Two terms, and both matter for a push.

        The vertical part is any weight the hands are carrying, which acts as
        an added point mass outside the footprint.

        The horizontal part is the one that ends a push. Shoving a log
        downhill with force F puts F straight back into the robot at hand
        height, and that reaction tips it over backwards exactly as if its
        centre of mass had moved uphill by F*h/W. A stabiliser reasoning only
        about ``subtree_com`` cannot see it, insists the robot is balanced,
        and lets it fall over while pushing -- which is precisely what
        happened before this term was added.
        """
        f = np.asarray(force_on_object, dtype=float)
        self._load[prefix] = (f, np.asarray(hand_point, float))

    def effective_com(self, data: mujoco.MjData, prefix: str) -> np.ndarray:
        """Where the robot's mass effectively acts, given what it is doing.

        Combines its own centre of mass with any weight on the hands and the
        reaction to any horizontal force it is applying.
        """
        r = self.ix.robot(prefix)
        com = data.subtree_com[r.base_body].copy()
        f, hand = self._load.get(prefix, (None, None))
        if hand is None or f is None:
            return com
        m_r = self._mass[prefix]
        m_l = max(float(f[2]), 0.0) / G
        out = (m_r * com + m_l * hand) / (m_r + m_l) if m_l > 1e-6 else com.copy()
        # Reaction to the horizontal push, referred to the ankle.
        weight = (m_r + m_l) * G
        if weight > 1e-6:
            out[:2] = out[:2] - f[:2] * max(hand[2], 0.05) / weight
        return out

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Latch each robot's support polygon from its planted feet."""
        for p in ROBOTS:
            self._mass[p] = float(model.body_subtreemass[self.ix.robot(p).base_body])
            self._load[p] = (None, None)
            hull = convex_hull(foot_polygon(model, data, self.ix, p))
            self.hull[p] = hull
            self.ref[p] = 0.5 * (hull.min(axis=0) + hull.max(axis=0))
            self._vel[p] = np.zeros(2)
            self._int[p] = 0.0
            self._last[p] = np.zeros(2)

    def margin(self, data: mujoco.MjData, prefix: str) -> float:
        """Signed distance from the ground-projected CoM to the stance edge."""
        return polygon_margin(self.effective_com(data, prefix)[:2], self.hull[prefix])

    def cp_margin(self, data: mujoco.MjData, prefix: str) -> float:
        """Same, but for the capture point -- the dynamic version."""
        com3 = self.effective_com(data, prefix)
        omega = np.sqrt(G / max(com3[2], 0.25))
        xi = com3[:2] + self._vel[prefix] / omega
        return polygon_margin(xi, self.hull[prefix])

    def stance_hull(self, prefix: str) -> np.ndarray:
        return self.hull[prefix]

    def capture_point(
        self, data: mujoco.MjData, prefix: str, dt: float
    ) -> np.ndarray:
        """Filtered instantaneous capture point in world xy."""
        c = self.cfg
        com3 = self.effective_com(data, prefix)
        v = self.ix.sensor(data, f"{prefix}comvel")[:2]
        a = dt / max(c.vel_tau, dt)
        self._vel[prefix] += a * (v - self._vel[prefix])
        omega = np.sqrt(G / max(com3[2], 0.25))
        return com3[:2] + self._vel[prefix] / omega

    def correct(
        self,
        data: mujoco.MjData,
        prefix: str,
        dt: float,
        com_ref: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Return (com_target_shift, ankle_offsets(29,), diagnostics)."""
        c = self.cfg
        off = np.zeros(NJ)
        xi = self.capture_point(data, prefix, dt)
        ref = np.asarray(com_ref[:2], dtype=float)
        err = xi - ref

        yaw = YAW[prefix]
        fwd = np.array([np.cos(yaw), np.sin(yaw)])
        lat = np.array([-np.sin(yaw), np.cos(yaw)])
        e_f, e_l = float(err @ fwd), float(err @ lat)
        diag = {"cp_fwd": e_f, "cp_lat": e_l}

        if not c.enabled:
            return np.zeros(2), off, diag

        # Pull the planned CoM to the far side of the error so the posture the
        # IK returns is already a restoring one.
        shift = -c.k_cp * err
        n = np.linalg.norm(shift)
        if n > c.max_shift:
            shift *= c.max_shift / n

        ap = float(np.clip(c.k_ankle_pitch * e_f, -c.max_ankle, c.max_ankle))
        ar = float(np.clip(-c.k_ankle_roll * e_l, -c.max_ankle, c.max_ankle))
        hp = float(np.clip(c.k_hip_pitch * e_f, -c.max_hip, c.max_hip))
        hr = float(np.clip(-c.k_hip_roll * e_l, -c.max_hip, c.max_hip))
        kn = float(np.clip(c.k_knee * e_f, -c.max_knee, c.max_knee))
        for leg in (LEG_L, LEG_R):
            off[leg[ANK_PITCH]] = ap
            off[leg[ANK_ROLL]] = ar
            off[leg[HIP_PITCH]] = hp
            off[leg[HIP_ROLL]] = hr
            off[leg[KNEE]] = kn
        diag.update(ankle=ap, hip=hp)
        return shift, off, diag


def foot_polygon(
    model: mujoco.MjModel, data: mujoco.MjData, ix: SceneIndex, prefix: str
) -> np.ndarray:
    """Ground-projected corners of both foot boxes, in world xy."""
    pts = []
    for side in ("left", "right"):
        gid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}{side}_foot_box_collision"
        )
        half = model.geom_size[gid]
        R = data.geom_xmat[gid].reshape(3, 3)
        c = data.geom_xpos[gid]
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    pts.append(c + R @ (np.array([sx, sy, sz]) * half))
    return np.asarray(pts)[:, :2]


def support_centre_world(
    model: mujoco.MjModel, data: mujoco.MjData, ix: SceneIndex, prefix: str
) -> np.ndarray:
    p = foot_polygon(model, data, ix, prefix)
    return 0.5 * (p.min(axis=0) + p.max(axis=0))


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotone-chain 2D hull. Kept dependency-free so the RL env can use it."""
    pts = np.unique(np.round(points, 6), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(ps):
        out: list[np.ndarray] = []
        for q in ps:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) > 0:
                    break
                out.pop()
            out.append(q)
        return out

    return np.asarray(half(pts)[:-1] + half(pts[::-1])[:-1])


def polygon_margin(point: np.ndarray, hull: np.ndarray) -> float:
    """Signed distance from ``point`` to the hull boundary; +ve is inside."""
    n = len(hull)
    if n < 3:
        return -float(np.linalg.norm(point - hull.mean(axis=0)))
    inside = True
    best = np.inf
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        e = b - a
        L = np.linalg.norm(e)
        if L < 1e-9:
            continue
        cross = (e[0] * (point[1] - a[1]) - e[1] * (point[0] - a[0])) / L
        if cross < 0:
            inside = False
        t = np.clip(((point - a) @ e) / (L * L), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(point - (a + t * e))))
    return best if inside else -best
