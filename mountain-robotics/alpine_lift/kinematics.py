"""Whole-body differential inverse kinematics for a floating-base G1.

The lift is a whole-body problem: to put a palm on a sling loop 20 cm off
the ground the robot must squat, which moves the pelvis, which moves the
shoulders. Solving only the arm chain does not work.

Each robot's full 35 DoF -- 6 floating-base plus 29 joints -- are free
variables, and the solver stacks weighted task residuals:

    feet     pinned to their stance pose. This is what makes the base move:
             with the feet fixed, bending the knees has to lower the pelvis.
    CoM      held over the support polygon. Without this the solver happily
             returns a configuration that reaches the target and falls over.
    palms    driven to the sling loops.
    pelvis   height from the squat profile, torso held upright.
    posture  weak pull toward a nominal pose to resolve the nullspace.

then solves the damped least-squares system

    (J^T W J + lambda I) dv = J^T W e

and integrates ``dv`` into a persistent reference configuration.

Two details matter for it to work on a physical position-controlled robot:

* Only the 29 joint entries become actuator targets. The base columns exist
  to make the solve consistent; real base motion falls out of the physics
  because the feet are genuinely planted.
* The foot targets are refreshed from the *measured* foot poses every tick,
  not held at the stance captured when the mission started. This is a
  correctness requirement, not an optimisation. Position actuators with
  finite gain sag under load, so the pelvis is always a few millimetres
  below plan; if the feet were pinned to a stale world-frame stance, the
  solver would see feet that had dropped with it and "correct" them by
  retracting the legs -- which lowers the real pelvis further. That loop
  collapses the robot in about a quarter of a second.

  Refreshed each tick, the foot rows carry zero residual and act purely as
  a constraint: whatever the joints do must not move the feet. That is
  exactly the planted-stance assumption the whole squat depends on, and it
  is what forces bent knees to lower the pelvis rather than lift the feet.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .indexing import NJ, RobotIndex, SceneIndex


def rot_error(R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """World-frame axis-angle rotation taking ``R_cur`` onto ``R_des``."""
    R_err = R_des @ R_cur.T
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(R_err).reshape(9))
    vel = np.zeros(3)
    mujoco.mju_quat2Vel(vel, quat, 1.0)
    return vel


@dataclass
class IKTargets:
    """Desired task-space pose for one robot at one control tick."""

    palm: dict[str, np.ndarray]          # "left"/"right" -> world xyz
    foot: dict[str, np.ndarray]          # pinned stance positions
    foot_rot: dict[str, np.ndarray]      # pinned stance orientations (3x3)
    pelvis_z: float                      # squat profile
    pelvis_rot: np.ndarray               # 3x3, upright at the stance yaw
    com_xy: np.ndarray                   # (2,) support-polygon target
    posture: np.ndarray                  # (29,) nullspace bias


@dataclass
class IKWeights:
    foot_pos: float = 300.0
    foot_rot: float = 120.0
    com_xy: float = 140.0
    palm_pos: float = 90.0
    pelvis_z: float = 35.0
    pelvis_rot: float = 30.0
    posture: float = 0.12
    damping: float = 4e-3
    max_step: float = 0.09      # rad per iteration, per joint
    iters: int = 8


class WholeBodyIK:
    """Damped least-squares whole-body IK over a persistent reference pose."""

    # feet 12 | com 2 | palms 6 | pelvis z 1 | pelvis rot 3 | posture 29
    NROWS = 12 + 2 + 6 + 1 + 3 + NJ

    def __init__(self, index: SceneIndex, weights: IKWeights | None = None):
        self.ix = index
        self.m = index.model
        self.w = weights or IKWeights()
        self._d = mujoco.MjData(self.m)
        nv = self.m.nv
        self._jacp = np.zeros((3, nv))
        self._jacr = np.zeros((3, nv))
        self._jc = np.zeros((3, nv))
        self._J = np.zeros((self.NROWS, 6 + NJ))
        self._e = np.zeros(self.NROWS)
        self._wt = np.zeros(self.NROWS)
        self._ref: dict[str, np.ndarray] = {}

    # -- reference management ----------------------------------------------
    def seed(self, robot: RobotIndex, qpos: np.ndarray) -> None:
        self._ref[robot.prefix] = np.asarray(qpos[robot.qpos_adr], dtype=float).copy()

    def reference(self, robot: RobotIndex) -> np.ndarray:
        return self._ref[robot.prefix].copy()

    # -- main solve ---------------------------------------------------------
    def solve(
        self,
        qpos: np.ndarray,
        robot: RobotIndex,
        tgt: IKTargets,
        iters: int | None = None,
        info: dict | None = None,
    ) -> np.ndarray:
        """Return 29 joint position targets for ``robot``."""
        d = self._d
        w = self.w
        cols = robot.dof_slice
        n_iter = w.iters if iters is None else iters

        d.qpos[:] = qpos

        for _ in range(n_iter):
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_comPos(self.m, d)

            J, e, wt = self._J, self._e, self._wt
            J[:] = 0.0
            e[:] = 0.0
            row = 0

            # --- feet pinned to the stance pose ---------------------------
            for side in ("left", "right"):
                sid = robot.foot_site[side]
                mujoco.mj_jacSite(self.m, d, self._jacp, self._jacr, sid)
                J[row:row + 3] = self._jacp[:, cols]
                e[row:row + 3] = tgt.foot[side] - d.site_xpos[sid]
                wt[row:row + 3] = w.foot_pos
                row += 3
                J[row:row + 3] = self._jacr[:, cols]
                e[row:row + 3] = rot_error(
                    d.site_xmat[sid].reshape(3, 3), tgt.foot_rot[side]
                )
                wt[row:row + 3] = w.foot_rot
                row += 3

            # --- centre of mass over the feet -----------------------------
            mujoco.mj_jacSubtreeCom(self.m, d, self._jc, robot.base_body)
            com = d.subtree_com[robot.base_body]
            J[row:row + 2] = self._jc[:2, cols]
            e[row:row + 2] = tgt.com_xy - com[:2]
            wt[row:row + 2] = w.com_xy
            row += 2

            # --- palms onto the sling loops -------------------------------
            for side in ("left", "right"):
                sid = robot.palm_site[side]
                mujoco.mj_jacSite(self.m, d, self._jacp, self._jacr, sid)
                J[row:row + 3] = self._jacp[:, cols]
                e[row:row + 3] = tgt.palm[side] - d.site_xpos[sid]
                wt[row:row + 3] = w.palm_pos
                row += 3

            # --- pelvis height and attitude -------------------------------
            bid = robot.base_body
            mujoco.mj_jacBody(self.m, d, self._jacp, self._jacr, bid)
            J[row] = self._jacp[2, cols]
            e[row] = tgt.pelvis_z - d.xpos[bid][2]
            wt[row] = w.pelvis_z
            row += 1
            J[row:row + 3] = self._jacr[:, cols]
            e[row:row + 3] = rot_error(d.xmat[bid].reshape(3, 3), tgt.pelvis_rot)
            wt[row:row + 3] = w.pelvis_rot
            row += 3

            # --- nullspace posture ----------------------------------------
            J[row:row + NJ, 6:] = np.eye(NJ)
            e[row:row + NJ] = tgt.posture - d.qpos[robot.qpos_adr]
            wt[row:row + NJ] = w.posture
            row += NJ

            # --- damped least squares -------------------------------------
            JW = J.T * wt
            H = JW @ J
            H[np.diag_indices_from(H)] += w.damping * (1.0 + np.trace(H) / H.shape[0])
            dv = np.linalg.solve(H, JW @ e)

            big = np.abs(dv[6:]).max()
            if big > w.max_step:
                dv *= w.max_step / big

            # Scatter into a full-model velocity so mj_integratePos handles
            # the floating-base quaternion correctly.
            vfull = np.zeros(self.m.nv)
            vfull[cols] = dv
            mujoco.mj_integratePos(self.m, d.qpos, vfull, 1.0)
            # Assign, do not use out=. ``d.qpos[robot.qpos_adr]`` is fancy
            # indexing and returns a copy, so an in-place clip writes into a
            # temporary and silently does nothing -- which let the solver
            # hand back a waist target of 1.18 rad against a +-0.52 limit.
            # The physics then clamped it, the commanded pose and the
            # achieved pose diverged by 37 degrees, and the hands missed the
            # log by a third of a metre.
            d.qpos[robot.qpos_adr] = np.clip(
                d.qpos[robot.qpos_adr],
                robot.jnt_range[:, 0], robot.jnt_range[:, 1],
            )

        q = d.qpos[robot.qpos_adr].copy()
        self._ref[robot.prefix] = q

        if info is not None:
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_comPos(self.m, d)
            info["palm_err"] = {
                s: float(np.linalg.norm(d.site_xpos[robot.palm_site[s]] - tgt.palm[s]))
                for s in ("left", "right")
            }
            info["foot_err"] = {
                s: float(np.linalg.norm(d.site_xpos[robot.foot_site[s]] - tgt.foot[s]))
                for s in ("left", "right")
            }
            info["com_err"] = float(
                np.linalg.norm(d.subtree_com[robot.base_body][:2] - tgt.com_xy)
            )
            info["pelvis"] = d.xpos[robot.base_body].copy()
        return q


def capture_stance(
    model: mujoco.MjModel, data: mujoco.MjData, robot: RobotIndex
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Snapshot the current foot poses to pin the stance against."""
    pos, rot = {}, {}
    for side in ("left", "right"):
        sid = robot.foot_site[side]
        pos[side] = data.site_xpos[sid].copy()
        rot[side] = data.site_xmat[sid].reshape(3, 3).copy()
    return pos, rot


def support_centre(stance_pos: dict[str, np.ndarray]) -> np.ndarray:
    """Midpoint of the two feet -- the nominal CoM target in xy."""
    return 0.5 * (stance_pos["left"][:2] + stance_pos["right"][:2])
