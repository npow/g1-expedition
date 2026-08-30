"""Mission controller for the trail-clearing roll.

Coordination principle
----------------------
Both robots derive their hand targets from a single measured object -- the
trunk's own axis -- rather than from independent arm trajectories. Each hand
is placed on the trunk's uphill flank at a fixed azimuth, at a radius
slightly *inside* the bark. That inward offset is the whole actuator: a
position-controlled palm commanded 3 cm into a solid object pushes on it,
and the harder it is commanded in, the harder it pushes.

Following the *measured* trunk rather than a planned trajectory matters. A
planned one either outruns the log, in which case the hands leave the bark
and the push stops, or lags it, in which case the robots are dragged. Riding
the measurement keeps the palms in contact for as long as the arms can
reach, and the log coasts the rest of the way on its own -- which is what
actually clears the trail.

The only differential between the two robots is the square controller in
``coordinator.py``, which advances one robot's hands relative to the other
to keep the trunk from winding round.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .balance import BalanceConfig, BalanceStabilizer
from .coordinator import Coordinator, CoordinatorConfig
from .forces import contact_loads, log_skew
from .indexing import ARM_L, ARM_R, ARMS, SceneIndex
from .kinematics import (
    IKTargets, IKWeights, WholeBodyIK, capture_stance, support_centre,
)
from .poses import CARRY_POSTURE, READY, YAW, stagger_pose, torso_frame
from .scene import A, B, ROBOTS, SceneConfig, grip_x, lip_y

# Arms forward and open, the nullspace bias while reaching for the bark.
REACH_POSTURE = READY.copy()
REACH_POSTURE[ARM_L] = [0.55, 0.20, 0.0, 0.60, 0.0, 0.10, 0.0]
REACH_POSTURE[ARM_R] = [0.55, -0.20, 0.0, 0.60, 0.0, 0.10, 0.0]

# Arms in and braced, for pushing.
PUSH_POSTURE = READY.copy()
PUSH_POSTURE[ARM_L] = [0.42, 0.16, 0.0, 0.90, 0.0, 0.20, 0.0]
PUSH_POSTURE[ARM_R] = [0.42, -0.16, 0.0, 0.90, 0.0, 0.20, 0.0]

PUSH_ELBOW = 0.35   # rad; smaller is straighter

PHASES: tuple[tuple[str, float], ...] = (
    ("READY", 0.6),      # settle on the stance
    ("APPROACH", 2.6),   # squat, palms out to just clear of the bark
    ("PROBE", 2.4),      # creep in and nudge: how heavy is it, really
    ("BRACE", 0.8),      # back off a fingerwidth, set the stance, decide
    ("SHOVE", 1.3),      # one committed stroke, both robots together
    ("FOLLOW", 1.2),     # it runs; hands back, stand up
    ("RETREAT", 1.6),
)
PHASE_NAMES = tuple(p[0] for p in PHASES)
PUSH_PHASES = ("PROBE", "SHOVE")


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


@dataclass
class MissionConfig:
    # Clear of the bark at the end of APPROACH. Generous, because the palm
    # *site* sits inside the hand geom: a small gap here means the hand is
    # already touching, and the approach knocks the log rolling before the
    # team has even weighed it.
    approach_gap: float = 0.095
    # The push is force-controlled, not position-controlled. Commanding a
    # palm a fixed depth into a rigid trunk is a stiffness, not a force: the
    # first version asked for 3-5 cm of penetration and got 200-450 N, five
    # times what the log needs and enough to tip the robots over backwards.
    # Here the press depth is the *output* of a loop that tracks a commanded
    # force, which is what a person pushing something actually does.
    probe_touch: float = 0.010     # m of penetration during the nudge
    # Rate-limited, not time-limited. Closing a fixed gap over a fixed phase
    # duration means the hand arrives at whatever speed that implies -- 0.15
    # m/s here -- and a palm landing on 150 kg of rigid timber at 0.15 m/s is
    # a 148 N impact that knocks the robot backwards and slides its feet.
    # Creeping in at a fixed slow rate makes contact an event with no
    # momentum in it.
    touch_rate: float = 0.045      # m/s while closing the last of the gap
    # A trunk lying on a level tread is in neutral equilibrium: the instant a
    # hand touches it, it moves. There is no sustained push to regulate,
    # because the log accelerates away faster than a standing arm can follow.
    # What the team actually has is one committed shove, and these are its
    # shape. Coordination is then about both robots shoving together and
    # equally -- which is exactly what keeps the trunk square.
    stroke: float = 0.22           # m the hands drive through
    stroke_time: float = 0.75      # s
    k_press: float = 1.0e-3        # m/s of press per N of force error
    creep_rate: float = 0.110      # m/s of press while still out of contact
    contact_n: float = 2.0         # N that counts as touching
    press_min: float = -0.010
    # The palm *site* sits ~8 cm inside the hand's collision geom, so the
    # commanded site depth that produces contact is that much larger than
    # the physical penetration. The force loop finds the right value; this
    # just has to leave it room.
    press_max: float = 0.090
    squat_pelvis: float = 0.62
    stand_pelvis: float = 0.72
    lean_gain: float = 1.5         # rad of forward torso lean per m of squat
    max_lean: float = 0.55
    reach_limit: float = 0.48      # m the hands can follow the trunk downhill
    away_speed: float = 0.30       # m/s at which the trunk is gone and we let go
    away_travel: float = 0.26      # ...or this far moved, whichever comes first
    release_recover: float = 0.45  # s to get the hands back and stand up
    control_hz: float = 50.0
    ik_iters: int = 8
    load_ff: float = 0.65
    # No integral trim on the palms here. It exists to close a reach error
    # against free space; against a log it is a wind-up that discharges the
    # moment the target crosses the bark. During APPROACH the palm cannot
    # reach its target anyway (that is what the stand-off is), so the term
    # saturates at its clamp and then delivers 5.5 cm of extra penetration
    # as an impact.
    palm_ki: float = 1.6
    palm_i_clamp: float = 0.055

    # residual authority, per robot: press, advance, CoM x, CoM y, pelvis z
    residual_press: float = 0.014
    residual_adv: float = 0.020
    residual_com: float = 0.016
    residual_pelvis: float = 0.020
    residual_tau: float = 0.10

    abort_back: float = 1.0        # s to take the hands off after an abort
    abort_stand: float = 1.6


@dataclass
class Telemetry:
    t: float = 0.0
    phase: str = "READY"
    phase_idx: int = 0
    phase_frac: float = 0.0
    pace: float = 1.0
    log_y: float = 0.0
    log_travel: float = 0.0        # m moved downhill from where it lay
    log_z: float = 0.0
    to_edge: float = 0.0           # m still to go before the fall line takes it
    over_edge: bool = False
    skew_deg: float = 0.0
    speed: float = 0.0
    push_total: float = 0.0
    force_cmd: float = 0.0
    breakaway: float = 0.0
    mass_est: float = 0.0
    share: float = 0.5
    hand_force: dict = field(default_factory=dict)
    robot_load: dict = field(default_factory=dict)
    margin: dict = field(default_factory=dict)
    cp_margin: dict = field(default_factory=dict)
    balance: dict = field(default_factory=dict)
    palm_gap: dict = field(default_factory=dict)
    engaged: bool = False
    aborted: bool = False
    abort_reason: str = ""
    events: list = field(default_factory=list)
    go: bool = True


class RollController:
    """Phase-scripted whole-body controller for the coordinated roll."""

    def __init__(
        self,
        model: mujoco.MjModel,
        ix: SceneIndex,
        scene_cfg: SceneConfig,
        mission: MissionConfig | None = None,
        coord_cfg: CoordinatorConfig | None = None,
        ik_weights: IKWeights | None = None,
        balance_cfg: BalanceConfig | None = None,
    ):
        self.m = model
        self.ix = ix
        self.scfg = scene_cfg
        self.cfg = mission or MissionConfig()
        self.ik = WholeBodyIK(ix, ik_weights or IKWeights(iters=self.cfg.ik_iters))
        self.coord = Coordinator(ix, scene_cfg, coord_cfg)
        self.balance = BalanceStabilizer(ix, balance_cfg)
        self.n_sub = max(1, int(round(1.0 / (self.cfg.control_hz * model.opt.timestep))))
        self.phase_ends = np.cumsum([d for _, d in PHASES])
        self.duration = float(self.phase_ends[-1])
        self._kp = {
            p: np.maximum(model.actuator_gainprm[ix.robot(p).act_ids, 0], 1.0)
            for p in ROBOTS
        }
        # The IK's nullspace posture has to be built on the *stance the robot
        # is actually in*. Leaving these at the parallel-footed READY pose
        # while the feet are pinned staggered sets the posture task fighting
        # the foot task every tick, and the robot loses the stance it was
        # braced in exactly when it needs it.
        base = stagger_pose(scene_cfg.stagger) if scene_cfg.stagger else READY.copy()
        self._posture = {}
        for tag, arms in (
            ("ready", None),
            ("reach", ([0.55, 0.20, 0.0, 0.60, 0.0, 0.10, 0.0],
                       [0.55, -0.20, 0.0, 0.60, 0.0, 0.10, 0.0])),
            # Straighter arms for the push. A bent elbow puts the load on
            # the shoulder actuator; a straight one carries it down the
            # skeleton into the stance, which is why a person shoves with
            # locked arms.
            ("push", ([0.30, 0.14, 0.0, PUSH_ELBOW, 0.0, 0.15, 0.0],
                      [0.30, -0.14, 0.0, PUSH_ELBOW, 0.0, 0.15, 0.0])),
        ):
            q = base.copy()
            if arms is not None:
                q[ARM_L], q[ARM_R] = arms
            self._posture[tag] = q

        self._res_gain = np.array(
            [self.cfg.residual_press, self.cfg.residual_adv,
             self.cfg.residual_com, self.cfg.residual_com,
             self.cfg.residual_pelvis] * 2
        )
        self.reset_state()

    # ------------------------------------------------------------------ init
    def reset_state(self) -> None:
        self.t = 0.0
        self.engaged = False
        self.aborted = False
        self.abort_reason = ""
        self._abort_t = 0.0
        self.events: list[tuple[float, str]] = []
        self._stance: dict[str, tuple] = {}
        self._palm0: dict[str, dict[str, np.ndarray]] = {}
        self._rest_y = 0.0
        self._pace = 1.0
        self._res_lp = np.zeros(10)
        self._press = {p: 0.0 for p in ROBOTS}
        self._stroke_y0: dict[str, float] = {}
        self._released = False
        self._release_t = 0.0
        self._palm_i = {
            f"{p}{s}": np.zeros(3) for p in ROBOTS for s in ("left", "right")
        }
        self.coord.reset()
        self.tele = Telemetry()

    def bind(self, data: mujoco.MjData) -> None:
        for p in ROBOTS:
            r = self.ix.robot(p)
            self._stance[p] = capture_stance(self.m, data, r)
            self._palm0[p] = {
                s: data.site_xpos[r.palm_site[s]].copy() for s in ("left", "right")
            }
            self.ik.seed(r, data.qpos)
        self.balance.bind(self.m, data)
        self.coord.cp_fn = self.balance.cp_margin
        self._rest_y = float(data.xpos[self.ix.log_body][1])

    # -------------------------------------------------------------- phasing
    def phase_at(self, t: float) -> tuple[int, str, float]:
        if self.aborted:
            c = self.cfg
            e = t - self._abort_t
            if e < c.abort_back:
                return -1, "ABORT-BACK", e / c.abort_back
            return -2, "SAFE", min(1.0, (e - c.abort_back) / c.abort_stand)
        i = int(np.searchsorted(self.phase_ends, t, side="right"))
        i = min(i, len(PHASES) - 1)
        start = 0.0 if i == 0 else float(self.phase_ends[i - 1])
        return i, PHASE_NAMES[i], min(1.0, (t - start) / PHASES[i][1])

    def _reach_in(self, name: str, frac: float) -> float:
        """Commanded hand offset from the bark surface, along -y.

        Positive is clear of the trunk, negative is into it. This is a
        position schedule, not a force loop, and that is a consequence of the
        physics rather than a simplification: a log in neutral equilibrium
        accelerates away from any sustained force, so there is no steady
        contact to regulate. What the team has is a stand-off, a nudge, and
        a stroke.
        """
        c = self.cfg
        if name == "APPROACH":
            return c.approach_gap
        if name == "PROBE":
            travelled = c.touch_rate * frac * PHASES[PHASE_NAMES.index("PROBE")][1]
            return max(c.approach_gap - travelled, -c.probe_touch)
        if name == "BRACE":
            return -c.probe_touch + smoothstep(frac) * c.probe_touch
        if name == "SHOVE":
            return -c.stroke * smoothstep(min(1.0, frac * 1.3 / c.stroke_time))
        return c.approach_gap

    def _brace(self, force: np.ndarray, hand: np.ndarray, prefix: str) -> np.ndarray:
        w = max(self.balance._mass.get(prefix, 35.0), 1.0) * 9.81
        return np.asarray(force[:2], dtype=float) * max(float(hand[2]), 0.05) / w

    def _axis_point(self, data: mujoco.MjData, x: float) -> np.ndarray:
        """Where the trunk's axis sits at a given station along the trail.

        Uses the trunk's own x-axis rather than assuming it is square, so a
        skewed trunk still gets its hands in the right place -- which is what
        lets the square controller see the error instead of fighting it.
        """
        p = data.xpos[self.ix.log_body]
        a = self.ix.sensor(data, "log_xaxis")
        ax = a if abs(a[0]) > 1e-3 else np.array([1.0, 0.0, 0.0])
        t = (x - p[0]) / ax[0]
        return p + t * ax

    # ------------------------------------------------------------ main tick
    def control(
        self, data: mujoco.MjData, residual: np.ndarray | None = None
    ) -> Telemetry:
        c = self.cfg
        scfg = self.scfg
        idx, name, frac = self.phase_at(self.t)
        if self._released and name in ("SHOVE", "PROBE"):
            # Snap out of the push, do not ease out of it. The reaction that
            # was holding the robot up vanishes the instant the trunk goes,
            # so the hands have to come back and the trunk straighten in
            # about the time it takes to notice -- half a second, not one
            # and a half.
            name = "FOLLOW"
            frac = min(1.0, (self.t - self._release_t) / c.release_recover)
        dt = 1.0 / c.control_hz

        # ---- sense ------------------------------------------------------
        per_robot, per_hand = contact_loads(self.m, data, self.ix)
        obs = self.coord.update(data, per_robot, per_hand, self.engaged, dt)

        # Once it is rolling, let go and stand up. A robot that keeps
        # reaching after a trunk which is now accelerating away simply
        # follows it onto its face -- and a person shoving something heavy
        # straightens up the moment it moves, for the same reason.
        travel_now = self._rest_y - float(data.xpos[self.ix.log_body][1])
        if (not self._released and self.engaged
                and (obs.speed > c.away_speed or travel_now > c.away_travel)
                and name in ("PROBE", "BRACE", "SHOVE")):
            self._released = True
            self._release_t = self.t
            self.events.append((self.t, f"trunk away at {obs.speed:.2f} m/s"))

        touching = obs.push_total > 0.5
        if touching and not self.engaged and name in PUSH_PHASES:
            self.engaged = True
            self.coord.on_touch()
            self.events.append((self.t, "hands on the bark"))

        # Safety supervision applies while the team is working. Once the
        # trunk is away there is nothing left to abort -- the job is done and
        # the only task remaining is for each robot to get its own balance
        # back, which the stabiliser handles. Calling that an abort would
        # report a mission failure for a mission that had already succeeded.
        trunk_gone = self._released or travel_now > c.away_travel
        if self.engaged and not self.aborted and not trunk_gone:
            reason = self.coord.abort_reason(self.m, data, obs)
            if reason:
                self._abort(reason)

        # ---- residual ----------------------------------------------------
        raw = np.zeros(10) if residual is None else np.clip(residual, -1.0, 1.0)
        alpha = dt / max(c.residual_tau, dt)
        self._res_lp += alpha * (raw - self._res_lp)
        res = self._res_lp * self._res_gain

        # ---- references --------------------------------------------------
        reach_sched = self._reach_in(name, frac)
        pelvis_z = (
            c.stand_pelvis + smoothstep(frac) * (c.squat_pelvis - c.stand_pelvis)
            if name == "APPROACH" else
            c.squat_pelvis + smoothstep(frac) * (c.stand_pelvis - c.squat_pelvis)
            if name in ("RETREAT", "SAFE", "FOLLOW") else
            c.stand_pelvis if name == "READY" else c.squat_pelvis
        )
        lean = float(np.clip(
            c.lean_gain * max(0.0, c.stand_pelvis - pelvis_z), 0.0, c.max_lean))
        square = self.coord.square_command(obs) if self.engaged else 0.0

        palm_gap: dict[str, float] = {}
        bal_diag: dict[str, dict] = {}
        for k, p in enumerate(ROBOTS):
            r = self.ix.robot(p)
            stance_pos, stance_rot = capture_stance(self.m, data, r)
            self._stance[p] = (stance_pos, stance_rot)
            rb = res[5 * k:5 * (k + 1)]

            # Differential advance: hold back whichever robot's end is ahead.
            adv = (square if p == A else -square) + rb[1]
            depth = float(np.clip(rb[0], -0.03, 0.03))
            # The press acts horizontally, into the hill face, not radially.
            # Radial press at a 43-degree grip drives the hand target 9 cm
            # *downward* for every 13 cm of commanded press, which is both
            # the wrong direction for rolling a log and out of the arm's
            # reach. Horizontal is what a person does and what turns the
            # trunk.
            reach_in = reach_sched - depth
            if name == "SHOVE" and p not in self._stroke_y0:
                # Latch where the trunk was when the stroke began.
                self._stroke_y0[p] = float(
                    self._axis_point(data, grip_x(scfg, p, "left"))[1])

            palm: dict[str, np.ndarray] = {}
            for side in ("left", "right"):
                x_h = grip_x(scfg, p, side)
                axis = self._axis_point(data, x_h)
                phi = scfg.grip_phi
                axis_y = axis[1]
                if name == "SHOVE":
                    # Drive from where the trunk was when the stroke began,
                    # not from where it is now -- following it just backs the
                    # hands off as fast as it leaves.
                    axis_y = self._stroke_y0[p]
                target = np.array([
                    x_h,
                    axis_y + adv + scfg.log_radius * np.sin(phi) + reach_in,
                    axis[2] + scfg.log_radius * np.cos(phi),
                ])
                # Do not chase it past the arms. Once the trunk is rolling it
                # outruns a standing robot in well under a second, and an IK
                # that keeps reaching after it simply walks the machine onto
                # its face. Stop at the edge of the envelope and let it go --
                # the log does not need us any more.
                target[1] = max(target[1], scfg.robot_y - c.reach_limit)

                if name in ("READY",):
                    target = self._palm0[p][side].copy()
                elif name == "APPROACH":
                    s = smoothstep(frac)
                    target = (1 - s) * self._palm0[p][side] + s * target
                elif name in ("FOLLOW", "RETREAT", "ABORT-BACK", "SAFE"):
                    s = smoothstep(frac)
                    back = np.array([0.0, 0.34 * s, 0.22 * s])
                    target = target + back

                key = f"{p}{side}"
                actual = data.site_xpos[r.palm_site[side]]
                err = target - actual
                # Integral trim while reaching only: once the palm is on the
                # bark the residual is the trunk pushing back, and
                # integrating that just winds the arm up against the log.
                # Integral authority during the shove only. On the approach
                # the target is deliberately unreachable (that is the
                # stand-off), so integrating there just winds up and then
                # discharges as an impact when the target crosses the bark.
                # During the shove the target is *inside* the trunk and the
                # residual is the log resisting, which is exactly the error
                # worth integrating -- it is what turns a commanded stroke
                # into a sustained push.
                if name == "SHOVE":
                    self._palm_i[key] = np.clip(
                        self._palm_i[key] + c.palm_ki * err * dt,
                        -c.palm_i_clamp, c.palm_i_clamp)
                else:
                    self._palm_i[key] *= 0.94
                palm[side] = target + self._palm_i[key]
                # Gap from the palm to the bark surface, the pace signal.
                surf = np.array([
                    x_h,
                    axis[1] + scfg.log_radius * np.sin(phi),
                    axis[2] + scfg.log_radius * np.cos(phi)])
                palm_gap[key] = float(np.linalg.norm(actual - surf))

            hand_mid = 0.5 * (
                data.site_xpos[r.palm_site["left"]] + data.site_xpos[r.palm_site["right"]]
            )
            self.balance.set_load(p, per_robot[p], hand_mid)
            com_xy = support_centre(stance_pos) + rb[2:4]
            if self.engaged:
                # Brace feed-forward, derived rather than tuned. The reaction
                # to a push F at hand height h displaces the effective centre
                # of mass by F*h/W, so planning the real one the same distance
                # the other way puts the effective one back over the feet.
                com_xy = com_xy + self._brace(per_robot[p], hand_mid, p)
            shift, ankle_off, bal_info = self.balance.correct(data, p, dt, com_xy)
            com_xy = com_xy + shift
            bal_diag[p] = bal_info

            posture = self._posture[
                "reach" if name == "APPROACH"
                else "push" if name in PUSH_PHASES
                else "ready"
            ]
            tgt = IKTargets(
                palm=palm, foot=stance_pos, foot_rot=stance_rot,
                pelvis_z=pelvis_z + rb[4],
                pelvis_rot=torso_frame(YAW[p], lean),
                com_xy=com_xy, posture=posture,
            )
            q_des = self.ik.solve(data.qpos, r, tgt) + ankle_off
            tau = data.qfrc_bias[r.dof_adr].copy()
            tau[ARMS] -= c.load_ff * data.qfrc_constraint[r.dof_adr[ARMS]]
            q_cmd = q_des + np.clip(tau / self._kp[p], -0.35, 0.35)
            data.ctrl[r.act_ids] = np.clip(
                q_cmd, r.ctrl_range[:, 0], r.ctrl_range[:, 1])

        # ---- pace + clock -------------------------------------------------
        self._pace = (
            self.coord.pace(max(palm_gap.values()))
            if (self.engaged and name == "PROBE" and palm_gap) else 1.0
        )
        self.t += self._pace * dt

        # ---- telemetry ----------------------------------------------------
        lp = data.xpos[self.ix.log_body]
        edge = lip_y(scfg)
        self.tele = Telemetry(
            t=self.t, phase=name, phase_idx=idx, phase_frac=frac, pace=self._pace,
            force_cmd=float(reach_sched),
            log_y=float(lp[1]),
            log_travel=float(self._rest_y - lp[1]),
            log_z=float(lp[2]),
            to_edge=float(lp[1] - edge),
            over_edge=bool(lp[1] < edge - 0.15),
            skew_deg=float(np.degrees(obs.skew)),
            speed=obs.speed,
            push_total=obs.push_total,
            breakaway=obs.breakaway,
            mass_est=obs.mass_est,
            share=obs.share,
            hand_force={k: float(np.linalg.norm(v)) for k, v in per_hand.items()},
            robot_load={p: obs.load[p] for p in ROBOTS},
            margin={p: self.balance.margin(data, p) for p in ROBOTS},
            cp_margin={p: self.balance.cp_margin(data, p) for p in ROBOTS},
            balance=bal_diag,
            palm_gap=palm_gap,
            engaged=self.engaged,
            aborted=self.aborted,
            abort_reason=self.abort_reason,
            events=list(self.events),
            go=self.coord.go,
        )
        return self.tele

    @property
    def done(self) -> bool:
        if self.aborted:
            c = self.cfg
            return self.t - self._abort_t >= c.abort_back + c.abort_stand
        return self.t >= self.duration

    def _abort(self, reason: str) -> None:
        self.aborted = True
        self.abort_reason = reason
        self._abort_t = self.t
        self.events.append((self.t, f"ABORT: {reason}"))
