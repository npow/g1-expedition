"""Team-level regulation for the trail-clearing roll.

The controller decides where the trunk should go; the coordinator decides
whether that is possible and keeps the two machines honest while it happens.
It owns one decision and four loops.

probe / go-no-go  Push gently and watch. The force at which the trunk breaks
                  out of its bed is the number that matters -- not its mass,
                  which is irrelevant to whether it can be rolled. Nobody
                  hands a mountain robot a spec sheet, so the team finds out
                  by leaning on it, and declines if the breakaway force is
                  past what the arms can hold.

square            The one genuinely cooperative loop. Two robots pushing a
                  2.2 m trunk at different rates do not move it crookedly by
                  a little; they wind it round until an end digs in and it
                  stops. A PD on the measured yaw drives a *differential
                  advance* between the two robots' hands -- the one whose end
                  is ahead is held back. Without it the roll jams.

share             Push force split between the robots, trimmed through the
                  press depth. Reported whether or not it is regulated,
                  because it is the thing a human crew argues about.

pace              A shared clock that slows when the trunk is not keeping up
                  with the commanded advance. The hands cannot outrun the
                  log: if they do they simply leave the bark, and the team
                  has to stop and re-establish contact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .forces import log_skew, log_speed, push_share
from .indexing import SceneIndex
from .scene import A, B, ROBOTS, SceneConfig

G = 9.81


@dataclass
class CoordinatorConfig:
    # capacity
    arm_capacity: float = 60.0      # N per hand, braced push
    reflex_margin: float = 2.2      # multiple of capacity that trips the reflex
    reflex_ticks: int = 10
    reflex_grace: int = 25

    # square (the cooperative loop)
    kp_skew: float = 0.55           # m of differential advance per rad of yaw
    kd_skew: float = 0.07
    max_skew_cmd: float = 0.10      # m

    # force sharing, trimmed through press depth
    kp_share: float = 0.030         # m of press per unit of share error
    max_press_trim: float = 0.018

    # stance
    com_lean: float = 0.050         # m of forward lean at full push
    com_lean_ref: float = 160.0     # N at which the full lean applies

    # pace
    pace_tol: float = 0.070         # m of hand-to-bark gap before slowing
    pace_min: float = 0.25

    # probe
    probe_ticks: int = 40           # ticks of steady push before deciding
    breakaway_speed: float = 0.02   # m/s that counts as "it moved"

    # abort
    max_skew: float = 0.42          # rad
    min_cp_margin: float = -0.055
    cp_ticks: int = 12
    # A robot shoving something heavy is legitimately pitched a long way
    # forward -- that is where the force comes from. The attitude check is
    # for a machine that is going over, not one that is working.
    min_torso_up: float = 0.30

    tau: float = 0.10               # s, filter on force signals


@dataclass
class CoordObs:
    share: float = 0.5              # fraction of the push supplied by robot A
    push_total: float = 0.0         # N downhill into the trunk
    breakaway: float = 0.0          # N at which it started to move
    mass_est: float = 0.0           # kg, from F = m a once it is moving
    load: dict = field(default_factory=dict)
    hand: dict = field(default_factory=dict)
    peak_hand: float = 0.0
    skew: float = 0.0               # rad, +ve = the +x end has swung uphill
    skew_rate: float = 0.0
    speed: float = 0.0              # m/s downhill
    moving: bool = False


class Coordinator:
    def __init__(
        self,
        ix: SceneIndex,
        scene_cfg: SceneConfig,
        cfg: CoordinatorConfig | None = None,
    ):
        self.ix = ix
        self.scfg = scene_cfg
        self.cfg = cfg or CoordinatorConfig()
        self.reset()

    def reset(self) -> None:
        self._f = {f"{p}{s}": 0.0 for p in ROBOTS for s in ("left", "right")}
        self._push = 0.0
        self._share = 0.5
        self._skew = 0.0
        self._skew_prev = 0.0
        self._skew_rate = 0.0
        self._speed = 0.0
        self._accel = 0.0
        self._speed_prev = 0.0
        self._mass = 0.0
        self.breakaway = 0.0
        self._probe = 0
        self._since_touch = 0
        self._reflex = 0
        self._cp_bad = {p: 0 for p in ROBOTS}
        self.decided = False
        self.go = True
        self.go_reason = "not yet probed"
        self.cp_fn = None

    def on_touch(self) -> None:
        self._since_touch = 0
        self._probe = 0
        self._reflex = 0
        self.decided = False

    # ------------------------------------------------------------- sensing
    def update(
        self,
        data: mujoco.MjData,
        per_robot: dict[str, np.ndarray],
        per_hand: dict[str, np.ndarray],
        engaged: bool,
        dt: float,
    ) -> CoordObs:
        c = self.cfg
        a = dt / max(c.tau, dt)

        for k, v in per_hand.items():
            self._f[k] += a * (float(np.linalg.norm(v)) - self._f[k])

        push = sum(max(-per_robot[p][1], 0.0) for p in ROBOTS)
        self._push += a * (push - self._push)
        self._share += a * (push_share(per_robot) - self._share)

        skew = log_skew(self.ix.model, data, self.ix)
        self._skew_rate += a * ((skew - self._skew_prev) / dt - self._skew_rate)
        self._skew_prev = skew
        self._skew += a * (skew - self._skew)

        v = log_speed(data, self.ix)
        self._accel += a * ((v - self._speed_prev) / dt - self._accel)
        self._speed_prev = v
        self._speed += a * (v - self._speed)
        moving = self._speed > c.breakaway_speed

        # Mass from F = m a, but only while it is actually accelerating --
        # a quasi-static push tells you about friction, not inertia.
        if moving and self._accel > 0.15 and self._push > 5.0:
            est = self._push / max(self._accel, 1e-3)
            if 5.0 < est < 2000.0:
                self._mass += a * (est - self._mass)

        if engaged:
            self._since_touch += 1
            if not moving and self._push > self.breakaway:
                # Highest force reached while it was still stuck.
                self.breakaway = self._push
            if moving and not self.decided and self._probe >= c.probe_ticks:
                self._decide()

        obs = CoordObs(
            share=self._share,
            push_total=self._push,
            breakaway=self.breakaway,
            mass_est=self._mass,
            load={p: float(max(-per_robot[p][1], 0.0)) for p in ROBOTS},
            hand=dict(self._f),
            peak_hand=max(self._f.values()) if self._f else 0.0,
            skew=self._skew,
            skew_rate=self._skew_rate,
            speed=self._speed,
            moving=moving,
        )
        if engaged:
            self._probe += 1
            if not self.decided:
                self._reflex_check(obs)
        return obs

    # ------------------------------------------------------------ decision
    def _reflex_check(self, obs: CoordObs) -> None:
        c = self.cfg
        if self._since_touch <= c.reflex_grace:
            return
        if obs.peak_hand > c.reflex_margin * c.arm_capacity:
            self._reflex += 1
        else:
            self._reflex = 0
        if self._reflex >= c.reflex_ticks:
            self.go = False
            self.go_reason = (
                f"{obs.peak_hand:.0f}N/hand, well over the {c.arm_capacity:.0f}N rating"
            )
            self.decided = True
        elif self._probe >= c.probe_ticks and not obs.moving:
            # Leaned on it for the whole probe and it never budged.
            per_hand = self.breakaway / 4.0
            if per_hand > c.arm_capacity:
                self.go = False
                self.go_reason = (
                    f"needs {self.breakaway:.0f}N to break out "
                    f"({per_hand:.0f}N/hand vs {c.arm_capacity:.0f}N)"
                )
                self.decided = True

    def _decide(self) -> None:
        c = self.cfg
        per_hand = max(self.breakaway, self._push) / 4.0
        if per_hand > c.arm_capacity:
            self.go = False
            self.go_reason = f"{per_hand:.0f}N/hand over {c.arm_capacity:.0f}N rating"
        else:
            self.go = True
            self.go_reason = (
                f"broke out at {self.breakaway:.0f}N, {per_hand:.0f}N/hand"
                + (f", ~{self._mass:.0f}kg" if self._mass > 1.0 else "")
            )
        self.decided = True

    # ------------------------------------------------------------ regulate
    def square_command(self, obs: CoordObs) -> float:
        """Differential advance (m). Positive holds robot A's hands back."""
        c = self.cfg
        u = c.kp_skew * obs.skew + c.kd_skew * obs.skew_rate
        return float(np.clip(u, -c.max_skew_cmd, c.max_skew_cmd))

    def press_trim(self, obs: CoordObs, prefix: str) -> float:
        """Press-depth trim (m) that evens out the push between robots."""
        c = self.cfg
        err = obs.share - 0.5
        sign = -1.0 if prefix == A else 1.0
        return float(np.clip(sign * c.kp_share * err * 2.0,
                             -c.max_press_trim, c.max_press_trim))

    def com_bias(self, obs: CoordObs, prefix: str) -> np.ndarray:
        """Lean into the push, in proportion to how hard it is being pushed.

        Both robots face downhill, so bracing means shifting the stance
        toward -y -- the opposite of the carry task, where they leaned away
        from a load they were holding up.
        """
        c = self.cfg
        frac = np.clip(obs.load.get(prefix, 0.0) / c.com_lean_ref, 0.0, 1.4)
        return np.array([0.0, -c.com_lean * float(frac)])

    def pace(self, max_gap: float) -> float:
        c = self.cfg
        p = 1.0 / (1.0 + max(0.0, max_gap - c.pace_tol) / c.pace_tol)
        return float(np.clip(p, c.pace_min, 1.0))

    # -------------------------------------------------------------- safety
    def abort_reason(
        self, model: mujoco.MjModel, data: mujoco.MjData, obs: CoordObs
    ) -> str:
        c = self.cfg
        if self.decided and not self.go:
            return f"no-go: {self.go_reason}"
        if abs(obs.skew) > c.max_skew:
            return f"trunk skewed {np.degrees(obs.skew):.0f}deg"
        for p in ROBOTS:
            if self.cp_fn is not None:
                if self.cp_fn(data, p) < c.min_cp_margin:
                    self._cp_bad[p] += 1
                else:
                    self._cp_bad[p] = 0
                if self._cp_bad[p] >= c.cp_ticks:
                    return f"robot {p[0]} losing balance"
            if self.ix.sensor(data, f"{p}torso_zaxis")[2] < c.min_torso_up:
                return f"robot {p[0]} attitude"
        return ""
