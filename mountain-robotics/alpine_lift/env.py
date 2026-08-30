"""Residual-RL environment for the trail-clearing roll.

The policy does not learn to shove a log from scratch. It learns a small
correction on a controller that already works, which keeps a working
baseline to fall back on and makes the comparison -- with and without the
policy, same seed, same trunk, same gust -- a controlled experiment.

This task is much better posed for learning than the carry it replaced.
There, episode return was dominated by which disturbance was drawn and a
few centimetres of palm correction barely registered. Here the reward is
*progress*: how much closer the trunk got to the edge. The residual's press
depth and differential advance change that directly and immediately, so the
advantage estimate is measuring the policy rather than the weather.

Action (10-d in [-1, 1], per robot: press depth, advance, CoM x, CoM y,
squat depth) -- the same handles the coordinator uses, so the policy argues
with the controller in its own units instead of reaching past it to the
joints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controller import PHASE_NAMES, MissionConfig
from .coordinator import CoordinatorConfig
from .mission import Disturbance, Mission
from .scene import A, B, ROBOTS, SceneConfig, lip_y

N_PHASE = len(PHASE_NAMES)
PER_ROBOT = 13
ACT_DIM = 10
OBS_DIM = N_PHASE + 1 + 6 + 2 * PER_ROBOT + 4 + 1


@dataclass
class DomainRandomization:
    """What varies between episodes.

    Ranges chosen so the scripted controller sits well short of perfect --
    enough successes to bootstrap from, enough failures to learn from. Mass
    is the dominant axis: the team clears a hollowed 35 kg trunk and cannot
    shift a solid 150 kg one, so somewhere between is where the residual has
    something to contribute.
    """

    mass: tuple[float, float] = (28.0, 60.0)
    log_y: tuple[float, float] = (-0.38, -0.30)
    friction: tuple[float, float] = (0.70, 1.0)
    roll_friction: tuple[float, float] = (0.002, 0.010)
    wind: tuple[float, float] = (0.0, 20.0)
    push: tuple[float, float] = (0.0, 45.0)
    push_prob: float = 0.30
    ice_prob: float = 0.15
    jitter: float = 0.005


@dataclass
class RewardConfig:
    """Shaping, sized so that working scores better than stopping.

    Carried over from the carry task, where the first reward had a small
    survival bonus against dense penalties: a whole episode summed negative,
    which made quitting early the highest-value action available and sent
    training backwards. Here a step that moves the trunk scores positive,
    and failure is charged for the rest of the episode it threw away.
    """

    progress: float = 26.0     # per metre of trunk travel toward the edge
    cleared: float = 40.0      # terminal, trunk over the edge
    square: float = 1.4        # per rad of yaw off square
    upright: float = 2.0       # per unit of torso-axis deficit
    margin: float = 1.2
    effort: float = 0.05
    alive: float = 0.5
    fell: float = 25.0
    abort: float = 12.0
    horizon: int = 620


class TrailClearEnv:
    """Gymnasium-style env; one policy commands both robots.

    They are a crew, not competitors. The thing being learned is a joint
    correction, and giving each robot its own policy would reintroduce the
    disagreement that makes a two-robot shove go crooked.
    """

    def __init__(
        self,
        randomize: bool = True,
        dr: DomainRandomization | None = None,
        reward: RewardConfig | None = None,
        mission: MissionConfig | None = None,
        seed: int = 0,
        scenery: bool = False,
    ):
        self.dr = dr or DomainRandomization()
        self.rc = reward or RewardConfig()
        self.mcfg = mission or MissionConfig()
        self.randomize = randomize
        self.scenery = scenery
        self.rng = np.random.default_rng(seed)
        self._mi: Mission | None = None
        self._cache: dict[tuple, Mission] = {}
        self.observation_dim = OBS_DIM
        self.action_dim = ACT_DIM

    # ------------------------------------------------------------- episodes
    def _sample(self) -> tuple[SceneConfig, Disturbance]:
        d, r = self.dr, self.rng
        if not self.randomize:
            return SceneConfig(scenery=self.scenery), Disturbance()
        scfg = SceneConfig(
            log_mass=float(r.uniform(*d.mass)),
            log_y=float(r.uniform(*d.log_y)),
            ground_friction=float(r.uniform(*d.friction)),
            log_roll_friction=float(r.uniform(*d.roll_friction)),
            scenery=self.scenery,
        )
        dist = Disturbance(
            wind_gust=float(r.uniform(*d.wind)),
            gust_hz=float(r.uniform(0.2, 0.6)),
            gust_dir=(float(r.uniform(-1, 1)), float(r.uniform(-1, 1))),
            push_force=float(r.uniform(*d.push)) if r.random() < d.push_prob else 0.0,
            push_at=float(r.uniform(4.0, 8.0)),
            push_robot=A if r.random() < 0.5 else B,
            push_dir=(float(r.uniform(-1, 1)), float(r.uniform(-1, 1)), 0.0),
            ice_friction=float(r.uniform(0.30, 0.5)) if r.random() < d.ice_prob else 0.0,
            ice_at=float(r.uniform(3.0, 6.0)),
            seed=int(r.integers(1 << 30)),
        )
        return scfg, dist

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        scfg, dist = self._sample()

        # Recompiling the MJCF costs ~0.4 s, which would dominate a 2 s
        # episode. Only fields that change the XML key the cache; mass,
        # friction and the trunk's resting station are patched into the
        # compiled model instead.
        key = (round(scfg.log_radius, 4), round(scfg.tread_half_width, 3),
               scfg.scenery, round(scfg.robot_y, 3))
        mi = self._cache.get(key)
        if mi is None:
            mi = Mission(scene=scfg, mission=self.mcfg, disturbance=dist)
            self._cache[key] = mi
        mi.scfg = scfg
        mi.dist = dist
        mi.ctrl.scfg = scfg
        mi.ctrl.coord.scfg = scfg
        self._patch(mi, scfg)
        mi.reset(jitter=self.dr.jitter if self.randomize else 0.0, seed=dist.seed)
        self._mi = mi
        self.steps = 0
        self._prev_to_edge = self._to_edge()
        return self._obs(mi.ctrl.tele)

    def _patch(self, mi: Mission, scfg: SceneConfig) -> None:
        m = mi.model
        bid = mi.ix.log_body
        old = float(m.body_mass[bid])
        if old > 1e-9:
            m.body_inertia[bid] *= scfg.log_mass / old
        m.body_mass[bid] = scfg.log_mass
        gset = set(int(g) for g in mi.ix.ground_geoms)
        for i in range(m.npair):
            g1, g2 = int(m.pair_geom1[i]), int(m.pair_geom2[i])
            if g1 in gset or g2 in gset:
                m.pair_friction[i, 0] = m.pair_friction[i, 1] = scfg.ground_friction
                if mi.ix.log_geom in (g1, g2):
                    m.pair_friction[i, 3] = m.pair_friction[i, 4] = scfg.log_roll_friction
        mi._base_friction = m.pair_friction.copy()

    def _to_edge(self) -> float:
        mi = self._mi
        return float(mi.data.xpos[mi.ix.log_body][1] - lip_y(mi.scfg))

    # ---------------------------------------------------------- observation
    def _obs(self, t) -> np.ndarray:
        mi = self._mi
        ix, data = mi.ix, mi.data
        o = np.zeros(OBS_DIM, dtype=np.float32)
        i = 0
        o[i + min(t.phase_idx if t.phase_idx >= 0 else N_PHASE - 1, N_PHASE - 1)] = 1.0
        i += N_PHASE
        o[i] = t.phase_frac; i += 1

        o[i] = np.clip(t.log_travel * 2.0, -3, 3); i += 1
        o[i] = np.clip(t.to_edge * 2.0, -3, 3); i += 1
        o[i] = np.clip(t.speed, -4, 4); i += 1
        o[i] = np.clip(np.radians(t.skew_deg) * 4.0, -3, 3); i += 1
        o[i] = np.clip(mi.ctrl.coord._skew_rate, -3, 3); i += 1
        o[i] = np.clip((t.log_z - mi.scfg.log_radius) * 4.0, -3, 3); i += 1

        for p in ROBOTS:
            r = ix.robot(p)
            o[i] = np.clip(t.robot_load.get(p, 0.0) / 120.0, -3, 3); i += 1
            o[i] = np.clip(t.cp_margin.get(p, 0.0) * 8.0, -3, 3); i += 1
            b = t.balance.get(p, {})
            o[i] = np.clip(b.get("cp_fwd", 0.0) * 8.0, -3, 3); i += 1
            o[i] = np.clip(b.get("cp_lat", 0.0) * 8.0, -3, 3); i += 1
            o[i:i + 3] = ix.sensor(data, f"{p}torso_zaxis"); i += 3
            o[i:i + 3] = np.clip(ix.sensor(data, f"{p}gyro") * 0.2, -4, 4); i += 3
            o[i] = np.clip(t.palm_gap.get(f"{p}left", 0.0) * 5.0, 0, 4); i += 1
            o[i] = np.clip(t.palm_gap.get(f"{p}right", 0.0) * 5.0, 0, 4); i += 1
            o[i] = np.clip((data.xpos[r.base_body][2] - 0.65) * 5.0, -3, 3); i += 1

        for k in ("A_left", "A_right", "B_left", "B_right"):
            o[i] = np.clip(t.hand_force.get(k, 0.0) / 150.0, 0, 4); i += 1
        o[i] = float(t.share) - 0.5; i += 1
        return np.nan_to_num(o[:OBS_DIM], nan=0.0, posinf=0.0, neginf=0.0)

    # --------------------------------------------------------------- stepping
    def step(self, action: np.ndarray):
        mi, rc = self._mi, self.rc
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        before_abort = mi.ctrl.aborted
        tele = mi.step(residual=a)
        self.steps += 1

        # Dense progress: how much closer the trunk got to going over. This
        # is the signal the carry task never had -- it responds to the
        # action on the same tick.
        now = self._to_edge()
        r = rc.alive + rc.progress * (self._prev_to_edge - now)
        self._prev_to_edge = now

        r -= rc.square * min(abs(np.radians(tele.skew_deg)), 0.6)
        ups = [float(mi.ix.sensor(mi.data, f"{p}torso_zaxis")[2]) for p in ROBOTS]
        r -= rc.upright * max(0.0, 0.60 - min(ups))
        worst = min(tele.cp_margin.values()) if tele.cp_margin else 0.0
        r += rc.margin * min(worst, 0.04)
        r -= rc.effort * float(np.mean(a * a))

        if mi.ctrl.aborted and not before_abort:
            left = max(0, rc.horizon - self.steps)
            r -= rc.abort + rc.alive * left

        done = mi.ctrl.done
        if done:
            res = mi.result()
            if res.over_edge:
                r += rc.cleared
            if res.robot_fell:
                r -= rc.fell
        return self._obs(tele), float(r), bool(done), {"tele": tele}

    @property
    def mission(self) -> Mission:
        return self._mi


# Back-compat alias: the training script imports this name.
AlpineLiftEnv = TrailClearEnv
