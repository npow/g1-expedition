"""Runs a trail-clearing mission: scene, physics, controller, disturbances.

This is the object both the demo and the RL environment sit on, so the policy
trains against exactly the system that gets demonstrated.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .controller import MissionConfig, RollController, Telemetry
from .coordinator import CoordinatorConfig
from .gains import apply_gains
from .indexing import build_index
from .kinematics import IKWeights
from .poses import reset as reset_scene, settle, stagger_pose
from .scene import ROBOTS, SceneConfig, lip_y, write_scene


@dataclass
class Disturbance:
    """What the mountain does while the team is working.

    Defaults are a calm day. Everything is keyed off the mission phase so a
    gust lands while the robots are actually braced against a moving log,
    which is when it matters.
    """

    wind_mean: float = 0.0
    wind_gust: float = 0.0
    gust_hz: float = 0.35
    gust_dir: tuple[float, float] = (1.0, 0.0)
    gust_phases: tuple[str, ...] = ("PROBE", "ROLL", "FOLLOW")

    push_force: float = 0.0
    push_at: float = 7.0
    push_dur: float = 0.35
    push_robot: str = "A_"
    push_dir: tuple[float, float, float] = (1.0, 0.0, 0.0)

    ice_friction: float = 0.0     # >0 replaces tread friction at ice_at
    ice_at: float = 0.0

    seed: int = 0

    def gust_active(self, phase: str) -> bool:
        return bool(self.wind_gust) and phase in self.gust_phases


@dataclass
class MissionResult:
    success: bool
    travel: float          # m the trunk moved downhill
    over_edge: bool        # did the fall line take it
    final_y: float
    final_z: float
    max_skew: float        # deg
    peak_push: float       # N
    breakaway: float       # N
    max_hand_force: float
    robot_fell: bool
    min_torso_up: float
    duration: float
    aborted: bool
    abort_reason: str
    events: list
    frames: int


class Mission:
    def __init__(
        self,
        scene: SceneConfig | None = None,
        mission: MissionConfig | None = None,
        coord: CoordinatorConfig | None = None,
        ik: IKWeights | None = None,
        disturbance: Disturbance | None = None,
    ):
        self.scfg = scene or SceneConfig()
        self.path = write_scene(self.scfg)
        self.model = mujoco.MjModel.from_xml_path(self.path)
        # See scene.py: setting these in the MJCF collides with the attached
        # robot's own values and MuJoCo warns on every load.
        self.model.opt.iterations = self.scfg.iterations
        self.model.opt.ls_iterations = self.scfg.ls_iterations
        self.data = mujoco.MjData(self.model)
        self.ix = build_index(self.model)
        apply_gains(self.model, self.ix, self.scfg.gain_scale)
        self.mcfg = mission or MissionConfig()
        self.dist = disturbance or Disturbance()
        self.ctrl = RollController(self.model, self.ix, self.scfg, self.mcfg, coord, ik)
        self._base_friction = self.model.pair_friction.copy()
        self.reset()

    # ---------------------------------------------------------------- setup
    def reset(self, jitter: float = 0.0, seed: int | None = None) -> None:
        rng = np.random.default_rng(self.dist.seed if seed is None else seed)
        self.model.pair_friction[:] = self._base_friction
        pose = stagger_pose(self.scfg.stagger) if self.scfg.stagger else None
        reset_scene(self.model, self.data, self.ix, self.scfg, pose=pose,
                    jitter=jitter, rng=rng)
        settle(self.model, self.data, 0.4)
        self.ctrl.reset_state()
        self.ctrl.bind(self.data)
        self._rng = rng
        self._iced = False
        self.stats = {
            "travel": 0.0, "max_skew": 0.0, "peak_push": 0.0,
            "max_hand": 0.0, "frames": 0, "min_up": 1.0,
        }

    # ------------------------------------------------------- disturbances
    def _apply_disturbance(self, tele: Telemetry) -> None:
        d = self.dist
        self.data.xfrc_applied[:] = 0.0

        if d.wind_mean or d.gust_active(tele.phase):
            amp = d.wind_mean
            if tele.phase in d.gust_phases:
                amp += d.wind_gust * (
                    0.55 + 0.45 * np.sin(2 * np.pi * d.gust_hz * tele.t)
                ) * (0.8 + 0.4 * self._rng.random())
            v = np.array([d.gust_dir[0], d.gust_dir[1], 0.0], dtype=float)
            n = np.linalg.norm(v)
            if n > 1e-9 and amp:
                v = v / n * amp
                for p in ROBOTS:
                    self.data.xfrc_applied[self.ix.robot(p).torso_body, :3] += v * 0.5

        if d.push_force and d.push_at <= tele.t < d.push_at + d.push_dur:
            v = np.array(d.push_dir, dtype=float)
            v = v / max(np.linalg.norm(v), 1e-9) * d.push_force
            self.data.xfrc_applied[self.ix.robot(d.push_robot).torso_body, :3] += v

        if d.ice_friction > 0.0 and not self._iced and tele.t >= d.ice_at:
            f = self.model.pair_friction
            gset = set(int(g) for g in self.ix.ground_geoms)
            for i in range(self.model.npair):
                if int(self.model.pair_geom1[i]) in gset or int(self.model.pair_geom2[i]) in gset:
                    f[i, 0] = f[i, 1] = d.ice_friction
            self._iced = True
            self.ctrl.events.append((tele.t, f"verglas: mu -> {d.ice_friction:.2f}"))

    # -------------------------------------------------------------- stepping
    def step(self, residual: np.ndarray | None = None) -> Telemetry:
        tele = self.ctrl.control(self.data, residual)
        self._apply_disturbance(tele)
        for _ in range(self.ctrl.n_sub):
            mujoco.mj_step(self.model, self.data)

        s = self.stats
        s["travel"] = max(s["travel"], tele.log_travel)
        if tele.engaged:
            s["max_skew"] = max(s["max_skew"], abs(tele.skew_deg))
        s["peak_push"] = max(s["peak_push"], tele.push_total)
        s["max_hand"] = max(s["max_hand"], max(tele.hand_force.values(), default=0.0))
        s["min_up"] = min(
            s["min_up"],
            min(float(self.ix.sensor(self.data, f"{p}torso_zaxis")[2]) for p in ROBOTS),
        )
        s["frames"] += 1
        return tele

    def run(self, max_seconds: float | None = None):
        limit = max_seconds or (self.ctrl.duration + 1.0)
        while not self.ctrl.done and self.ctrl.t < limit:
            yield self.step()

    def result(self) -> MissionResult:
        s = self.stats
        c = self.ctrl
        lp = self.data.xpos[self.ix.log_body]
        over = bool(lp[1] < lip_y(self.scfg) - 0.15)
        return MissionResult(
            # The trail is clear when the trunk is past the edge and heading
            # down; nothing else counts, least of all how far the hands moved.
            # The trail is clear when the trunk is past the edge and heading
            # down, and the crew is still standing. Both matter: a team that
            # clears the path and then falls off it has not succeeded.
            success=bool(over and not c.aborted and s["min_up"] > 0.35),
            travel=s["travel"],
            over_edge=over,
            final_y=float(lp[1]),
            final_z=float(lp[2]),
            max_skew=s["max_skew"],
            peak_push=s["peak_push"],
            breakaway=c.coord.breakaway,
            max_hand_force=s["max_hand"],
            robot_fell=bool(s["min_up"] <= 0.35),
            min_torso_up=s["min_up"],
            duration=c.t,
            aborted=c.aborted,
            abort_reason=c.abort_reason,
            events=list(c.events),
            frames=s["frames"],
        )
