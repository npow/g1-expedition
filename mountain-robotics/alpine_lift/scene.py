"""Procedural MJCF for the trail-clearing scene.

A conifer trunk has come down across a cut path on a Himalayan approach.
Two Unitree G1s stand on the uphill side and roll it over the outer edge,
where the fall line takes it away down the mountainside.

Why rolling and not carrying
----------------------------
A 2.2 m trunk 36 cm thick weighs about a hundred kilos, and no pair of G1s
is going to pick that up -- a G1 masses 35 kg itself. But rolling it does
not have to beat gravity, only rolling resistance, and that is a small
fraction of the weight. It is also simply what a trail crew does: you get
low on the uphill side and roll the thing off the edge. Building the task
around a load the robots could lift would have meant a log light enough to
be dishonest about, so the task changed instead of the log.

The terrain is a real cut trail: a level tread with a bank rising on the
uphill side and the fall line dropping away downhill. That geometry is not
decoration -- the bank is what the robots brace against, and the drop is
what finishes the job once the log crosses the outer edge.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import asdict, dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_GEN_DIR = os.environ.get(
    "ALPINE_GEN_DIR", os.path.join(_ROOT, "assets", "generated")
)
_G1_PATH = os.path.join(_ROOT, "assets", "g1", "g1_mjx.xml")

A, B = "A_", "B_"
ROBOTS = (A, B)

# --------------------------------------------------------------------------
# Contact pairs. g1_mjx.xml sets contype/conaffinity to 0 on every geom and
# relies on explicit <pair> elements, so composing two robots into one scene
# means re-emitting the pair list per robot. Transcribed from menagerie's
# scene_mjx.xml, the collision set validated on hardware in MuJoCo Playground.
# --------------------------------------------------------------------------
SELF_PAIRS: tuple[tuple[str, str, int], ...] = (
    ("left_foot_box_collision", "right_foot_box_collision", 1),
    ("left_foot_box_collision", "right_shin_collision", 1),
    ("right_foot_box_collision", "left_shin_collision", 1),
    ("left_foot_box_collision", "right_linkage_brace_collision", 1),
    ("right_foot_box_collision", "left_linkage_brace_collision", 1),
    ("left_hand_collision", "left_hip_collision", 1),
    ("right_hand_collision", "right_hip_collision", 1),
    ("left_hand_collision", "left_thigh_collision", 1),
    ("right_hand_collision", "right_thigh_collision", 1),
    ("left_shin_collision", "right_shin_collision", 1),
    ("torso_collision", "left_shoulder_yaw_collision", 1),
    ("torso_collision", "right_shoulder_yaw_collision", 1),
    ("torso_collision", "left_elbow_yaw_collision", 1),
    ("torso_collision", "right_elbow_yaw_collision", 1),
    ("torso_collision", "left_wrist_collision", 1),
    ("torso_collision", "right_wrist_collision", 1),
    ("torso_collision", "left_hand_collision", 1),
    ("torso_collision", "right_hand_collision", 1),
    ("left_thigh_collision", "right_thigh_collision", 1),
    ("left_shin_collision", "right_thigh_collision", 1),
    ("right_shin_collision", "left_thigh_collision", 1),
    ("left_shin_collision", "right_hip_collision", 1),
    ("right_shin_collision", "left_hip_collision", 1),
    ("left_hip_collision", "right_thigh_collision", 1),
    ("right_hip_collision", "left_thigh_collision", 1),
    ("left_hand_collision", "right_hand_collision", 1),
)

FOOT_GEOMS: tuple[str, ...] = (
    "left_foot1_collision", "left_foot2_collision", "left_foot3_collision",
    "right_foot1_collision", "right_foot2_collision", "right_foot3_collision",
)
BODY_GROUND_GEOMS: tuple[str, ...] = (
    "left_hand_collision", "right_hand_collision",
    "left_shoulder_yaw_collision", "right_shoulder_yaw_collision",
    "left_elbow_yaw_collision", "right_elbow_yaw_collision",
    "left_wrist_collision", "right_wrist_collision",
    "left_hip_collision", "right_hip_collision",
    "left_thigh_collision", "right_thigh_collision",
    "left_shin_collision", "right_shin_collision",
    "pelvis_collision", "torso_collision", "head_collision",
)
# Geoms allowed to touch the trunk. The hands do the work; the rest is here
# so a robot that leans into the log is stopped by it rather than passing
# through it.
LOG_CONTACT_GEOMS: tuple[str, ...] = (
    "left_hand_collision", "right_hand_collision",
    "left_wrist_collision", "right_wrist_collision",
    "left_elbow_yaw_collision", "right_elbow_yaw_collision",
    "left_thigh_collision", "right_thigh_collision",
    "left_shin_collision", "right_shin_collision",
    "left_foot_box_collision", "right_foot_box_collision",
    "torso_collision", "pelvis_collision", "head_collision",
)
CROSS_ROBOT_GEOMS: tuple[str, ...] = (
    "left_hand_collision", "right_hand_collision",
    "left_wrist_collision", "right_wrist_collision",
    "left_elbow_yaw_collision", "right_elbow_yaw_collision",
    "torso_collision", "head_collision",
    "left_foot_box_collision", "right_foot_box_collision",
)
FALL_GEOMS: tuple[str, ...] = (
    "pelvis_collision", "torso_collision", "head_collision",
    "left_hip_collision", "right_hip_collision",
    "left_shoulder_yaw_collision", "right_shoulder_yaw_collision",
)
GROUND_GEOMS: tuple[str, ...] = ("tread", "bank", "fall_line")
CHOCK_GEOMS: tuple[str, ...] = ("chock0", "chock1")


@dataclass
class SceneConfig:
    """Every physical knob of the trail-clearing scenario."""

    # --- the trunk ---------------------------------------------------------
    # 2.2 m of conifer, 44 cm through, rot-hollowed: a 2.7 cm shell at
    # 450 kg/m^3 is 35 kg, and standing deadwood really does hollow out like
    # this. The diameter is what the robots can get a hand on; the mass is
    # what they can actually shift.
    #
    # Solid, the same trunk is 150 kg, and the measured answer there is that
    # this team cannot clear it -- they move it 0.16 m against the 0.27 m it
    # needs. That is not a tuning failure, it is the force a standing
    # humanoid can produce at arm's length, and it is why the go/no-go check
    # exists. Set log_mass=150 to see the team decline it.
    log_radius: float = 0.22
    log_length: float = 2.2
    log_mass: float = 35.0
    log_y: float = -0.34              # where it lies across the tread
    log_com_offset: float = 0.0       # m along the trunk, for an off-centre CoM
    log_roll_friction: float = 0.004  # rolling resistance at the tread
    # Chocks are modelled but off. Measured: a 150 kg trunk needs about
    # 950 N at the axis to climb even a 3.5 cm rock, which two G1s are never
    # going to produce -- so a chocked log is one this team correctly
    # declines. On a level tread the trunk rests in neutral equilibrium and
    # rolls for a fraction of that, which is the regime the task lives in.
    # Set chock_height > 0 to make it a no-go scenario.
    chock_height: float = 0.0
    chock_x: tuple[float, float] = (-0.62, 0.62)
    log_condim: int = 6               # 6 keeps rolling resistance; 3 drops it

    # --- the trail ---------------------------------------------------------
    tread_half_width: float = 0.55    # walking surface, |y| (1.1 m path)
    # Real trail tread is built with a few degrees of outslope so water sheds
    # off the downhill edge instead of running along it and gullying the
    # path. It also means a round log lying on it is already on the verge of
    # rolling, held only by rolling resistance -- so the crew's job is to
    # break it loose, not to carry it. That is a property of how trails are
    # built, not a convenience added to make the task work.
    tread_outslope: float = 0.0       # rad; the tread is cut level here
    trail_length: float = 16.0
    bank_angle: float = 0.68          # rad, cut bank rising on the uphill side
    bank_run: float = 7.0
    fall_angle: float = 0.62          # rad, the drop on the outer edge
    fall_run: float = 60.0
    ground_friction: float = 0.95     # 0.95 dry dirt, ~0.3 verglas

    # --- the team ----------------------------------------------------------
    robot_x: float = 0.50             # |x| of each robot along the trail
    robot_y: float = 0.20             # uphill of the trunk
    grip_dx: float = 0.24             # hand separation along the trunk
    # Stagger is implemented (poses.stagger_pose) and left at zero. Measured:
    # a symmetric stagger doubles fore-aft support but puts the forward boot
    # under the trunk, and a rearward-only stagger is not statically
    # consistent -- the unmoved foot slides and the robot pivots. Both are
    # worse than a parallel stance here. The finding is kept; the feature is
    # off.
    stagger: float = 0.0              # rad of braced stance stagger
    grip_phi: float = 0.35            # rad from vertical; near the top, where
                                      # a horizontal push has the most leverage

    # --- disturbances ------------------------------------------------------
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gravity: float = -9.81

    # --- actuation / solver ------------------------------------------------
    gain_scale: float = 1.0
    timestep: float = 0.004
    iterations: int = 10
    ls_iterations: int = 15
    scenery: bool = True

    def key(self) -> str:
        blob = repr(sorted(asdict(self).items())).encode()
        return hashlib.sha1(blob).hexdigest()[:10]


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def tread_z(cfg: SceneConfig, y: float) -> float:
    """Height of the outsloped tread surface at a given y."""
    return y * math.tan(cfg.tread_outslope)


def log_rest_z(cfg: SceneConfig) -> float:
    """Height of the trunk's axis when it is lying on the tread."""
    return tread_z(cfg, cfg.log_y) + cfg.log_radius / math.cos(cfg.tread_outslope)


def grip_point(cfg: SceneConfig, prefix: str, side: str, axis_y: float,
               axis_z: float, phi: float | None = None):
    """World target for one palm, on the trunk's uphill flank.

    Deliberately expressed against the trunk's *axis* and a world-frame
    azimuth rather than as a site fixed to the body. The log rotates -- that
    is the entire point of the task -- and a body-fixed grip site would drag
    the hands around with it. An angle measured from vertical toward the
    uphill side stays put while the trunk spins underneath it, which is also
    how a person's hands behave.
    """
    phi = cfg.grip_phi if phi is None else phi
    x = grip_x(cfg, prefix, side)
    r = cfg.log_radius
    return (x, axis_y + r * math.sin(phi), axis_z + r * math.cos(phi))


def grip_x(cfg: SceneConfig, prefix: str, side: str) -> float:
    """Where along the trunk this hand works.

    Both robots face downhill, so for both of them the left hand is the one
    on the +x side.
    """
    base = -cfg.robot_x if prefix == A else cfg.robot_x
    return base + (cfg.grip_dx if side == "left" else -cfg.grip_dx)


def lip_y(cfg: SceneConfig) -> float:
    """The outer edge of the tread -- once past this the fall line has it."""
    return -cfg.tread_half_width


# --------------------------------------------------------------------------
# XML fragments
# --------------------------------------------------------------------------

def _terrain(cfg: SceneConfig) -> str:
    """A cut trail: outsloped tread, bank rising uphill, fall line dropping.

    All three slabs are boxes rotated about +x, placed from their shared
    edges rather than eyeballed. For a surface at angle ``a``, the up-slope
    direction is (0, cos a, sin a) and the outward normal is
    (0, -sin a, cos a); a slab's centre is its top-face centre pushed half a
    thickness along -normal. Getting that half-thickness term wrong leaves a
    step at the tread edge, and a rolling log jams on it instead of going
    over.
    """
    w = cfg.tread_half_width
    L = cfg.trail_length / 2.0
    fr = cfg.ground_friction
    t = 0.40
    th, ba, fa = cfg.tread_outslope, cfg.bank_angle, cfg.fall_angle

    # Tread: top face through the origin, outsloped toward -y.
    tread_y, tread_z = t * math.sin(th), -t * math.cos(th)
    # Its two edges, where the bank and the fall line have to meet it.
    up_edge = (w * math.cos(th), w * math.sin(th))
    dn_edge = (-w * math.cos(th), -w * math.sin(th))

    hb, hf = cfg.bank_run / 2.0, cfg.fall_run / 2.0
    bank_y = up_edge[0] + hb * math.cos(ba) + t * math.sin(ba)
    bank_z = up_edge[1] + hb * math.sin(ba) - t * math.cos(ba)
    fall_y = dn_edge[0] - hf * math.cos(fa) + t * math.sin(fa)
    fall_z = dn_edge[1] - hf * math.sin(fa) - t * math.cos(fa)

    return f"""
    <geom name="tread" type="box" pos="0 {tread_y:.4f} {tread_z:.4f}"
          size="{L:.3f} {w:.3f} {t:.3f}" euler="{th:.5f} 0 0"
          material="tread" friction="{fr:.4f} 0.02 0.002" condim="3"/>
    <geom name="bank" type="box" pos="0 {bank_y:.4f} {bank_z:.4f}"
          size="{L:.3f} {hb:.3f} {t:.3f}" euler="{ba:.5f} 0 0"
          material="bankrock" friction="{fr:.4f} 0.02 0.002" condim="3"/>
    <geom name="fall_line" type="box" pos="0 {fall_y:.4f} {fall_z:.4f}"
          size="{L:.3f} {hf:.3f} {t:.3f}" euler="{fa:.5f} 0 0"
          material="scree" friction="{cfg.ground_friction * 0.8:.4f} 0.02 0.002" condim="3"/>"""


def _chocks(cfg: SceneConfig) -> str:
    if cfg.chock_height <= 0.0:
        return ""
    h = cfg.chock_height
    y = cfg.log_y - cfg.log_radius * 0.86
    out = []
    for i, x in enumerate(cfg.chock_x):
        out.append(
            f'    <geom name="chock{i}" type="box" pos="{x:.3f} {y:.4f}'
            f' {tread_z(cfg, y) + h / 2:.4f}" size="0.11 0.075 {h / 2:.4f}"'
            f' euler="{cfg.tread_outslope:.4f} 0 {0.3 * (i + 1):.2f}"'
            f' material="granite" friction="{cfg.ground_friction:.3f} 0.02 0.002"'
            f' condim="3"/>')
    return "\n".join(out)


def _log(cfg: SceneConfig) -> str:
    r, hl = cfg.log_radius, cfg.log_length / 2.0
    # Collision is a capsule: analytic contacts, and rolls like a cylinder.
    # The visible trunk is a cylinder plus sawn ends and a couple of sheared
    # limbs, so the silhouette reads as timber rather than a pill.
    return f"""
    <body name="log" pos="0 {cfg.log_y:.4f} {log_rest_z(cfg):.4f}">
      <freejoint name="log_free"/>
      <inertial pos="{cfg.log_com_offset:.4f} 0 0" mass="{cfg.log_mass:.3f}"
                diaginertia="{0.5 * cfg.log_mass * r * r:.4f}
                             {cfg.log_mass * (3 * r * r + cfg.log_length ** 2) / 12:.4f}
                             {cfg.log_mass * (3 * r * r + cfg.log_length ** 2) / 12:.4f}"/>
      <geom name="log_core" type="capsule" fromto="{-hl + r:.4f} 0 0 {hl - r:.4f} 0 0"
            size="{r:.4f}" mass="{cfg.log_mass:.3f}" material="bark" condim="6"
            friction="1.0 0.02 {cfg.log_roll_friction:.5f}"/>
      <geom name="log_shell" type="cylinder" fromto="{-hl:.4f} 0 0 {hl:.4f} 0 0"
            size="{r * 1.005:.4f}" material="bark" mass="0"
            contype="0" conaffinity="0" group="2"/>
      <geom name="log_endA" type="cylinder" fromto="{-hl - 0.006:.4f} 0 0 {-hl + 0.02:.4f} 0 0"
            size="{r * 0.99:.4f}" material="sapwood" mass="0"
            contype="0" conaffinity="0" group="2"/>
      <geom name="log_endB" type="cylinder" fromto="{hl - 0.02:.4f} 0 0 {hl + 0.006:.4f} 0 0"
            size="{r * 0.99:.4f}" material="sapwood" mass="0"
            contype="0" conaffinity="0" group="2"/>
      <geom name="log_limb1" type="capsule"
            fromto="{0.32 * hl:.3f} 0 {r * 0.6:.3f} {0.52 * hl:.3f} {r * 1.5:.3f} {r * 1.7:.3f}"
            size="{r * 0.17:.4f}" material="bark" mass="0"
            contype="0" conaffinity="0" group="2"/>
      <geom name="log_limb2" type="capsule"
            fromto="{-0.45 * hl:.3f} 0 {r * 0.5:.3f} {-0.62 * hl:.3f} {-r * 1.3:.3f} {r * 1.6:.3f}"
            size="{r * 0.14:.4f}" material="bark" mass="0"
            contype="0" conaffinity="0" group="2"/>
      <site name="log_axis" pos="0 0 0" size="0.02" group="4"/>
    </body>"""


def _scenery(cfg: SceneConfig) -> str:
    if not cfg.scenery:
        return ""
    out: list[str] = []
    # Peaks are stacks of rotated boxes: a Himalayan skyline is angular, and
    # ellipsoids read as eggs. Distances are large so the fog does the
    # aerial-perspective work rather than the geometry looking like a wall.
    peaks = [
        (-150, 330, 78.0, 42.0, 0.5), (70, 400, 112.0, 60.0, 0.2),
        (240, 300, 62.0, 35.0, 1.1), (-300, 260, 58.0, 33.0, 0.8),
        (-40, 500, 134.0, 76.0, 0.35), (-140, 430, 92.0, 50.0, 0.9),
        (170, 460, 84.0, 46.0, 0.15), (340, 380, 70.0, 40.0, 0.6),
        (20, 250, 46.0, 27.0, 0.75), (-80, 230, 38.0, 22.0, 1.4),
        (120, 225, 34.0, 20.0, 0.3), (-215, 350, 64.0, 36.0, 1.2),
    ]
    for i, (px, py, h, r, yaw) in enumerate(peaks):
        out.append(
            f'    <geom name="peak{i}a" type="box" pos="{px} {py} {h * 0.34:.2f}"'
            f' size="{r:.2f} {r * 0.82:.2f} {h * 0.42:.2f}" euler="0 0 {yaw:.2f}"'
            f' material="rockdark" contype="0" conaffinity="0" group="2"/>')
        out.append(
            f'    <geom name="peak{i}b" type="box" pos="{px} {py} {h * 0.66:.2f}"'
            f' size="{r * 0.58:.2f} {r * 0.46:.2f} {h * 0.40:.2f}"'
            f' euler="0.06 0.09 {yaw + 0.7:.2f}" material="rockface"'
            f' contype="0" conaffinity="0" group="2"/>')
        out.append(
            f'    <geom name="snowcap{i}" type="box" pos="{px} {py} {h * 0.93:.2f}"'
            f' size="{r * 0.30:.2f} {r * 0.24:.2f} {h * 0.20:.2f}"'
            f' euler="0.10 0.14 {yaw + 0.35:.2f}" material="snow"'
            f' contype="0" conaffinity="0" group="2"/>')
    for i, (px, py, w, h) in enumerate(
        [(-120, 175, 52.0, 16.0), (95, 190, 62.0, 19.0), (225, 165, 45.0, 13.0),
         (-255, 155, 42.0, 12.0), (-20, 150, 36.0, 10.0)]
    ):
        out.append(
            f'    <geom name="ridge{i}" type="box" pos="{px} {py} {h * 0.3:.2f}"'
            f' size="{w:.2f} {w * 0.5:.2f} {h:.2f}" euler="0 0 {0.2 * i:.2f}"'
            f' material="rockdark" contype="0" conaffinity="0" group="2"/>')

    # Trailside detail: boulders set into the cut bank, loose rock on the
    # tread, and a few stumps downslope so the drop reads as forested.
    w = cfg.tread_half_width
    for i, (px, dy, sz, yaw) in enumerate(
        [(-3.1, 0.35, 0.30, 0.6), (2.4, 0.55, 0.24, 1.9), (-1.6, 0.75, 0.19, 0.4),
         (4.0, 0.30, 0.34, 2.6), (0.9, 0.85, 0.16, 1.2), (-4.6, 0.60, 0.27, 0.9)]
    ):
        py = w + dy * math.cos(cfg.bank_angle)
        pz = dy * math.sin(cfg.bank_angle)
        out.append(
            f'    <geom name="bankrock{i}" type="box" pos="{px:.2f} {py:.2f} {pz:.2f}"'
            f' size="{sz:.3f} {sz * 0.8:.3f} {sz * 0.75:.3f}"'
            f' euler="{cfg.bank_angle:.3f} 0.2 {yaw:.2f}" material="granite"'
            f' contype="0" conaffinity="0" group="2"/>')
    for i, (px, py, sz, yaw) in enumerate(
        [(-2.6, 0.42, 0.09, 0.5), (3.2, -0.55, 0.07, 1.7), (1.5, 0.66, 0.06, 2.3),
         (-4.2, -0.30, 0.10, 0.8), (5.1, 0.20, 0.08, 1.1)]
    ):
        out.append(
            f'    <geom name="scree{i}" type="box" pos="{px:.2f} {py:.2f} {sz * 0.5:.3f}"'
            f' size="{sz:.3f} {sz * 0.8:.3f} {sz * 0.6:.3f}" euler="0.1 0.2 {yaw:.2f}"'
            f' material="granite" contype="0" conaffinity="0" group="2"/>')
    for i, (px, dy, h) in enumerate(
        [(-2.2, 2.6, 0.55), (3.6, 4.2, 0.42), (0.4, 6.5, 0.62), (-5.0, 3.4, 0.48)]
    ):
        py = -w - dy * math.cos(cfg.fall_angle)
        pz = -dy * math.sin(cfg.fall_angle)
        out.append(
            f'    <geom name="stump{i}" type="cylinder"'
            f' pos="{px:.2f} {py:.2f} {pz + h / 2:.2f}" size="0.16 {h / 2:.2f}"'
            f' euler="{-cfg.fall_angle * 0.5:.3f} 0 0" material="bark"'
            f' contype="0" conaffinity="0" group="2"/>')
    return "\n".join(out)


def _pairs(cfg: SceneConfig) -> str:
    fr = f"{cfg.ground_friction:.4f}"
    lines: list[str] = []
    for p in ROBOTS:
        lines.append(f"    <!-- ===== robot {p[0]} ===== -->")
        for g in FOOT_GEOMS:
            for terrain in GROUND_GEOMS:
                lines.append(
                    f'    <pair name="{p}{g}_{terrain}" geom1="{p}{g}" geom2="{terrain}"'
                    f' solref="0.008 1" friction="{fr} {fr} 0.005" condim="3"/>')
        for g in BODY_GROUND_GEOMS:
            for terrain in GROUND_GEOMS:
                lines.append(
                    f'    <pair name="{p}{g}_{terrain}" geom1="{p}{g}" geom2="{terrain}"'
                    f' friction="{fr} {fr} 0.005" condim="3"/>')
        for g1, g2, cd in SELF_PAIRS:
            lines.append(
                f'    <pair name="{p}{g1}__{g2}" geom1="{p}{g1}" geom2="{p}{g2}" condim="{cd}"/>')
        for g in LOG_CONTACT_GEOMS:
            # High friction and condim 4 at the hands: the whole task is
            # shear between a palm and bark, and a frictionless palm just
            # slides up over the trunk without turning it.
            cd = 4 if "hand" in g else 3
            lines.append(
                f'    <pair name="{p}{g}_log" geom1="{p}{g}" geom2="log_core"'
                f' condim="{cd}" friction="1.2 1.2 0.01 0.01 0.01"'
                # A palm is not a steel plate. Softening this contact keeps
                # the force loop stable: on a rigid pair, a millimetre of
                # tracking error is tens of newtons.
                f' solref="0.04 1"/>')
    lines.append("    <!-- ===== robot A vs robot B ===== -->")
    for g1 in CROSS_ROBOT_GEOMS:
        for g2 in CROSS_ROBOT_GEOMS:
            lines.append(
                f'    <pair name="AB_{g1}__{g2}" geom1="{A}{g1}" geom2="{B}{g2}" condim="1"/>')
    lines.append("    <!-- ===== trunk vs the chocks holding it ===== -->")
    if cfg.chock_height > 0.0:
        for ch in CHOCK_GEOMS:
            lines.append(
                f'    <pair name="log_{ch}" geom1="log_core" geom2="{ch}"'
                f' condim="3" friction="0.9 0.9 0.02" solref="0.01 1"/>')
    lines.append("    <!-- ===== trunk vs terrain ===== -->")
    for terrain in GROUND_GEOMS:
        f_t = cfg.ground_friction * (0.8 if terrain == "fall_line" else 1.0)
        lines.append(
            f'    <pair name="log_{terrain}" geom1="log_core" geom2="{terrain}"'
            f' condim="{cfg.log_condim}"'
            f' friction="{f_t:.4f} {f_t:.4f} 0.02'
            f' {cfg.log_roll_friction:.5f} {cfg.log_roll_friction:.5f}"'
            f' solref="0.01 1"/>')
    return "\n".join(lines)


def _sensors(cfg: SceneConfig) -> str:
    out = [
        '    <framepos name="log_pos" objtype="body" objname="log"/>',
        '    <framequat name="log_quat" objtype="body" objname="log"/>',
        '    <framelinvel name="log_linvel" objtype="body" objname="log"/>',
        '    <frameangvel name="log_angvel" objtype="body" objname="log"/>',
        '    <framexaxis name="log_xaxis" objtype="body" objname="log"/>',
    ]
    for p in ROBOTS:
        out += [
            f'    <framepos name="{p}pelvis_pos" objtype="body" objname="{p}pelvis"/>',
            f'    <framequat name="{p}torso_quat" objtype="body" objname="{p}torso_link"/>',
            f'    <framezaxis name="{p}torso_zaxis" objtype="body" objname="{p}torso_link"/>',
            f'    <gyro name="{p}gyro" site="{p}imu_in_torso"/>',
            f'    <accelerometer name="{p}accel" site="{p}imu_in_torso"/>',
            f'    <subtreecom name="{p}com" body="{p}pelvis"/>',
            f'    <subtreelinvel name="{p}comvel" body="{p}pelvis"/>',
        ]
        for side in ("left", "right"):
            out += [
                f'    <framepos name="{p}{side}_palm_pos" objtype="site" objname="{p}{side}_palm"/>',
                f'    <framepos name="{p}{side}_foot_pos" objtype="site" objname="{p}{side}_foot"/>',
            ]
    return "\n".join(out)


def build_xml(cfg: SceneConfig | None = None) -> str:
    cfg = cfg or SceneConfig()
    # Both robots face downhill, standing on the uphill side of the trunk.
    yaw = -math.pi / 2
    return f"""<mujoco model="alpine_trail_clearing">
  <!-- Generated by alpine_lift.scene.build_xml - do not edit by hand.
       config key: {cfg.key()} -->
  <compiler angle="radian" autolimits="true"/>

  <!-- iterations/ls_iterations are declared here to *match* the attached
       g1_mjx.xml exactly (5/8). Any other value -- including MuJoCo's own
       default -- counts as an attach conflict and prints a four-line warning
       per robot on every load. The values this scene runs are applied after
       compile, in Mission. -->
  <option timestep="{cfg.timestep}" integrator="implicitfast"
          iterations="5" ls_iterations="8"
          gravity="0 0 {cfg.gravity}"
          wind="{cfg.wind[0]} {cfg.wind[1]} {cfg.wind[2]}" density="1.0" viscosity="0.0">
    <flag eulerdamp="disable"/>
  </option>

  <statistic meansize="0.04" extent="5.0" center="0 0 0.5"/>

  <visual>
    <headlight diffuse="0.30 0.31 0.34" ambient="0.33 0.35 0.41" specular="0.2 0.2 0.2"/>
    <!-- MuJoCo's fog defaults to black, so enabling the FOG render flag
         without setting this paints the sky and the peaks dark. -->
    <rgba force="1 0.25 0.1 1" haze="0.80 0.86 0.93 1" fog="0.80 0.86 0.93 1"/>
    <global azimuth="118" elevation="-13" offwidth="1920" offheight="1080"/>
    <!-- zfar/fogstart/fogend are multiples of statistic.extent (5 m). -->
    <map force="0.005" zfar="800" fogstart="4" fogend="90"/>
    <scale forcewidth="0.03" contactwidth="0.08" contactheight="0.03"/>
    <quality shadowsize="4096" offsamples="8"/>
  </visual>

  <asset>
    <model name="g1" file="{_G1_PATH}"/>

    <texture type="skybox" builtin="gradient" rgb1="0.14 0.32 0.60" rgb2="0.87 0.92 0.97"
             width="512" height="3072"/>
    <texture type="2d" name="treadtex" builtin="flat" rgb1="0.44 0.39 0.32"
             rgb2="0.44 0.39 0.32" mark="random" markrgb="0.29 0.25 0.20"
             random="0.5" width="900" height="900"/>
    <material name="tread" texture="treadtex" texuniform="true" texrepeat="6 6"
              reflectance="0.0" specular="0.05" shininess="0.04"/>
    <texture type="2d" name="banktex" builtin="flat" rgb1="0.40 0.36 0.31"
             rgb2="0.40 0.36 0.31" mark="random" markrgb="0.26 0.23 0.20"
             random="0.42" width="700" height="700"/>
    <material name="bankrock" texture="banktex" texuniform="true" texrepeat="5 5"
              specular="0.06" shininess="0.05"/>
    <texture type="2d" name="screetex" builtin="flat" rgb1="0.47 0.45 0.44"
             rgb2="0.47 0.45 0.44" mark="random" markrgb="0.31 0.30 0.30"
             random="0.55" width="700" height="700"/>
    <material name="scree" texture="screetex" texuniform="true" texrepeat="9 9"
              specular="0.1" shininess="0.1"/>
    <texture type="2d" name="barktex" builtin="flat" rgb1="0.33 0.24 0.16"
             rgb2="0.33 0.24 0.16" mark="random" markrgb="0.20 0.14 0.09"
             random="0.5" width="512" height="512"/>
    <material name="bark" texture="barktex" texuniform="true" texrepeat="6 2"
              specular="0.05" shininess="0.04"/>
    <material name="sapwood" rgba="0.78 0.66 0.46 1" specular="0.1" shininess="0.1"/>
    <texture type="2d" name="granitetex" builtin="flat" rgb1="0.46 0.45 0.47"
             rgb2="0.46 0.45 0.47" mark="random" markrgb="0.31 0.30 0.33"
             random="0.35" width="512" height="512"/>
    <material name="granite" texture="granitetex" texuniform="true" texrepeat="2 2"
              specular="0.18" shininess="0.25"/>
    <material name="rockface" rgba="0.40 0.42 0.49 1" specular="0.08" shininess="0.08"/>
    <material name="rockdark" rgba="0.29 0.30 0.36 1" specular="0.06" shininess="0.06"/>
    <material name="snow" rgba="0.95 0.96 0.99 1" specular="0.35" shininess="0.4"/>
  </asset>

  <worldbody>
    <light name="sun" directional="true" diffuse="1.05 0.98 0.88" specular="0.35 0.34 0.30"
           pos="8 -10 14" dir="-0.42 0.52 -0.74" castshadow="true"/>
    <light name="skyfill" directional="true" diffuse="0.26 0.32 0.42" specular="0 0 0"
           pos="-9 8 11" dir="0.52 -0.44 -0.73" castshadow="false"/>

{_terrain(cfg)}

{_scenery(cfg)}

{_chocks(cfg)}

{_log(cfg)}

    <!-- Both robots uphill of the trunk, facing the fall line. -->
    <frame pos="{-cfg.robot_x:.4f} {cfg.robot_y:.4f} 0" euler="0 0 {yaw:.7f}">
      <attach model="g1" prefix="{A}"/>
    </frame>
    <frame pos="{cfg.robot_x:.4f} {cfg.robot_y:.4f} 0" euler="0 0 {yaw:.7f}">
      <attach model="g1" prefix="{B}"/>
    </frame>
  </worldbody>

  <contact>
{_pairs(cfg)}
  </contact>

  <sensor>
{_sensors(cfg)}
  </sensor>
</mujoco>
"""


def write_scene(cfg: SceneConfig | None = None, name: str | None = None) -> str:
    cfg = cfg or SceneConfig()
    os.makedirs(_GEN_DIR, exist_ok=True)
    name = name or f"trail_{cfg.key()}"
    path = os.path.join(_GEN_DIR, f"{name}.xml")
    with open(path, "w") as fh:
        fh.write(build_xml(cfg))
    return path
