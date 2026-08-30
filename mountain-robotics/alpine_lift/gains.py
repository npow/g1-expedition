"""Actuator gain schedule.

``g1_mjx.xml`` ships the gains MuJoCo Playground trains RL locomotion with
(kp 75, ankle kp 20). Those are deliberately soft: a learned policy runs at
50 Hz and supplies its own stiffness by moving the target every tick.

A model-based whole-body controller doing precision manipulation needs the
joint servos to actually hold a commanded posture, and the G1's real joint
controllers are far stiffer than 75. These are raised to the point where
the commanded torque stays inside each joint's declared
``actuatorfrcrange`` -- so nothing here asks the hardware for torque it does
not have; it only stops the simulated servo from being spongier than the
real one.

The arms are the exception and are deliberately left soft. Stiffening them
to match the legs (kp 270) puts an ~8 Hz resonance between the arm servos
and the compliant sling right through the middle of the carry: with the
controller frozen the sling force still rings +-60 N about a 108 N load,
and the payload eventually tips itself over. At kp 145 the same measurement
gives a standard deviation of 3.9 N. Holding a compliant load is a case
where a softer joint is the more stable one, and the palm integral in
controller.py recovers the tracking the lower gain gives up.
"""

from __future__ import annotations

import numpy as np

from .indexing import JOINT_ORDER, SceneIndex
from .scene import ROBOTS

# joint-name substring -> (kp, kv)
GAIN_TABLE: tuple[tuple[str, float, float], ...] = (
    ("ankle", 120.0, 6.0),
    ("knee", 260.0, 10.0),
    ("hip", 260.0, 10.0),
    ("waist", 220.0, 9.0),
    ("shoulder", 145.0, 7.0),
    ("elbow", 125.0, 6.0),
    ("wrist", 60.0, 3.0),
)


def gains_for(joint_name: str) -> tuple[float, float]:
    for key, kp, kv in GAIN_TABLE:
        if key in joint_name:
            return kp, kv
    return 100.0, 5.0


def apply_gains(model, ix: SceneIndex, scale: float = 1.0) -> None:
    """Rewrite the position actuators' kp/kv in place."""
    for p in ROBOTS:
        r = ix.robot(p)
        for i, name in enumerate(JOINT_ORDER):
            kp, kv = gains_for(name)
            kp *= scale
            kv *= np.sqrt(scale)
            a = int(r.act_ids[i])
            model.actuator_gainprm[a, 0] = kp
            model.actuator_biasprm[a, 1] = -kp
            model.actuator_biasprm[a, 2] = -kv
