"""Named oblique-fall scenarios shared by evaluation and video recording."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FallScenario:
    name: str
    label: str
    speed_mps: float
    heading_degrees: float
    lateral_speed_mps: float
    roll_degrees: float
    description: str

    def reset_options(self) -> dict[str, float | bool]:
        return {
            "randomize": False,
            "speed": self.speed_mps,
            "heading_degrees": self.heading_degrees,
            "lateral_speed": self.lateral_speed_mps,
            "roll_degrees": self.roll_degrees,
        }

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


SCENARIOS = (
    FallScenario(
        "fall_line_reference",
        "REFERENCE — FALL LINE",
        4.5,
        0.0,
        0.0,
        0.0,
        "Clean fall-line reference retained only as a control.",
    ),
    FallScenario(
        "left_oblique",
        "LEFT-OBLIQUE FALL",
        4.2,
        20.0,
        0.80,
        3.0,
        "Body and velocity are oblique toward camera-left with mild roll.",
    ),
    FallScenario(
        "right_oblique",
        "RIGHT-OBLIQUE FALL",
        4.8,
        -20.0,
        -0.80,
        -3.0,
        "Mirror of the left-oblique fall at a different downhill speed.",
    ),
    FallScenario(
        "left_crossed",
        "BODY/VELOCITY DISAGREE — LEFT",
        4.6,
        25.0,
        -0.90,
        5.0,
        "Body points left of the fall line while velocity crosses right.",
    ),
    FallScenario(
        "right_crossed",
        "BODY/VELOCITY DISAGREE — RIGHT",
        4.4,
        -25.0,
        0.90,
        -5.0,
        "Mirrored heading/velocity disagreement with opposite roll.",
    ),
    FallScenario(
        "hard_left_compound",
        "HARD COMPOUND FALL — LEFT",
        4.7,
        35.0,
        1.20,
        8.0,
        "Large heading error, cross-slope velocity, and body roll combined.",
    ),
    FallScenario(
        "hard_right_compound",
        "HARD COMPOUND FALL — RIGHT",
        4.3,
        -35.0,
        -1.20,
        -8.0,
        "Mirror hard case to expose one-sided policies or geometry exploits.",
    ),
    FallScenario(
        "hard_left_crossed_roll",
        "HARD CROSSED/ROLLED FALL — LEFT",
        4.6,
        35.0,
        -0.80,
        -9.0,
        "Extreme heading, cross-slope velocity, and roll have mixed signs.",
    ),
    FallScenario(
        "hard_right_crossed_roll",
        "HARD CROSSED/ROLLED FALL — RIGHT",
        4.6,
        -35.0,
        0.80,
        9.0,
        "Mirror of the randomized shallow-plant outlier found by audit.",
    ),
)


RANDOMIZED_ENVELOPE = {
    "downhill_speed_range_mps": (4.0, 5.0),
    "heading_range_degrees": (-40.0, 40.0),
    "lateral_speed_range_mps": (-1.50, 1.50),
    "roll_range_degrees": (-10.0, 10.0),
}
