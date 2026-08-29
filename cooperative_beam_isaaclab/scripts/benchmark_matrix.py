#!/usr/bin/env python3
"""Print the factorized train/held-out benchmark condition manifest."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Payload:
    task: str
    label: str
    nominal_team: int
    nominal_mass_kg: float


PAYLOADS = (
    Payload("Isaac-Cooperative-G1-Rescue-Crate-Direct-v0", "rescue_crate", 2, 7.0),
    Payload("Isaac-Cooperative-G1-Timber-Direct-v0", "fallen_timber", 3, 15.0),
    Payload("Isaac-Cooperative-G1-Footbridge-Girder-Direct-v0", "footbridge_girder", 5, 24.0),
)

TRAIN_OBJECTS = frozenset(("rescue_crate", "fallen_timber"))
TRAIN_TEAM_SIZES = frozenset((2, 3, 5))
TRAIN_KG_PER_ROBOT = frozenset((2.0, 4.0, 6.0))


def condition_split(object_label: str, team_size: int, kg_per_robot: float) -> tuple[str, str]:
    """Classify one condition without letting held-out factors leak into training."""
    holdout_factors = []
    if object_label not in TRAIN_OBJECTS:
        holdout_factors.append("geometry")
    if team_size not in TRAIN_TEAM_SIZES:
        holdout_factors.append("team_size")
    if kg_per_robot not in TRAIN_KG_PER_ROBOT:
        holdout_factors.append("mass")
    if holdout_factors:
        return "heldout", "+".join(holdout_factors)
    return "train", "none"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team_sizes", type=int, nargs="+", default=(2, 3, 4, 5, 6))
    parser.add_argument("--kg_per_robot", type=float, nargs="+", default=(2.0, 4.0, 6.0, 9.0))
    args = parser.parse_args()

    writer = csv.writer(sys.stdout)
    writer.writerow(
        (
            "object",
            "team_size",
            "payload_mass_kg",
            "kg_per_robot",
            "kg_per_arm",
            "split",
            "holdout_factors",
            "regime",
            "scripted_smoke_command",
        )
    )
    for payload in PAYLOADS:
        for team_size in args.team_sizes:
            for kg_per_robot in args.kg_per_robot:
                mass = team_size * kg_per_robot
                regime = "stress" if kg_per_robot > 6.0 else "nominal"
                split, holdout_factors = condition_split(payload.label, team_size, kg_per_robot)
                command = (
                    f"python scripts/smoke.py --task {payload.task} --team_size {team_size} "
                    f"--payload_mass {mass:g} --steps 180 --scripted_lift --viz none"
                )
                writer.writerow(
                    (
                        payload.label,
                        team_size,
                        f"{mass:.1f}",
                        f"{kg_per_robot:.1f}",
                        f"{kg_per_robot / 2.0:.1f}",
                        split,
                        holdout_factors,
                        regime,
                        command,
                    )
                )


if __name__ == "__main__":
    main()
