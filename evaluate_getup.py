"""Evaluate the physical fall/get-up and causal controller ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from getup_controller import FallCondition, G1PhysicalGetup


PHYSICS_CASES = (
    FallCondition("nominal backward shove", (-100.0, 0.0, 0.0)),
    FallCondition("softer backward shove", (-80.0, 0.0, 0.0)),
    FallCondition("left-biased shove", (-100.0, 4.0, 0.0)),
    FallCondition("right-biased shove", (-100.0, -4.0, 0.0)),
)


def run_case(condition: FallCondition, mode: str) -> dict[str, Any]:
    controller = G1PhysicalGetup()
    controller.reset(condition=condition, mode=mode)
    for _ in controller.rollout():
        pass
    report = controller.report()
    controller.close()
    return report


def evaluate(output_path: str) -> dict[str, Any]:
    learned_cases = [run_case(condition, "policy") for condition in PHYSICS_CASES]
    reference_only = run_case(PHYSICS_CASES[0], "reference_only")
    motors_off = run_case(PHYSICS_CASES[0], "motors_off")

    gates = {
        "all_perturbed_falls_recover": all(row["success"] for row in learned_cases),
        "all_final_stances_stable": all(
            row["final_pelvis_height_m"] > 0.72
            and row["final_torso_upright"] > 0.95
            and row["final_base_linear_speed_mps"] < 0.15
            and row["final_base_angular_speed_radps"] < 0.35
            and row["final_both_feet_contact"]
            for row in learned_cases
        ),
        "all_falls_settle_on_nonfoot_geometry": all(
            row["fall_end"]["nonfoot_ground_contact"] for row in learned_cases
        ),
        "all_falls_settle_below_velocity_limits": all(
            row["fall_end"]["base_linear_speed_mps"] < 0.05
            and row["fall_end"]["base_angular_speed_radps"] < 0.10
            for row in learned_cases
        ),
        "no_floating_base_teleports": all(
            row["root_teleports_after_fall_start"] == 0 for row in learned_cases
        ),
        "motor_catalog_limits_enforced": all(
            row["peak_motor_torque_ratio"] <= 1.00001 for row in learned_cases
        ),
        "contact_penetration_below_30_mm": all(
            row["maximum_contact_penetration_m"] < 0.030 for row in learned_cases
        ),
        "reference_trajectory_alone_fails": not reference_only["success"],
        "motors_off_recovery_fails": not motors_off["success"],
    }
    result = {
        "passed": all(gates.values()),
        "summary": {
            "learned_policy_successes": sum(row["success"] for row in learned_cases),
            "learned_policy_cases": len(learned_cases),
            "maximum_contact_penetration_m": max(
                row["maximum_contact_penetration_m"] for row in learned_cases
            ),
            "maximum_motor_torque_ratio": max(
                row["peak_motor_torque_ratio"] for row in learned_cases
            ),
            "minimum_final_pelvis_height_m": min(
                row["final_pelvis_height_m"] for row in learned_cases
            ),
            "minimum_final_torso_upright": min(
                row["final_torso_upright"] for row in learned_cases
            ),
        },
        "gates": gates,
        "learned_policy_cases": learned_cases,
        "ablations": {
            "reference_only": reference_only,
            "motors_off": motors_off,
        },
        "policy_provenance": {
            "source": "wbc-mjlab/wbc-g1-deploy",
            "commit": "6dabf86fddc2b7b429b09e74999732fcde3441f9",
            "license": "Apache-2.0",
            "claim": "pretrained upstream WBC policy adapted here; not trained in this repository",
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="models/wbc_getup/evaluation_report.json",
        help="JSON evaluation destination",
    )
    args = parser.parse_args()
    result = evaluate(args.output)
    print(json.dumps(result["summary"], indent=2))
    print(json.dumps(result["gates"], indent=2))
    if not result["passed"]:
        raise SystemExit("One or more physical get-up gates failed")


if __name__ == "__main__":
    main()
