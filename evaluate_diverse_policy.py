"""Evaluate one learned PPO policy over fixed and randomized oblique falls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from diverse_scenarios import RANDOMIZED_ENVELOPE, SCENARIOS, FallScenario
from himalaya_env import G1SelfArrestEnv
from record_demo import resolve_model_path


def run_rollout(
    policy: PPO,
    *,
    seed: int,
    options: dict[str, Any] | None = None,
    randomized: bool = False,
    body_friction_enabled: bool = True,
    snow_resistance_enabled: bool = True,
    pick_enabled: bool = True,
) -> dict[str, float | bool | int]:
    envelope = RANDOMIZED_ENVELOPE
    env = G1SelfArrestEnv(
        randomize_reset=randomized,
        initial_speed_range=envelope["downhill_speed_range_mps"],
        heading_range_degrees=envelope["heading_range_degrees"],
        lateral_speed_range=envelope["lateral_speed_range_mps"],
        roll_range_degrees=envelope["roll_range_degrees"],
    )
    env.set_body_slope_friction_enabled(body_friction_enabled)
    env.set_pick_enabled(pick_enabled)
    if not snow_resistance_enabled:
        env.snow_drag_force_limit = 0.0
    observation, reset_info = env.reset(seed=seed, options=options)
    info: dict[str, Any] = {}
    minimum_front_margin = float("inf")
    for step in range(1, env.max_episode_steps + 1):
        action, _ = policy.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        minimum_front_margin = min(
            minimum_front_margin, float(info["ventral_placement_margin"])
        )
        if terminated or truncated:
            break
    row: dict[str, float | bool | int] = {
        "success": bool(info.get("success", False)),
        "policy_steps": step,
        "initial_total_slope_speed_mps": float(reset_info["v_slope"]),
        "final_speed_mps": float(info["v_slope"]),
        "stopping_distance_m": float(info["stopping_distance"]),
        "heading_degrees": float(info["reset_heading_degrees"]),
        "lateral_speed_mps": float(info["reset_lateral_speed_mps"]),
        "roll_degrees": float(info["reset_roll_degrees"]),
        "valid_learned_plant_motion": bool(info["valid_learned_plant_motion"]),
        "first_contact_step": int(info["first_rigid_contact_step"]),
        "stroke_at_first_contact_m": float(info["stroke_at_first_contact"]),
        "lowering_at_first_contact_m": float(info["lowering_at_first_contact"]),
        "first_contact_blade_angle_deg": float(
            info["blade_angle_at_first_contact_deg"]
        ),
        "terminal_rolling_blade_angle_deg": float(
            info["rolling_pick_blade_into_slope_angle_deg"]
        ),
        "terminal_rigid_pick_contact_fraction": float(
            info["rolling_rigid_pick_contact_fraction"]
        ),
        "terminal_mean_snow_drag_force_n": float(
            info["rolling_mean_snow_drag_force"]
        ),
        "left_grasp_contact_fraction": float(
            info["left_grasp_contact_fraction"]
        ),
        "right_grasp_contact_fraction": float(
            info["right_grasp_contact_fraction"]
        ),
        "grip_score": float(info["grip_score"]),
        "minimum_front_of_torso_margin_m": float(
            minimum_front_margin
        ),
    }
    env.close()
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_final_speed_mps": float(
            np.mean([row["final_speed_mps"] for row in rows])
        ),
        "maximum_final_speed_mps": float(
            max(row["final_speed_mps"] for row in rows)
        ),
        "valid_plant_rate": float(
            np.mean([row["valid_learned_plant_motion"] for row in rows])
        ),
        "minimum_first_contact_step": int(
            min(row["first_contact_step"] for row in rows)
        ),
        "minimum_stroke_at_first_contact_m": float(
            min(row["stroke_at_first_contact_m"] for row in rows)
        ),
        "minimum_lowering_at_first_contact_m": float(
            min(row["lowering_at_first_contact_m"] for row in rows)
        ),
        "minimum_first_contact_blade_angle_deg": float(
            min(row["first_contact_blade_angle_deg"] for row in rows)
        ),
        "terminal_rolling_blade_angle_range_deg": [
            float(min(row["terminal_rolling_blade_angle_deg"] for row in rows)),
            float(max(row["terminal_rolling_blade_angle_deg"] for row in rows)),
        ],
        "minimum_left_grasp_contact_fraction": float(
            min(row["left_grasp_contact_fraction"] for row in rows)
        ),
        "minimum_right_grasp_contact_fraction": float(
            min(row["right_grasp_contact_fraction"] for row in rows)
        ),
        "minimum_grip_score": float(min(row["grip_score"] for row in rows)),
        "minimum_front_of_torso_margin_m": float(
            min(row["minimum_front_of_torso_margin_m"] for row in rows)
        ),
        "rows": rows,
    }


def fixed_condition(
    policy: PPO,
    *,
    seed: int,
    body_friction_enabled: bool,
    snow_resistance_enabled: bool,
    pick_enabled: bool = True,
) -> list[dict[str, Any]]:
    rows = []
    for index, scenario in enumerate(SCENARIOS):
        row = run_rollout(
            policy,
            seed=seed + index,
            options=scenario.reset_options(),
            body_friction_enabled=body_friction_enabled,
            snow_resistance_enabled=snow_resistance_enabled,
            pick_enabled=pick_enabled,
        )
        row.update(scenario.to_dict())
        rows.append(row)
    return rows


def randomized_condition(
    policy: PPO,
    *,
    episodes: int,
    seed: int,
    body_friction_enabled: bool,
    snow_resistance_enabled: bool,
    pick_enabled: bool = True,
) -> list[dict[str, Any]]:
    return [
        run_rollout(
            policy,
            seed=seed + episode,
            randomized=True,
            body_friction_enabled=body_friction_enabled,
            snow_resistance_enabled=snow_resistance_enabled,
            pick_enabled=pick_enabled,
        )
        for episode in range(episodes)
    ]


def evaluate_diversity(
    model_path: str | None,
    randomized_episodes: int,
    seed: int,
    output: str,
) -> dict[str, Any]:
    checkpoint = resolve_model_path(model_path)
    policy = PPO.load(checkpoint, device="cpu")
    fixed_normal = fixed_condition(
        policy, seed=seed, body_friction_enabled=True, snow_resistance_enabled=True
    )
    fixed_no_body_friction = fixed_condition(
        policy, seed=seed, body_friction_enabled=False, snow_resistance_enabled=True
    )
    fixed_no_snow = fixed_condition(
        policy, seed=seed, body_friction_enabled=False, snow_resistance_enabled=False
    )
    fixed_no_pick = fixed_condition(
        policy,
        seed=seed,
        body_friction_enabled=False,
        snow_resistance_enabled=True,
        pick_enabled=False,
    )
    random_normal = randomized_condition(
        policy,
        episodes=randomized_episodes,
        seed=seed + 10_000,
        body_friction_enabled=True,
        snow_resistance_enabled=True,
    )
    random_no_body_friction = randomized_condition(
        policy,
        episodes=randomized_episodes,
        seed=seed + 10_000,
        body_friction_enabled=False,
        snow_resistance_enabled=True,
    )
    random_no_snow = randomized_condition(
        policy,
        episodes=max(30, randomized_episodes // 2),
        seed=seed + 20_000,
        body_friction_enabled=False,
        snow_resistance_enabled=False,
    )
    random_no_pick = randomized_condition(
        policy,
        episodes=max(30, randomized_episodes // 2),
        seed=seed + 20_000,
        body_friction_enabled=False,
        snow_resistance_enabled=True,
        pick_enabled=False,
    )
    report = {
        "checkpoint": str(checkpoint.resolve()),
        "seed": seed,
        "fixed_scenarios": [scenario.to_dict() for scenario in SCENARIOS],
        "randomized_envelope": RANDOMIZED_ENVELOPE,
        "fixed_normal": summarize(fixed_normal),
        "fixed_no_body_friction": summarize(fixed_no_body_friction),
        "fixed_no_axe_snow_resistance": summarize(fixed_no_snow),
        "fixed_pick_disabled": summarize(fixed_no_pick),
        "randomized_normal": summarize(random_normal),
        "randomized_no_body_friction": summarize(random_no_body_friction),
        "randomized_no_axe_snow_resistance": summarize(random_no_snow),
        "randomized_pick_disabled": summarize(random_no_pick),
    }
    strict_rows = fixed_no_body_friction + random_no_body_friction
    gates = {
        "all_fixed_scenarios_succeed_without_body_friction": all(
            row["success"] for row in fixed_no_body_friction
        ),
        "randomized_success_rate_at_least_95pct_without_body_friction": (
            report["randomized_no_body_friction"]["success_rate"] >= 0.95
        ),
        "normal_and_no_body_friction_randomized_success_match": (
            report["randomized_normal"]["success_rate"]
            == report["randomized_no_body_friction"]["success_rate"]
        ),
        "every_strict_rollout_has_valid_learned_plant": all(
            row["valid_learned_plant_motion"] for row in strict_rows
        ),
        "every_strict_plant_moves_8cm_and_lowers_5cm": all(
            row["stroke_at_first_contact_m"] > 0.08
            and row["lowering_at_first_contact_m"] > 0.05
            for row in strict_rows
        ),
        "every_strict_plant_enters_above_18deg": all(
            row["first_contact_blade_angle_deg"] > 18.0 for row in strict_rows
        ),
        "every_successful_strict_terminal_angle_is_22_to_42deg": all(
            (not row["success"])
            or 22.0 < row["terminal_rolling_blade_angle_deg"] < 42.0
            for row in strict_rows
        ),
        "axe_and_wrists_always_remain_in_front_of_torso": all(
            row["minimum_front_of_torso_margin_m"] > 0.03
            for row in strict_rows
        ),
        "every_fixed_scenario_fails_without_axe_snow_resistance": not any(
            row["success"] for row in fixed_no_snow
        ),
        "randomized_never_succeeds_without_axe_snow_resistance": not any(
            row["success"] for row in random_no_snow
        ),
        "every_fixed_scenario_fails_when_pick_contact_is_disabled": not any(
            row["success"] for row in fixed_no_pick
        ),
        "randomized_never_succeeds_when_pick_contact_is_disabled": not any(
            row["success"] for row in random_no_pick
        ),
        "pick_disabled_randomized_mean_final_speed_above_2mps": (
            report["randomized_pick_disabled"]["mean_final_speed_mps"] > 2.0
        ),
    }
    report["verification_gates"] = gates
    report["all_verification_gates_passed"] = all(gates.values())
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["all_verification_gates_passed"]:
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"Diverse scenario verification failed: {', '.join(failed)}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--randomized-episodes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=73_000)
    parser.add_argument(
        "--output", default="models/ppo_self_arrest/diverse_evaluation_report.json"
    )
    arguments = parser.parse_args()
    evaluate_diversity(
        arguments.model,
        arguments.randomized_episodes,
        arguments.seed,
        arguments.output,
    )
