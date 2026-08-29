"""Evaluate the learned arrest and causal/no-policy baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3 import PPO

from himalaya_env import G1SelfArrestEnv
from record_demo import resolve_model_path


def run_condition(
    action_fn: Callable[[np.ndarray, G1SelfArrestEnv], np.ndarray],
    episodes: int,
    seed: int,
    pick_enabled: bool,
    body_friction_enabled: bool = True,
    snow_resistance_enabled: bool = True,
) -> dict[str, float | int]:
    rows = []
    for episode in range(episodes):
        env = G1SelfArrestEnv(
            randomize_reset=True,
            initial_speed_range=(4.0, 5.0),
        )
        env.set_pick_enabled(pick_enabled)
        env.set_body_slope_friction_enabled(body_friction_enabled)
        if not snow_resistance_enabled:
            env.snow_drag_force_limit = 0.0
        observation, reset_info = env.reset(seed=seed + episode)
        info = {}
        minimum_ventral_margin = float("inf")
        minimum_axe_head_height = float("inf")
        maximum_axe_head_height = float("-inf")
        rigid_pick_contact_steps = 0
        snow_drag_sum = 0.0
        minimum_pick_height = float("inf")
        policy_steps = 0
        for policy_steps in range(1, env.max_episode_steps + 1):
            action = action_fn(observation, env)
            observation, _, terminated, truncated, info = env.step(action)
            minimum_ventral_margin = min(
                minimum_ventral_margin, info["ventral_placement_margin"]
            )
            minimum_axe_head_height = min(
                minimum_axe_head_height, info["axe_head_torso_z"]
            )
            maximum_axe_head_height = max(
                maximum_axe_head_height, info["axe_head_torso_z"]
            )
            rigid_pick_contact_steps += int(info["rigid_pick_contact"] > 0.5)
            snow_drag_sum += info["snow_drag_force"]
            minimum_pick_height = min(minimum_pick_height, info["pick_height"])
            if terminated or truncated:
                break
        info = dict(info)
        info.update(
            {
                "policy_steps": policy_steps,
                "reset_initial_speed": reset_info["v_slope"],
                "minimum_ventral_margin": minimum_ventral_margin,
                "minimum_axe_head_height": minimum_axe_head_height,
                "maximum_axe_head_height": maximum_axe_head_height,
                "rigid_pick_contact_fraction": (
                    rigid_pick_contact_steps / policy_steps
                ),
                "mean_snow_drag_force": snow_drag_sum / policy_steps,
                "minimum_pick_height": minimum_pick_height,
            }
        )
        rows.append(info)
        env.close()
    return {
        "episodes": episodes,
        "initial_speed_range_mps": [
            float(min(row["reset_initial_speed"] for row in rows)),
            float(max(row["reset_initial_speed"] for row in rows)),
        ],
        "success_rate": float(np.mean([row.get("success", False) for row in rows])),
        "mean_policy_steps": float(np.mean([row["policy_steps"] for row in rows])),
        "mean_final_speed_mps": float(np.mean([row["v_slope"] for row in rows])),
        "maximum_final_speed_mps": float(max(row["v_slope"] for row in rows)),
        "mean_stopping_distance_m": float(np.mean([row["stopping_distance"] for row in rows])),
        "mean_pick_contact_fraction": float(
            np.mean([row["pick_contact_fraction"] for row in rows])
        ),
        "mean_left_grasp_contact_fraction": float(
            np.mean([row["left_grasp_contact_fraction"] for row in rows])
        ),
        "mean_right_grasp_contact_fraction": float(
            np.mean([row["right_grasp_contact_fraction"] for row in rows])
        ),
        "mean_grip_score": float(np.mean([row["grip_score"] for row in rows])),
        "valid_learned_plant_motion_rate": float(
            np.mean([row["valid_learned_plant_motion"] for row in rows])
        ),
        "minimum_first_rigid_contact_step": float(
            min(row["first_rigid_contact_step"] for row in rows)
        ),
        "minimum_stroke_at_first_contact_m": float(
            min(row["stroke_at_first_contact"] for row in rows)
        ),
        "minimum_lowering_at_first_contact_m": float(
            min(row["lowering_at_first_contact"] for row in rows)
        ),
        "minimum_blade_angle_at_first_contact_deg": float(
            min(row["blade_angle_at_first_contact_deg"] for row in rows)
        ),
        "minimum_terminal_rolling_blade_angle_deg": float(
            min(row["rolling_pick_blade_into_slope_angle_deg"] for row in rows)
        ),
        "maximum_terminal_rolling_blade_angle_deg": float(
            max(row["rolling_pick_blade_into_slope_angle_deg"] for row in rows)
        ),
        "mean_rigid_pick_contact_fraction": float(
            np.mean([row["rigid_pick_contact_fraction"] for row in rows])
        ),
        "mean_rigid_pick_contact_substep_fraction": float(
            np.mean([row["rigid_pick_contact_substep_fraction"] for row in rows])
        ),
        "minimum_terminal_rolling_rigid_pick_contact_fraction": float(
            min(row["rolling_rigid_pick_contact_fraction"] for row in rows)
        ),
        "minimum_terminal_rolling_mean_snow_drag_force_n": float(
            min(row["rolling_mean_snow_drag_force"] for row in rows)
        ),
        "mean_snow_drag_force_n": float(
            np.mean([row["mean_snow_drag_force"] for row in rows])
        ),
        "minimum_pick_height_m": float(
            min(row["minimum_pick_height"] for row in rows)
        ),
        "minimum_ventral_placement_margin_m": float(
            min(row["minimum_ventral_margin"] for row in rows)
        ),
        "minimum_axe_head_torso_z_m": float(
            min(row["minimum_axe_head_height"] for row in rows)
        ),
        "maximum_axe_head_torso_z_m": float(
            max(row["maximum_axe_head_height"] for row in rows)
        ),
    }


def evaluate(
    model_path: str | None = None,
    episodes: int = 30,
    seed: int = 50_000,
    output: str = "models/ppo_self_arrest/evaluation_report.json",
) -> dict:
    checkpoint = resolve_model_path(model_path)
    policy = PPO.load(checkpoint, device="cpu")
    rng = np.random.default_rng(seed)

    def learned(observation: np.ndarray, _: G1SelfArrestEnv) -> np.ndarray:
        action, _state = policy.predict(observation, deterministic=True)
        return action

    def neutral(_: np.ndarray, env: G1SelfArrestEnv) -> np.ndarray:
        return np.zeros(env.action_dim, dtype=np.float32)

    def random_action(_: np.ndarray, env: G1SelfArrestEnv) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32)

    report = {
        "checkpoint": str(checkpoint),
        "seed": seed,
        "learned_policy": run_condition(learned, episodes, seed, pick_enabled=True),
        "learned_policy_body_friction_disabled": run_condition(
            learned,
            episodes,
            seed,
            pick_enabled=True,
            body_friction_enabled=False,
        ),
        "learned_policy_snow_resistance_disabled": run_condition(
            learned,
            episodes,
            seed,
            pick_enabled=True,
            snow_resistance_enabled=False,
        ),
        "learned_policy_body_friction_and_snow_resistance_disabled": run_condition(
            learned,
            episodes,
            seed,
            pick_enabled=True,
            body_friction_enabled=False,
            snow_resistance_enabled=False,
        ),
        "learned_policy_pick_disabled": run_condition(
            learned, episodes, seed, pick_enabled=False
        ),
        "neutral_action": run_condition(neutral, episodes, seed, pick_enabled=True),
        "random_action": run_condition(random_action, episodes, seed, pick_enabled=True),
    }
    learned_result = report["learned_policy"]
    no_body_friction_result = report["learned_policy_body_friction_disabled"]
    no_snow_resistance_result = report["learned_policy_snow_resistance_disabled"]
    no_body_or_snow_resistance_result = report[
        "learned_policy_body_friction_and_snow_resistance_disabled"
    ]
    gates = {
        "learned_success_rate_at_least_95pct": learned_result["success_rate"] >= 0.95,
        "learned_without_body_friction_success_rate_at_least_95pct": (
            no_body_friction_result["success_rate"] >= 0.95
        ),
        "body_friction_ablation_changes_success_rate_by_at_most_5pct": (
            abs(
                learned_result["success_rate"]
                - no_body_friction_result["success_rate"]
            )
            <= 0.05
        ),
        "learned_final_speed_below_0_20_mps": (
            learned_result["mean_final_speed_mps"] < 0.20
        ),
        "every_terminal_contact_window_is_majority_real_pick_contact": (
            learned_result[
                "minimum_terminal_rolling_rigid_pick_contact_fraction"
            ]
            > 0.50
        ),
        "every_terminal_contact_window_has_mean_axe_load_above_100N": (
            learned_result["minimum_terminal_rolling_mean_snow_drag_force_n"]
            > 100.0
        ),
        "learned_left_grasp_contact_fraction_above_0_40": (
            learned_result["mean_left_grasp_contact_fraction"] > 0.40
        ),
        "learned_right_grasp_contact_fraction_above_0_90": (
            learned_result["mean_right_grasp_contact_fraction"] > 0.90
        ),
        "learned_grip_score_above_0_85": learned_result["mean_grip_score"] > 0.85,
        "every_learned_rollout_has_a_valid_visible_plant": (
            learned_result["valid_learned_plant_motion_rate"] == 1.0
        ),
        "every_learned_plant_waits_at_least_20_policy_steps": (
            learned_result["minimum_first_rigid_contact_step"] >= 20
        ),
        "every_learned_plant_moves_pick_at_least_8cm_before_contact": (
            learned_result["minimum_stroke_at_first_contact_m"] > 0.08
        ),
        "every_learned_plant_lowers_pick_at_least_8cm_before_contact": (
            learned_result["minimum_lowering_at_first_contact_m"] > 0.08
        ),
        "every_learned_plant_enters_slope_at_more_than_18deg": (
            learned_result["minimum_blade_angle_at_first_contact_deg"] > 18.0
        ),
        "every_terminal_blade_angle_is_between_22deg_and_42deg": (
            learned_result["minimum_terminal_rolling_blade_angle_deg"] > 22.0
            and learned_result["maximum_terminal_rolling_blade_angle_deg"] < 42.0
        ),
        "axe_and_wrists_remain_at_least_3cm_ventral": (
            learned_result["minimum_ventral_placement_margin_m"] > 0.03
        ),
        "axe_head_remains_in_front_of_upper_torso": (
            learned_result["minimum_axe_head_torso_z_m"] > 0.15
            # The new visibly raised ready pose reaches 35.8 cm above the
            # torso origin before the stroke.  The prior 34 cm ceiling was
            # calibrated to an already-planted pose and would forbid the very
            # pre-contact motion this audit is intended to require.
            and learned_result["maximum_axe_head_torso_z_m"] < 0.40
        ),
        "same_policy_without_pick_never_succeeds": (
            report["learned_policy_pick_disabled"]["success_rate"] == 0.0
        ),
        "same_policy_without_axe_snow_resistance_never_succeeds": (
            no_snow_resistance_result["success_rate"] == 0.0
        ),
        "same_policy_without_axe_snow_resistance_keeps_moving_above_2mps": (
            no_snow_resistance_result["mean_final_speed_mps"] > 2.0
        ),
        "same_policy_without_body_friction_or_axe_snow_resistance_never_succeeds": (
            no_body_or_snow_resistance_result["success_rate"] == 0.0
        ),
        "same_policy_without_body_friction_or_axe_snow_resistance_keeps_moving_above_2mps": (
            no_body_or_snow_resistance_result["mean_final_speed_mps"] > 2.0
        ),
        "neutral_action_never_succeeds": (
            report["neutral_action"]["success_rate"] == 0.0
        ),
        "random_action_success_rate_at_most_10pct": (
            report["random_action"]["success_rate"] <= 0.10
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
        raise RuntimeError(f"Self-arrest verification failed: {', '.join(failed)}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=50_000)
    parser.add_argument("--output", default="models/ppo_self_arrest/evaluation_report.json")
    args = parser.parse_args()
    evaluate(args.model, args.episodes, args.seed, args.output)
