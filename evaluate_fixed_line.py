"""Evaluate learned fixed-line ascent and causal/no-policy controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3 import PPO

from fixed_line_slope_env import G1FixedLineEnv


ActionFn = Callable[[np.ndarray, G1FixedLineEnv, int], np.ndarray]


def resolve_fixed_line_model(requested: str | None) -> Path:
    candidates = [
        Path(requested) if requested else None,
        Path("models/ppo_fixed_line_slope/g1_fixed_line_final.zip"),
        Path("models/ppo_fixed_line_hf/g1_fixed_line_final.zip"),
        Path("models/ppo_fixed_line/g1_fixed_line_final.zip"),
        Path("models/ppo_fixed_line/best/best_model.zip"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No fixed-line PPO checkpoint found. Run train_fixed_line.py or pass --model."
    )


def run_condition(
    action_fn: ActionFn,
    episodes: int,
    seed: int,
    *,
    line_enabled: bool = True,
    foot_ascender_enabled: bool = True,
) -> dict[str, float | int]:
    rows: list[dict] = []
    for episode in range(episodes):
        env = G1FixedLineEnv(randomize_reset=True)
        env.set_line_enabled(line_enabled)
        env.set_foot_ascender_enabled(foot_ascender_enabled)
        observation, _ = env.reset(seed=seed + episode, options={"randomize": True})
        info: dict = {}
        minimum_grasp_score = float("inf")
        maximum_left_grip_error = 0.0
        maximum_right_grip_error = 0.0
        minimum_hand_separation = float("inf")
        any_cross_hand_collision = False
        any_wall_hand_collision = False
        minimum_grounded_fraction = 1.0
        minimum_left_contact_fraction = 1.0
        minimum_right_contact_fraction = 1.0
        maximum_airborne_streak = 0.0
        minimum_ground_load_bodyweight = float("inf")
        for step in range(env.max_episode_steps):
            action = action_fn(observation, env, step)
            observation, _, terminated, truncated, info = env.step(action)
            minimum_grasp_score = min(minimum_grasp_score, info["grasp_score"])
            maximum_left_grip_error = max(
                maximum_left_grip_error, info["left_grip_error"]
            )
            maximum_right_grip_error = max(
                maximum_right_grip_error, info["right_grip_error"]
            )
            minimum_hand_separation = min(
                minimum_hand_separation, info["hand_separation"]
            )
            any_cross_hand_collision = any_cross_hand_collision or bool(
                info["cross_hand_collision"]
            )
            any_wall_hand_collision = any_wall_hand_collision or bool(
                info["wall_hand_collision"]
            )
            minimum_grounded_fraction = min(
                minimum_grounded_fraction, info.get("grounded_fraction", 1.0)
            )
            minimum_left_contact_fraction = min(
                minimum_left_contact_fraction,
                info.get("left_boot_contact_fraction", 1.0),
            )
            minimum_right_contact_fraction = min(
                minimum_right_contact_fraction,
                info.get("right_boot_contact_fraction", 1.0),
            )
            maximum_airborne_streak = max(
                maximum_airborne_streak, info.get("maximum_airborne_streak", 0.0)
            )
            minimum_ground_load_bodyweight = min(
                minimum_ground_load_bodyweight,
                info.get("ground_load_bodyweight", 0.0),
            )
            if terminated or truncated:
                break
        info = dict(info)
        info.update(
            {
                "minimum_grasp_score": minimum_grasp_score,
                "maximum_left_grip_error": maximum_left_grip_error,
                "maximum_right_grip_error": maximum_right_grip_error,
                "minimum_hand_separation": minimum_hand_separation,
                "any_cross_hand_collision": any_cross_hand_collision,
                "any_wall_hand_collision": any_wall_hand_collision,
                "minimum_grounded_fraction": minimum_grounded_fraction,
                "minimum_left_contact_fraction": minimum_left_contact_fraction,
                "minimum_right_contact_fraction": minimum_right_contact_fraction,
                "maximum_airborne_streak": maximum_airborne_streak,
                "minimum_ground_load_bodyweight": minimum_ground_load_bodyweight,
            }
        )
        rows.append(info)
        env.close()
    return {
        "episodes": episodes,
        "success_rate": float(np.mean([row.get("success", False) for row in rows])),
        "mean_ascent_m": float(np.mean([row["ascent"] for row in rows])),
        "mean_high_water_ascent_m": float(
            np.mean([row["high_water_ascent"] for row in rows])
        ),
        "mean_completed_cycles": float(
            np.mean([row["completed_cycles"] for row in rows])
        ),
        "mean_descent_from_high_water_m": float(
            np.mean([row["descent_from_high_water"] for row in rows])
        ),
        "mean_chest_load_fraction": float(
            np.mean([row["chest_load_fraction"] for row in rows])
        ),
        "mean_foot_loop_load_fraction": float(
            np.mean([row["foot_loop_load_fraction"] for row in rows])
        ),
        "mean_upright_score": float(np.mean([row["upright_score"] for row in rows])),
        "mean_minimum_grasp_score": float(
            np.mean([row["minimum_grasp_score"] for row in rows])
        ),
        "mean_maximum_left_grip_error_m": float(
            np.mean([row["maximum_left_grip_error"] for row in rows])
        ),
        "mean_maximum_right_grip_error_m": float(
            np.mean([row["maximum_right_grip_error"] for row in rows])
        ),
        "mean_minimum_hand_separation_m": float(
            np.mean([row["minimum_hand_separation"] for row in rows])
        ),
        "cross_hand_collision_rate": float(
            np.mean([row["any_cross_hand_collision"] for row in rows])
        ),
        "wall_hand_collision_rate": float(
            np.mean([row["any_wall_hand_collision"] for row in rows])
        ),
        "mean_vertical_gain_m": float(
            np.mean([row.get("vertical_gain_m", 0.0) for row in rows])
        ),
        "mean_grounded_fraction": float(
            np.mean([row.get("grounded_fraction", 0.0) for row in rows])
        ),
        "mean_left_boot_contact_fraction": float(
            np.mean([row.get("left_boot_contact_fraction", 0.0) for row in rows])
        ),
        "mean_right_boot_contact_fraction": float(
            np.mean([row.get("right_boot_contact_fraction", 0.0) for row in rows])
        ),
        "mean_double_support_fraction": float(
            np.mean([row.get("double_support_fraction", 0.0) for row in rows])
        ),
        "maximum_airborne_streak": float(
            max(row.get("maximum_airborne_streak", 0.0) for row in rows)
        ),
        "mean_ground_load_bodyweight": float(
            np.mean([row.get("ground_load_bodyweight", 0.0) for row in rows])
        ),
        "mean_line_load_fraction": float(
            np.mean([row.get("line_load_fraction", 0.0) for row in rows])
        ),
    }


def evaluate(
    model_path: str | None = None,
    episodes: int = 20,
    seed: int = 70_000,
    output: str = "models/ppo_fixed_line/evaluation_report.json",
) -> dict:
    checkpoint = resolve_fixed_line_model(model_path)
    policy = PPO.load(checkpoint, device="cpu")
    rng = np.random.default_rng(seed)

    def learned(observation: np.ndarray, _: G1FixedLineEnv, __: int) -> np.ndarray:
        action, _state = policy.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    def neutral(_: np.ndarray, env: G1FixedLineEnv, __: int) -> np.ndarray:
        return np.zeros(env.action_dim, dtype=np.float32)

    def random_action(_: np.ndarray, env: G1FixedLineEnv, __: int) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32)

    def reference_cycle(_: np.ndarray, env: G1FixedLineEnv, __: int) -> np.ndarray:
        if env._progress() - env._start_progress >= env.target_ascent:
            return np.zeros(2, dtype=np.float32)
        side = env._swing_side if env._swing_side is not None else env._expected_side
        return (
            np.asarray([1.0, -1.0], dtype=np.float32)
            if side == 0
            else np.asarray([-1.0, 1.0], dtype=np.float32)
        )

    report = {
        "checkpoint": str(checkpoint),
        "seed": seed,
        "learned_policy": run_condition(learned, episodes, seed),
        "learned_policy_step_actuation_disabled": run_condition(
            learned,
            episodes,
            seed,
            foot_ascender_enabled=False,
        ),
        "learned_policy_line_disabled": run_condition(
            learned,
            episodes,
            seed,
            line_enabled=False,
            foot_ascender_enabled=False,
        ),
        "neutral_action": run_condition(neutral, episodes, seed),
        "random_action": run_condition(random_action, episodes, seed),
        # This is a mechanics smoke control, not evidence for learned behavior.
        "reference_cycle": run_condition(reference_cycle, min(episodes, 5), seed),
    }
    learned_result = report["learned_policy"]
    foot_ablation = report["learned_policy_step_actuation_disabled"]
    line_ablation = report["learned_policy_line_disabled"]
    neutral_result = report["neutral_action"]
    gates = {
        "learned_success_rate_at_least_90pct": learned_result["success_rate"] >= 0.90,
        "learned_mean_ascent_at_least_1_50m": learned_result["mean_ascent_m"] >= 1.50,
        "learned_completes_at_least_8_steps": (
            learned_result["mean_completed_cycles"] >= 8.0
        ),
        "learned_chest_catch_is_loaded": (
            learned_result["mean_chest_load_fraction"] > 0.05
        ),
        "learned_boots_remain_grounded": (
            learned_result["mean_grounded_fraction"] > 0.90
        ),
        "learned_left_boot_has_sustained_contact": (
            learned_result["mean_left_boot_contact_fraction"] > 0.30
        ),
        "learned_right_boot_has_sustained_contact": (
            learned_result["mean_right_boot_contact_fraction"] > 0.30
        ),
        "learned_never_has_extended_flight": (
            learned_result["maximum_airborne_streak"] <= 3.0
        ),
        "learned_gains_height_on_incline": (
            learned_result["mean_vertical_gain_m"] > 0.60
        ),
        "learned_finishes_upright": learned_result["mean_upright_score"] > 0.82,
        "learned_retains_ratchet_height": (
            learned_result["mean_descent_from_high_water_m"] < 0.08
        ),
        "learned_right_grasp_score_stays_above_0_35": (
            learned_result["mean_minimum_grasp_score"] > 0.35
        ),
        "learned_right_grip_error_stays_below_0_14m": (
            learned_result["mean_maximum_right_grip_error_m"] < 0.14
        ),
        "learned_hands_remain_separated": (
            learned_result["mean_minimum_hand_separation_m"] > 0.12
        ),
        "learned_hands_never_collide_with_each_other": (
            learned_result["cross_hand_collision_rate"] == 0.0
        ),
        "learned_hands_never_collide_with_wall": (
            learned_result["wall_hand_collision_rate"] == 0.0
        ),
        "same_policy_without_step_actuation_never_succeeds": (
            foot_ablation["success_rate"] == 0.0
        ),
        "same_policy_without_step_actuation_gains_under_25pct_of_learned": (
            foot_ablation["mean_high_water_ascent_m"]
            < 0.25 * learned_result["mean_high_water_ascent_m"]
        ),
        "same_policy_without_line_never_succeeds": line_ablation["success_rate"] == 0.0,
        "neutral_action_never_succeeds": neutral_result["success_rate"] == 0.0,
        "neutral_action_gains_under_0_25m": (
            neutral_result["mean_high_water_ascent_m"] < 0.25
        ),
        "random_action_success_rate_at_most_10pct": (
            report["random_action"]["success_rate"] <= 0.10
        ),
        "reference_cycle_validates_mechanics": (
            report["reference_cycle"]["success_rate"] == 1.0
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
        raise RuntimeError(f"Fixed-line verification failed: {', '.join(failed)}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=70_000)
    parser.add_argument(
        "--output", default="models/ppo_fixed_line/evaluation_report.json"
    )
    args = parser.parse_args()
    evaluate(args.model, args.episodes, args.seed, args.output)
