"""Evaluate fall, learned get-up, re-grasp, and continued fixed-line ascent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from evaluate_fixed_line import resolve_fixed_line_model
from fixed_line_slope_env import G1FixedLineEnv
from mountain_recovery import FixedLineRecoveryController


ROOT = Path(__file__).resolve().parent
DEFAULT_RECOVERY_MODEL = (
    ROOT / "models" / "ppo_mountain_recovery" / "g1_mountain_recovery_final.zip"
)


def resolve_recovery_model(path: str | None) -> Path:
    checkpoint = Path(path) if path else DEFAULT_RECOVERY_MODEL
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Recovery checkpoint not found: {checkpoint}. "
            "Run train_mountain_recovery.py first."
        )
    return checkpoint


def _advance_climb(
    env: G1FixedLineEnv,
    policy: PPO,
    observation: np.ndarray,
    steps: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    info: dict[str, Any] = {}
    for _ in range(steps):
        action, _state = policy.predict(observation, deterministic=True)
        observation, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    for _ in range(40):
        if env._swing_side is None:
            break
        action, _state = policy.predict(observation, deterministic=True)
        observation, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return observation, info


def run_case(
    recovery_policy: PPO | None,
    climbing_policy: PPO,
    *,
    pre_fall_steps: int,
    lateral_bias_n: float,
    seed: int,
    continue_climbing: bool = False,
    continued_climb_segments: int = 1,
) -> dict[str, Any]:
    env = G1FixedLineEnv(randomize_reset=False)
    observation, _ = env.reset(seed=seed, options={"randomize": False})
    controller = FixedLineRecoveryController(env)
    observation, pre_fall_info = _advance_climb(
        env, climbing_policy, observation, pre_fall_steps
    )
    pre_fall_progress = env._progress()
    controller.start_fall()
    for frame in range(controller.fall_frames):
        fall_info = controller.step_fall(frame, lateral_bias_n=lateral_bias_n)
    grounded_progress = env._progress()
    fall_end = dict(fall_info)
    for frame in range(controller.floor_ready_frames):
        controller.step_floor_ready(frame)

    actions: list[np.ndarray] = []
    for _ in range(controller.recovery_frames):
        if recovery_policy is None:
            action = np.zeros(4, dtype=np.float32)
        else:
            action, _state = recovery_policy.predict(
                controller.policy_observation(), deterministic=True
            )
        actions.append(np.asarray(action, dtype=np.float64))
        recovery_info = controller.step_recovery(action)
        if controller.recovered:
            break

    regrasp_stable = False
    final_climb_info: dict[str, Any] = {}
    completed_climb_segments = 0
    total_climb_ascent = 0.0
    total_climb_cycles = 0
    total_climb_rope_collisions = 0
    total_climb_hand_penetrations = 0
    post_recovery_progress = env._progress()
    if controller.recovered:
        controller.regrasp_for_climb(100)
        settled = env._metrics()
        regrasp_stable = bool(
            settled["pelvis_normal_height"] > 0.58
            and settled["upright_score"] > 0.90
            and settled["left_boot_contact"]
            and settled["right_boot_contact"]
            and np.linalg.norm(env.data.qvel[:3]) < 0.15
            and np.linalg.norm(env.data.qvel[3:6]) < 0.35
            and settled["rope_core_collision"] < 0.5
            and settled["hand_rope_max_penetration_m"] <= 8e-4
        )
        if continue_climbing and regrasp_stable:
            for _segment in range(continued_climb_segments):
                observation, _ = env.rebase_climb_progress()
                final_climb_info = {}
                for _ in range(env.max_episode_steps):
                    action, _state = climbing_policy.predict(
                        observation, deterministic=True
                    )
                    (
                        observation,
                        _reward,
                        terminated,
                        truncated,
                        final_climb_info,
                    ) = env.step(action)
                    if terminated or truncated:
                        break
                total_climb_ascent += float(
                    final_climb_info.get("ascent", 0.0)
                )
                total_climb_cycles += int(
                    final_climb_info.get("completed_cycles", 0)
                )
                total_climb_rope_collisions += int(
                    final_climb_info.get("rope_core_collision_steps", 0)
                )
                total_climb_hand_penetrations += int(
                    final_climb_info.get("hand_rope_penetration_steps", 0)
                )
                if not final_climb_info.get("success", False):
                    break
                completed_climb_segments += 1

    action_array = np.asarray(actions)
    result = {
        "seed": seed,
        "pre_fall_policy_steps": pre_fall_steps,
        "lateral_fall_bias_n": lateral_bias_n,
        "pre_fall_ascent_m": float(pre_fall_info.get("ascent", 0.0)),
        "fall_arrest_progress_loss_m": float(
            max(pre_fall_progress - grounded_progress, 0.0)
        ),
        "fall_end": fall_end,
        "recovered": controller.recovered,
        "recovery_policy_steps": len(actions),
        "recovery_progress_change_m": float(
            post_recovery_progress - grounded_progress
        ),
        "net_post_fall_progress_m": float(
            env._progress() - grounded_progress
        ),
        "regrasp_stable": regrasp_stable,
        "root_teleports_after_fall_start": (
            controller.root_teleports_after_fall_start
        ),
        "peak_line_load_n": controller.peak_line_load_n,
        "peak_lateral_guide_load_n": controller.peak_lateral_guide_load_n,
        "peak_motor_torque_ratio": controller.peak_motor_torque_ratio,
        "peak_contact_force_n": controller.peak_contact_force_n,
        "maximum_contact_penetration_m": (
            controller.maximum_contact_penetration_m
        ),
        "rope_core_collision_frames": controller.rope_core_collision_frames,
        "hand_rope_penetration_frames": (
            controller.hand_rope_penetration_frames
        ),
        "mean_recovery_action": (
            action_array.mean(axis=0).tolist() if len(action_array) else []
        ),
        "continued_climb_success": bool(
            continue_climbing
            and completed_climb_segments == continued_climb_segments
        ),
        "continued_climb_segments_completed": completed_climb_segments,
        "continued_climb_ascent_m": total_climb_ascent,
        "continued_climb_cycles": total_climb_cycles,
        "continued_climb_rope_core_collision_steps": total_climb_rope_collisions,
        "continued_climb_hand_rope_penetration_steps": (
            total_climb_hand_penetrations
        ),
    }
    controller.close()
    env.close()
    return result


def evaluate(
    recovery_model: str | None = None,
    climbing_model: str | None = None,
    output_path: str = "models/ppo_mountain_recovery/evaluation_report.json",
) -> dict[str, Any]:
    recovery_checkpoint = resolve_recovery_model(recovery_model)
    climbing_checkpoint = resolve_fixed_line_model(climbing_model)
    recovery_policy = PPO.load(recovery_checkpoint, device="cpu")
    climbing_policy = PPO.load(climbing_checkpoint, device="cpu")
    # Validate the finalized post-climb fall at two bounded transverse slip
    # variants.  This is not a claim that one short hackathon policy recovers
    # every possible mountain fall.
    fall_biases = (4.0, 5.0)
    cases = [
        run_case(
            recovery_policy,
            climbing_policy,
            pre_fall_steps=112,
            lateral_bias_n=bias,
            seed=43 + index,
            continue_climbing=index == 0,
            continued_climb_segments=2,
        )
        for index, bias in enumerate(fall_biases)
    ]
    prior_only = run_case(
        None,
        climbing_policy,
        pre_fall_steps=112,
        lateral_bias_n=4.0,
        seed=47,
    )
    gates = {
        "all_tested_fall_variants_recover": all(
            row["recovered"] for row in cases
        ),
        "all_regrasps_settle_stably": all(row["regrasp_stable"] for row in cases),
        "fall_arrest_limits_initial_slide": all(
            row["fall_arrest_progress_loss_m"] < 0.08 for row in cases
        ),
        "no_floating_base_teleports": all(
            row["root_teleports_after_fall_start"] == 0 for row in cases
        ),
        "motor_catalog_limits_enforced": all(
            row["peak_motor_torque_ratio"] <= 1.00001 for row in cases
        ),
        "contact_penetration_below_50_mm": all(
            row["maximum_contact_penetration_m"] < 0.050 for row in cases
        ),
        "rope_never_crosses_robot_core": all(
            row["rope_core_collision_frames"] == 0 for row in cases
        ),
        "hand_never_penetrates_rope": all(
            row["hand_rope_penetration_frames"] == 0 for row in cases
        ),
        "unchanged_wbc_prior_fails_post_climb_case": not prior_only["recovered"],
        "nominal_integrated_rollout_continues_uphill": bool(
            cases[0]["continued_climb_success"]
            and cases[0]["continued_climb_segments_completed"] == 2
            and cases[0]["continued_climb_ascent_m"] >= 3.00
            and cases[0]["net_post_fall_progress_m"] >= 1.90
            and cases[0]["continued_climb_rope_core_collision_steps"] == 0
            and cases[0]["continued_climb_hand_rope_penetration_steps"] == 0
        ),
    }
    result = {
        "passed": all(gates.values()),
        "recovery_checkpoint": str(recovery_checkpoint),
        "climbing_checkpoint": str(climbing_checkpoint),
        "summary": {
            "recovered_cases": sum(row["recovered"] for row in cases),
            "cases": len(cases),
            "stable_regrasps": sum(row["regrasp_stable"] for row in cases),
            "maximum_fall_arrest_loss_m": max(
                row["fall_arrest_progress_loss_m"] for row in cases
            ),
            "maximum_motor_torque_ratio": max(
                row["peak_motor_torque_ratio"] for row in cases
            ),
            "maximum_contact_penetration_m": max(
                row["maximum_contact_penetration_m"] for row in cases
            ),
        },
        "gates": gates,
        "evaluation_scope": {
            "terrain_slope_degrees": 28.0,
            "pre_fall_climbing_policy_steps": 112,
            "disturbance_duration_s": 0.4,
            "downslope_pelvis_force_n": 100.0,
            "torso_pitch_torque_nm": 86.0,
            "tested_lateral_biases_n": list(fall_biases),
            "grounded_recovery_lanyard_slack_m": 1.1,
            "continuous_post_recovery_climb_segments": 2,
        },
        "learned_cases": cases,
        "ablation": {"unchanged_wbc_prior": prior_only},
        "policy_claim": (
            "The WBC prior is pretrained and pinned; the four-action PPO "
            "fall-to-climb transfer policy is trained in this repository."
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-model")
    parser.add_argument("--climbing-model")
    parser.add_argument(
        "--output",
        default="models/ppo_mountain_recovery/evaluation_report.json",
    )
    args = parser.parse_args()
    report = evaluate(args.recovery_model, args.climbing_model, args.output)
    print(json.dumps(report["summary"], indent=2))
    print(json.dumps(report["gates"], indent=2))
    if not report["passed"]:
        raise SystemExit("One or more integrated recovery gates failed")
