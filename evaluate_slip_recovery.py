"""Ablation harness for induced-slip recovery on the fixed line.

Answers one question: when the robot is knocked off its stride, what puts it
back -- the policy, or the external balance assist?

The assist is an uncapped orientation PD (gain 420) holding the pelvis at a
target quaternion plus an uncapped lateral spring (700 N/m) pinning it to y=0.
Righting a stumbling humanoid is precisely their job. So the headline number is
not a single recovery rate; it is the recovery rate AS A FUNCTION OF ASSIST
SCALE. If recovery collapses as the assist is dialled out, the assist was doing
it. That is the whole design.

Controls, both required for the table to mean anything:

  disturb=off, assist=1.0   reproduces the parent env exactly. If this is not
                            ~100%, the harness is broken, not the policy.
  disturb=on,  assist=0.0   no stabilizer at all. If the robot cannot even
                            stand here, recovery is not yet a measurable
                            quantity and the intermediate rows are the story.

    python evaluate_slip_recovery.py \
        --model models/ppo_fixed_line_slope/g1_fixed_line_final.zip \
        --episodes 8 --output models/ppo_fixed_line_slope/slip_recovery_report.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

import numpy as np
from stable_baselines3 import PPO

import fixed_line_slope_env as parent_env
import slip_recovery_env


def rollout(env, model, seed: int) -> dict:
    observation, _ = env.reset(seed=seed)
    steps = 0
    info: dict = {}
    for _ in range(env.max_episode_steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, _reward, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            break
    return {
        "seed": seed,
        "steps": steps,
        "success": bool(info.get("success", False)),
        "failure": bool(info.get("failure", False)),
        "ascent_m": float(info.get("ascent", 0.0)),
        "slip_triggered": bool(info.get("slip_triggered", 0.0)),
        "slip_depth_m": float(info.get("slip_depth_m", 0.0)),
        "recovered": bool(info.get("recovered", 0.0)),
        "steps_to_recover": int(info.get("steps_to_recover", -1)),
        "upright_score": float(info.get("upright_score", 0.0)),
    }


def summarize(rows: list[dict]) -> dict:
    slipped = [r for r in rows if r["slip_triggered"]]
    recovered = [r for r in slipped if r["recovered"]]
    times = [r["steps_to_recover"] for r in recovered if r["steps_to_recover"] >= 0]
    return {
        "episodes": len(rows),
        "success_rate": sum(r["success"] for r in rows) / max(len(rows), 1),
        "slip_rate": len(slipped) / max(len(rows), 1),
        # Conditioned on a slip actually happening -- an episode that was never
        # disturbed must not be counted as a successful recovery.
        "recovery_rate": (len(recovered) / len(slipped)) if slipped else None,
        "mean_slip_depth_m": statistics.fmean(r["slip_depth_m"] for r in slipped)
        if slipped
        else 0.0,
        "max_slip_depth_m": max((r["slip_depth_m"] for r in slipped), default=0.0),
        "mean_steps_to_recover": statistics.fmean(times) if times else None,
        "mean_ascent_m": statistics.fmean(r["ascent_m"] for r in rows),
        "mean_upright": statistics.fmean(r["upright_score"] for r in rows),
        "rows": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/ppo_fixed_line_slope/g1_fixed_line_final.zip")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--seed", type=int, default=4100)
    p.add_argument("--slip-mode", choices=("friction", "impulse"), default="friction")
    p.add_argument("--assist-scales", type=float, nargs="+",
                   default=[1.0, 0.5, 0.25, 0.0])
    p.add_argument("--output", default="models/ppo_fixed_line_slope/slip_recovery_report.json")
    args = p.parse_args()

    model = PPO.load(args.model, device="cpu")
    seeds = [args.seed + i for i in range(args.episodes)]
    results: dict[str, dict] = {}

    print("control: parent env, no disturbance", flush=True)
    env = parent_env.G1FixedLineSlopeEnv()
    results["parent_no_disturb"] = summarize([rollout(env, model, s) for s in seeds])
    print(f"  success {results['parent_no_disturb']['success_rate']:.0%}", flush=True)

    for scale in args.assist_scales:
        key = f"disturb_assist_{scale:g}"
        env = slip_recovery_env.load(disturb=True, slip_mode=args.slip_mode,
                                     balance_assist_scale=scale)
        summary = summarize([rollout(env, model, s) for s in seeds])
        results[key] = summary
        rr = summary["recovery_rate"]
        print(f"  assist {scale:<4g} success {summary['success_rate']:>4.0%} "
              f"recovery {('n/a' if rr is None else f'{rr:.0%}'):>4} "
              f"slip {summary['mean_slip_depth_m']:.3f} m "
              f"ascent {summary['mean_ascent_m']:.3f} m", flush=True)

    header = "| Condition | Success | Recovery | Mean slip | Mean ascent |"
    lines = [header, "|---|---:|---:|---:|---:|"]
    s = results["parent_no_disturb"]
    lines.append(f"| parent env, no disturbance | {s['success_rate']:.0%} | n/a | "
                 f"0.000 m | {s['mean_ascent_m']:.3f} m |")
    for scale in args.assist_scales:
        s = results[f"disturb_assist_{scale:g}"]
        rr = "n/a" if s["recovery_rate"] is None else f"{s['recovery_rate']:.0%}"
        lines.append(f"| slip, balance assist &times;{scale:g} | {s['success_rate']:.0%} | "
                     f"{rr} | {s['mean_slip_depth_m']:.3f} m | {s['mean_ascent_m']:.3f} m |")
    table = "\n".join(lines)

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"model": args.model, "episodes": args.episodes, "seed": args.seed,
         "slip_mode": args.slip_mode, "assist_scales": args.assist_scales,
         "conditions": results, "markdown_table": table}, indent=2) + "\n")
    print(f"\nwrote {out}\n\n{table}", flush=True)

    control = results["parent_no_disturb"]["success_rate"]
    if control < 0.9:
        ascent = results["parent_no_disturb"]["mean_ascent_m"]
        print(f"\nWARNING: the no-disturbance control scored {control:.0%} at "
              f"{ascent:.3f} m mean ascent, not ~100% / ~1.51 m. Every row below "
              f"is uninterpretable.", flush=True)
        # Which of the two it is depends on whether this harness has ever
        # produced ~100% here. It has, for the shipped checkpoint -- so a 0%
        # control from a DIFFERENT model means that model cannot walk, not that
        # the harness is broken. Saying "the harness is wrong" unconditionally
        # (as this used to) sends you to debug working code while a broken
        # fine-tune goes unnoticed.
        print("  Check in this order:\n"
              "   1. Does models/ppo_fixed_line_slope/g1_fixed_line_final.zip "
              "still score ~100% here? If yes the harness is fine and THIS "
              "MODEL is broken -- most likely catastrophic forgetting during "
              "fine-tuning.\n"
              "   2. If the shipped checkpoint also fails, the harness or the "
              "env changed underneath it.", flush=True)


if __name__ == "__main__":
    main()
