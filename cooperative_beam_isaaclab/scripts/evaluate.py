#!/usr/bin/env python3
"""Evaluate a frozen cooperative policy for a finite number of Isaac Lab steps."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import gymnasium as gym
import torch
from isaaclab_tasks.utils import (
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

import cooperative_beam_isaaclab  # noqa: F401
from cooperative_beam_isaaclab.tasks import TASK_ID
from cooperative_beam_isaaclab.tasks.parameter_sharing import install_parameter_shared_runner, load_actor_only

parser = argparse.ArgumentParser(description="Finite frozen-policy evaluation for cooperative G1 transport.")
parser.add_argument("--task", default=TASK_ID)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=1200, help="Total vector control steps (600 is one 12 s episode).")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--team_size",
    type=int,
    default=None,
    help="Override the nominal team size. This requires --actor_only for a cross-team checkpoint.",
)
parser.add_argument("--payload_mass", type=float, default=None, help="Fix payload mass in kilograms.")
parser.add_argument(
    "--transport_scale",
    type=float,
    default=1.0,
    help="Fraction of the configured translation/yaw target to evaluate (0=lift only, 1=full move).",
)
parser.add_argument(
    "--actor_only",
    action="store_true",
    help="Load only the shared attention actor; use for zero-shot transfer to a different team size.",
)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]

EPISODE_METRICS = (
    "Metrics/success_rate",
    "Metrics/cooperative_time_ratio",
    "Metrics/mean_payload_jerk_mps3",
    "Metrics/final_payload_position_error_m",
    "Metrics/mean_transport_load_cv",
    "Metrics/mean_transport_load_target_rmse",
    "Metrics/episode_peak_sling_tension_n",
    "Metrics/episode_payload_kg_per_robot",
    "Metrics/episode_payload_kg_per_arm",
)


def _scalar(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        return float(value.float().mean().item())
    if isinstance(value, int | float):
        return float(value)
    return None


def _team_mask(signals: dict[str, torch.Tensor], possible_agents: list[str]) -> torch.Tensor:
    return torch.stack([signals[agent].reshape(-1).bool() for agent in possible_agents], dim=0).any(dim=0)


def main() -> None:
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if args_cli.steps < 1:
        raise ValueError("--steps must be positive")
    if not 0.0 <= args_cli.transport_scale <= 1.0:
        raise ValueError("--transport_scale must be between 0 and 1")

    env_cfg, experiment_cfg = resolve_task_config(args_cli.task, "skrl_mappo_cfg_entry_point")
    with launch_simulation(env_cfg, args_cli):
        install_parameter_shared_runner()
        from isaaclab.utils.seed import configure_seed
        from isaaclab_rl.skrl import SkrlVecEnvWrapper
        from skrl.utils.runner.torch import Runner

        if args_cli.team_size is not None:
            env_cfg.configure_team_size(args_cli.team_size)
        if args_cli.payload_mass is not None:
            env_cfg.curriculum_start_mass = args_cli.payload_mass
            env_cfg.curriculum_end_mass = args_cli.payload_mass
        env_cfg.transport_scale_override = args_cli.transport_scale
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device or env_cfg.sim.device
        experiment_cfg["seed"] = args_cli.seed if args_cli.seed != -1 else random.randint(0, 10_000)
        env_cfg.seed = experiment_cfg["seed"]
        env_cfg.log_dir = os.devnull

        raw_env = gym.make(args_cli.task, cfg=env_cfg)
        env = SkrlVecEnvWrapper(raw_env, ml_framework="torch")
        experiment_cfg["trainer"]["close_environment_at_exit"] = False
        experiment_cfg["agent"]["experiment"]["write_interval"] = 0
        experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        runner = Runner(env, experiment_cfg)
        if args_cli.deterministic:
            configure_seed(env_cfg.seed, True)

        if args_cli.actor_only:
            load_actor_only(runner.agent, str(checkpoint))
            load_mode = "actor_only"
        else:
            runner.agent.load(str(checkpoint))
            load_mode = "full_checkpoint"
        runner.agent.enable_training_mode(False, apply_to_models=True)

        observations, _ = env.reset()
        states = env.state()
        possible_agents = list(env.possible_agents)
        reward_sum = 0.0
        reward_values = 0
        completed_episodes = 0
        terminated_episodes = 0
        timed_out_episodes = 0
        failure_counts = {name: 0 for name in raw_env.unwrapped._task_failure_terms()}
        metric_sums = {name: 0.0 for name in EPISODE_METRICS}
        metric_weights = {name: 0 for name in EPISODE_METRICS}

        print(
            "[EVAL] "
            f"task={args_cli.task} payload={raw_env.unwrapped.cfg.payload_label!r} "
            f"team={raw_env.unwrapped.num_robots} mass={raw_env.unwrapped._payload_masses.mean().item():.2f}kg "
            f"transport_scale={args_cli.transport_scale:.2f} steps={args_cli.steps} load={load_mode}"
        )
        with torch.inference_mode():
            for step in range(args_cli.steps):
                outputs = runner.agent.act(observations, states, timestep=0, timesteps=0)
                actions = {
                    agent: outputs[-1][agent].get("mean_actions", outputs[0][agent]) for agent in possible_agents
                }
                observations, rewards, terminated, truncated, infos = env.step(actions)
                states = env.state()

                for reward in rewards.values():
                    reward_sum += float(reward.sum().item())
                    reward_values += reward.numel()

                terminated_mask = _team_mask(terminated, possible_agents)
                truncated_mask = _team_mask(truncated, possible_agents)
                completed_mask = terminated_mask | truncated_mask
                finished_now = int(completed_mask.sum().item())
                completed_episodes += finished_now
                terminated_episodes += int(terminated_mask.sum().item())
                timed_out_episodes += int(truncated_mask.sum().item())
                for name, mask in raw_env.unwrapped._last_failure_terms.items():
                    failure_counts[name] += int(mask.sum().item())

                if finished_now:
                    log = infos.get("log", {}) if isinstance(infos, dict) else {}
                    for name in EPISODE_METRICS:
                        value = _scalar(log.get(name))
                        if value is not None:
                            metric_sums[name] += finished_now * value
                            metric_weights[name] += finished_now
                if step % 300 == 299 or step == args_cli.steps - 1:
                    print(
                        f"[EVAL] progress={step + 1}/{args_cli.steps} episodes={completed_episodes} "
                        f"terminated={terminated_episodes} timed_out={timed_out_episodes}"
                    )

        episode_metrics = {
            name.removeprefix("Metrics/"): (
                metric_sums[name] / metric_weights[name] if metric_weights[name] else None
            )
            for name in EPISODE_METRICS
        }
        result = {
            "task": args_cli.task,
            "payload": raw_env.unwrapped.cfg.payload_label,
            "checkpoint": checkpoint.name,
            "load_mode": load_mode,
            "seed": experiment_cfg["seed"],
            "num_envs": args_cli.num_envs,
            "steps": args_cli.steps,
            "team_size": raw_env.unwrapped.num_robots,
            "payload_mass_kg": float(raw_env.unwrapped._payload_masses.mean().item()),
            "kg_per_robot": float(raw_env.unwrapped._payload_masses.mean().item()) / raw_env.unwrapped.num_robots,
            "transport_scale": args_cli.transport_scale,
            "mean_reward_per_agent_step": reward_sum / max(reward_values, 1),
            "completed_episodes": completed_episodes,
            "terminated_episodes": terminated_episodes,
            "timed_out_episodes": timed_out_episodes,
            "failure_counts": failure_counts,
            "episode_metrics": episode_metrics,
        }
        print("[EVAL_RESULT] " + json.dumps(result, sort_keys=True))
        env.close()


if __name__ == "__main__":
    main()
