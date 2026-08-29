#!/usr/bin/env python3
"""Create the environment and take a finite number of zero or random steps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import gymnasium as gym
import torch
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_tasks.utils import (
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

import cooperative_beam_isaaclab  # noqa: F401
from cooperative_beam_isaaclab.tasks import TASK_ID
from cooperative_beam_isaaclab.tasks.trajectory import rescue_trajectory

parser = argparse.ArgumentParser(description="Finite physics smoke test for the cooperative G1 task.")
parser.add_argument("--task", default=TASK_ID)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=180)
parser.add_argument(
    "--team_size",
    type=int,
    default=None,
    help="Override the task's nominal robot count while holding payload geometry fixed.",
)
parser.add_argument(
    "--payload_mass",
    type=float,
    default=None,
    help="Fix the payload mass in kilograms instead of sampling its curriculum.",
)
parser.add_argument("--random_actions", action="store_true")
parser.add_argument(
    "--scripted_lift",
    action="store_true",
    help="Ramp both wrist-z commands upward to exercise AGILE + IK + sling coupling.",
)
parser.add_argument(
    "--scripted_transport",
    action="store_true",
    help="Track the full lift/carry/turn/lower trajectory with a non-learned formation controller.",
)
parser.add_argument(
    "--pretension",
    type=float,
    default=0.0,
    help="Shorten every sling by this many metres after calibration (physics diagnostic).",
)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]


def main() -> None:
    if args_cli.scripted_lift and args_cli.scripted_transport:
        raise ValueError("Choose either --scripted_lift or --scripted_transport")
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    with launch_simulation(env_cfg, args_cli):
        if args_cli.team_size is not None:
            env_cfg.configure_team_size(args_cli.team_size)
        if args_cli.payload_mass is not None:
            env_cfg.curriculum_start_mass = args_cli.payload_mass
            env_cfg.curriculum_end_mass = args_cli.payload_mass
        if args_cli.scripted_transport:
            env_cfg.transport_scale_override = 1.0
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device or env_cfg.sim.device
        env = gym.make(args_cli.task, cfg=env_cfg)
        observations, _ = env.reset()
        print("[SMOKE] agents:", env.unwrapped.possible_agents)
        print(
            f"[SMOKE] payload={env.unwrapped.cfg.payload_label!r} "
            f"mass={env.unwrapped._payload_masses.mean().item():.1f}kg "
            f"team={env.unwrapped.num_robots} "
            f"kg/robot={env.unwrapped._payload_masses.mean().item() / env.unwrapped.num_robots:.2f}"
        )
        print(
            f"[SMOKE] sling_station_x={tuple(env.unwrapped.cfg.sling_station_x)} "
            f"target_load_ratios={tuple(round(value, 4) for value in env.unwrapped._expected_load_ratios.tolist())}"
        )
        print("[SMOKE] G1 body names:", env.unwrapped.robots[0].body_names)
        print("[SMOKE] observation shapes:", {key: tuple(value.shape) for key, value in observations.items()})

        initial_beam_z = env.unwrapped.beam.data.root_pos_w.torch[:, 2].clone()
        peak_beam_z = initial_beam_z.clone()
        peak_tension = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
        termination_count = torch.zeros(env.unwrapped.num_envs, dtype=torch.long, device=env.unwrapped.device)
        failure_counts: dict[str, int] = {}
        station_load_sum = torch.zeros(env.unwrapped.num_robots, device=env.unwrapped.device)

        for step in range(args_cli.steps):
            if step == 1 and args_cli.pretension > 0.0:
                env.unwrapped._sling_rest_lengths -= args_cli.pretension
                print(f"[SMOKE] applied {args_cli.pretension:.3f}m diagnostic sling pretension")
            actions = {
                agent: (
                    0.12
                    * (
                        2.0
                        * torch.rand(
                            (env.unwrapped.num_envs, env.unwrapped.cfg.action_spaces[agent]),
                            device=env.unwrapped.device,
                        )
                        - 1.0
                    )
                    if args_cli.random_actions
                    else torch.zeros(
                        (env.unwrapped.num_envs, env.unwrapped.cfg.action_spaces[agent]),
                        device=env.unwrapped.device,
                    )
                )
                for agent in env.unwrapped.possible_agents
            }
            if args_cli.scripted_lift:
                target_lift_command = min(
                    env.unwrapped.cfg.lift_height / env.unwrapped.cfg.wrist_action_scale[2],
                    1.0,
                )
                lift_command = min(step / 60.0, 1.0) * target_lift_command
                for action in actions.values():
                    action[:, 6] = lift_command
                    action[:, 9] = lift_command
            if args_cli.scripted_transport:
                unwrapped = env.unwrapped
                progress = unwrapped.episode_length_buf.float() / float(unwrapped.max_episode_length)
                next_progress = torch.clamp(
                    (unwrapped.episode_length_buf.float() + 1.0) / float(unwrapped.max_episode_length), max=1.0
                )
                start = torch.tensor(
                    unwrapped.cfg.beam_cfg.init_state.pos, device=unwrapped.device
                ).repeat(unwrapped.num_envs, 1)
                target, target_yaw = rescue_trajectory(
                    progress,
                    start,
                    unwrapped.cfg.lift_height,
                    unwrapped.cfg.carry_delta_xy,
                    unwrapped.cfg.final_beam_height,
                    unwrapped.cfg.target_yaw,
                )
                next_target, next_yaw = rescue_trajectory(
                    next_progress,
                    start,
                    unwrapped.cfg.lift_height,
                    unwrapped.cfg.carry_delta_xy,
                    unwrapped.cfg.final_beam_height,
                    unwrapped.cfg.target_yaw,
                )
                control_dt = unwrapped.cfg.sim.dt * unwrapped.cfg.decimation
                payload_velocity_w = (next_target - target) / control_dt
                yaw_rate = (next_yaw - target_yaw) / control_dt
                cos_yaw = torch.cos(target_yaw)
                sin_yaw = torch.sin(target_yaw)
                beam_start_xy = start[0, :2]
                for index, (agent, robot_cfg, robot) in enumerate(
                    zip(
                        unwrapped.possible_agents,
                        unwrapped.cfg.robot_cfgs,
                        unwrapped.robots,
                        strict=True,
                    )
                ):
                    relative_xy = torch.tensor(robot_cfg.init_state.pos[:2], device=unwrapped.device) - beam_start_xy
                    rotated_relative = torch.stack(
                        (
                            cos_yaw * relative_xy[0] - sin_yaw * relative_xy[1],
                            sin_yaw * relative_xy[0] + cos_yaw * relative_xy[1],
                        ),
                        dim=-1,
                    )
                    station_velocity_w = payload_velocity_w.clone()
                    station_velocity_w[:, 0] -= yaw_rate * rotated_relative[:, 1]
                    station_velocity_w[:, 1] += yaw_rate * rotated_relative[:, 0]
                    station_velocity_b = quat_apply_inverse(robot.data.root_quat_w.torch, station_velocity_w)
                    actions[agent][:, 0] = station_velocity_b[:, 0] / unwrapped.cfg.command_velocity_scale[0]
                    actions[agent][:, 1] = station_velocity_b[:, 1] / unwrapped.cfg.command_velocity_scale[1]
                    actions[agent][:, 2] = yaw_rate / unwrapped.cfg.command_velocity_scale[2]
                    wrist_z = (target[:, 2] - start[:, 2]) / unwrapped.cfg.wrist_action_scale[2]
                    actions[agent][:, 6] = wrist_z
                    actions[agent][:, 9] = wrist_z
                    actions[agent].clamp_(-1.0, 1.0)
            observations, rewards, terminated, truncated, _ = env.step(actions)
            current_beam_z = env.unwrapped.beam.data.root_pos_w.torch[:, 2]
            peak_beam_z = torch.maximum(peak_beam_z, current_beam_z)
            peak_tension = torch.maximum(peak_tension, env.unwrapped._sling_tensions.amax(dim=(1, 2)))
            station_load_sum += env.unwrapped._sling_tensions.sum(dim=-1).mean(dim=0)
            terminated_now = torch.stack(tuple(terminated.values()), dim=0).any(dim=0)
            termination_count += terminated_now.long()
            for failure_name, failure_mask in env.unwrapped._last_failure_terms.items():
                failure_counts[failure_name] = failure_counts.get(failure_name, 0) + int(failure_mask.sum().item())
            if (
                step % 30 == 0
                or step == args_cli.steps - 1
                or terminated_now.any().item()
                or (args_cli.pretension > 0.0 and step < 5)
            ):
                mean_reward = torch.stack([value.mean() for value in rewards.values()]).mean().item()
                beam_z = env.unwrapped.beam.data.root_pos_w.torch[:, 2].mean().item()
                max_tension = env.unwrapped._sling_tensions.max().item()
                print(
                    f"[SMOKE] step={step:04d} reward={mean_reward:+.3f} "
                    f"beam_z={beam_z:.3f}m max_tension={max_tension:.1f}N "
                    f"terminated={any(value.any().item() for value in terminated.values())} "
                    f"truncated={any(value.any().item() for value in truncated.values())}"
                )
        lift_gain = peak_beam_z - initial_beam_z
        mean_station_loads = station_load_sum / args_cli.steps
        station_x = torch.tensor(env.unwrapped.cfg.sling_station_x, device=env.unwrapped.device)
        negative_side_load = mean_station_loads[station_x < 0.0].sum().item()
        positive_side_load = mean_station_loads[station_x > 0.0].sum().item()
        print(
            f"[SMOKE] summary mean_lift_gain={lift_gain.mean().item():.3f}m "
            f"min_lift_gain={lift_gain.min().item():.3f}m "
            f"peak_tension={peak_tension.max().item():.1f}N "
            f"terminations={termination_count.sum().item()} "
            f"failure_counts={failure_counts}"
        )
        print(
            f"[SMOKE] summary mean_station_loads_n="
            f"{[round(value, 2) for value in mean_station_loads.tolist()]} "
            f"mean_side_loads_n={{'negative_x': {negative_side_load:.2f}, "
            f"'positive_x': {positive_side_load:.2f}}}"
        )
        env.close()
        print("[SMOKE] PASS")


if __name__ == "__main__":
    main()
