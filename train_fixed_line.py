"""Train the Unitree G1 fixed-line ascending policy with PPO."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from fixed_line_slope_env import G1FixedLineEnv


def make_env(rank: int, seed: int, randomize: bool = True):
    def _init():
        env = G1FixedLineEnv(randomize_reset=randomize)
        monitored = Monitor(
            env,
            info_keywords=(
                "success",
                "ascent",
                "completed_cycles",
                "descent_from_high_water",
                "chest_load_fraction",
                "foot_loop_load_fraction",
                "upright_score",
                "grasp_score",
                "hand_separation",
                "cross_hand_collision",
                "wall_hand_collision",
                "grounded_fraction",
                "left_boot_contact_fraction",
                "right_boot_contact_fraction",
                "double_support_fraction",
                "maximum_airborne_streak",
                "line_load_fraction",
                "arm_pull_load_fraction",
                "arm_pull_impulse_ns",
                "ground_load_bodyweight",
                "vertical_gain_m",
            ),
        )
        monitored.reset(seed=seed + rank)
        return monitored

    return _init


def train(
    total_timesteps: int = 300_000,
    num_envs: int = 12,
    save_dir: str = "models/ppo_fixed_line_slope",
    seed: int = 31,
    device: str = "auto",
    resume: str | None = None,
) -> Path:
    output = Path(save_dir)
    output.mkdir(parents=True, exist_ok=True)
    Path("logs/tb").mkdir(parents=True, exist_ok=True)
    set_random_seed(seed)

    constructors = [make_env(i, seed, randomize=True) for i in range(num_envs)]
    if num_envs > 1:
        env = SubprocVecEnv(constructors)
    else:
        env = DummyVecEnv(constructors)
    eval_env = DummyVecEnv([make_env(10_000, seed, randomize=False)])

    checkpoint_callback = CheckpointCallback(
        # Dense checkpoints make policy convergence directly inspectable in the
        # learning-progress video instead of showing only the final behavior.
        save_freq=max(20_000 // num_envs, 1),
        save_path=str(output / "checkpoints"),
        name_prefix="g1_fixed_line",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output / "best"),
        log_path=str(output / "evaluation"),
        eval_freq=max(20_000 // num_envs, 1),
        n_eval_episodes=4,
        deterministic=True,
        render=False,
    )

    selected_device = device
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Training fixed-line PPO with {num_envs} MuJoCo workers; "
        f"network device={selected_device}"
    )
    if resume:
        model = PPO.load(
            resume,
            env=env,
            device=selected_device,
            tensorboard_log="logs/tb",
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs={
                "net_arch": {"pi": [256, 256], "vf": [256, 256]},
                "activation_fn": torch.nn.Tanh,
                "log_std_init": -0.25,
            },
            learning_rate=3e-4,
            n_steps=512,
            batch_size=512,
            n_epochs=8,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.004,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log="logs/tb",
            seed=seed,
            device=selected_device,
        )
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        # Keep training runnable from the minimal hackathon requirements;
        # SB3's optional tqdm/rich packages are not needed for optimization.
        progress_bar=False,
        reset_num_timesteps=not bool(resume),
    )

    last_path = output / "g1_fixed_line_last"
    model.save(last_path)
    best_path = output / "best" / "best_model.zip"
    final_path = output / "g1_fixed_line_final.zip"
    if best_path.exists():
        shutil.copy2(best_path, final_path)
        selection_source = best_path
    else:
        selection_source = last_path.with_suffix(".zip")
        shutil.copy2(selection_source, final_path)

    selected_model = PPO.load(final_path, device="cpu")
    metadata = {
        "algorithm": "PPO",
        "training_run_target_timesteps": total_timesteps,
        "cumulative_training_timesteps": int(model.num_timesteps),
        "selected_checkpoint_timesteps": int(selected_model.num_timesteps),
        "num_parallel_envs": num_envs,
        "seed": seed,
        "device": selected_device,
        "torch_build": torch.__version__,
        "cloud_compute_used": False,
        "observation_dim": int(model.observation_space.shape[0]),
        "action_dim": int(model.action_space.shape[0]),
        "policy_network": [256, 256],
        "actions": [
            "request_left_uphill_step",
            "request_right_uphill_step",
            "right_arm_jumar_pull",
        ],
        "arm_controller": (
            "world-frame damped-least-squares IK retaining a right-hand "
            "ascender grasp plus learned force-balanced Jumar loading; "
            "relaxed left balance arm"
        ),
        "terrain": "28-degree inclined snow/ice slope",
        "slope_angle_degrees": 28.0,
        "target_along_slope_m": 1.50,
        "uphill_force_source": (
            "coordinated position-actuated leg motion through MuJoCo "
            "boot/slope contact and a learned right-wrist Jumar pull with "
            "equal-and-opposite reaction on the deformable fixed line"
        ),
        "resumed_from": resume,
        "selection_source": str(selection_source),
        "selection_rule": (
            "best deterministic EvalCallback reward, followed by the causal "
            "and contact-load gates in evaluate_fixed_line.py"
        ),
    }
    (output / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    env.close()
    eval_env.close()
    print(f"Selected fixed-line policy from {selection_source} and saved {final_path}")
    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--num-envs", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--save-dir", default="models/ppo_fixed_line_slope")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume")
    args = parser.parse_args()
    train(
        args.timesteps,
        args.num_envs,
        args.save_dir,
        args.seed,
        args.device,
        args.resume,
    )
