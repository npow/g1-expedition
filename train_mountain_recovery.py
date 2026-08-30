"""Train a slope-specific PPO residual for fixed-line fall recovery."""

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

from mountain_recovery import MountainRecoveryEnv


def make_env(rank: int, seed: int, randomize: bool = True):
    def _init():
        env = MountainRecoveryEnv(rank=rank, randomize=randomize)
        monitored = Monitor(
            env,
            info_keywords=(
                "success",
                "stand_score",
                "pelvis_normal_height_m",
                "torso_upright",
                "left_boot_contact",
                "right_boot_contact",
                "peak_line_load_n",
                "peak_lateral_guide_load_n",
                "rope_core_collision_frames",
                "hand_rope_penetration_frames",
                "peak_motor_torque_ratio",
            ),
        )
        monitored.reset(seed=seed + rank)
        return monitored

    return _init


def train(
    total_timesteps: int = 600_000,
    num_envs: int = 16,
    save_dir: str = "models/ppo_mountain_recovery",
    seed: int = 43,
    device: str = "auto",
    resume: str | None = None,
) -> Path:
    output = Path(save_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output / "best").mkdir(parents=True, exist_ok=True)
    (output / "evaluation").mkdir(parents=True, exist_ok=True)
    Path("logs/tb").mkdir(parents=True, exist_ok=True)
    set_random_seed(seed)
    constructors = [make_env(i, seed, True) for i in range(num_envs)]
    env = SubprocVecEnv(constructors) if num_envs > 1 else DummyVecEnv(constructors)
    eval_env = DummyVecEnv([make_env(10_000, seed, False)])

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // num_envs, 1),
        save_path=str(output / "checkpoints"),
        name_prefix="g1_mountain_recovery",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output / "best"),
        log_path=str(output / "evaluation"),
        eval_freq=max(50_000 // num_envs, 1),
        n_eval_episodes=6,
        deterministic=True,
        render=False,
    )
    selected_device = device
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
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
                "activation_fn": torch.nn.ELU,
                "log_std_init": -1.35,
            },
            learning_rate=5e-5,
            n_steps=512,
            batch_size=1024,
            n_epochs=4,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.002,
            vf_coef=0.6,
            max_grad_norm=0.7,
            target_kl=0.025,
            verbose=1,
            tensorboard_log="logs/tb",
            seed=seed,
            device=selected_device,
        )
        # The initial residual mean is exactly zero: training starts from the
        # upstream WBC prior instead of a random command offset.
        torch.nn.init.zeros_(model.policy.action_net.weight)
        torch.nn.init.zeros_(model.policy.action_net.bias)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=False,
        reset_num_timesteps=not bool(resume),
    )
    last = output / "g1_mountain_recovery_last"
    model.save(last)
    best = output / "best" / "best_model.zip"
    final = output / "g1_mountain_recovery_final.zip"
    source = best if best.exists() else last.with_suffix(".zip")
    shutil.copy2(source, final)
    selected = PPO.load(final, device="cpu")
    metadata = {
        "algorithm": "hierarchical PPO handoff policy over a pretrained WBC prior",
        "training_run_target_timesteps": total_timesteps,
        "cumulative_training_timesteps": int(model.num_timesteps),
        "selected_checkpoint_timesteps": int(selected.num_timesteps),
        "num_parallel_envs": num_envs,
        "seed": seed,
        "device": selected_device,
        "observation_dim": int(model.observation_space.shape[0]),
        "action_dim": int(model.action_space.shape[0]),
        "actions": [
            "left_leg_motion_brake",
            "right_leg_motion_brake",
            "waist_motion_brake",
            "arms_motion_brake",
        ],
        "terrain": "28-degree fixed-line snow/ice slope",
        "physics": (
            "MuJoCo gravity, articulated contacts, torque-capped motors, "
            "deformable rope, and equal/opposite one-way cam reaction"
        ),
        "hidden_recovery_support": False,
        "floating_base_prescribed_after_reset": False,
        "prior": {
            "source": "wbc-mjlab/wbc-g1-deploy",
            "commit": "6dabf86fddc2b7b429b09e74999732fcde3441f9",
            "claim": "pretrained upstream prior; slope residual trained here",
        },
        "resumed_from": resume,
        "selection_source": str(source),
        "cloud_job_id": os.environ.get("JOB_ID"),
        "cloud_accelerator": os.environ.get("ACCELERATOR"),
        "cloud_cpu_cores": os.environ.get("CPU_CORES"),
    }
    (output / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    env.close()
    eval_env.close()
    print(final)
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=600_000)
    parser.add_argument("--num-envs", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--save-dir", default="models/ppo_mountain_recovery")
    parser.add_argument("--seed", type=int, default=43)
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
