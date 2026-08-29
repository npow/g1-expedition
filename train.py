"""Train the Unitree G1 self-arrest policy with PPO."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn, set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from himalaya_env import G1SelfArrestEnv
from diverse_scenarios import SCENARIOS


CURRICULUM = (
    (0, (0.25, 0.75)),
    (250_000, (0.50, 1.00)),
    (500_000, (1.00, 2.00)),
    (800_000, (2.00, 3.00)),
    (1_100_000, (3.00, 4.00)),
    (1_400_000, (4.00, 5.00)),
)

DIVERSITY_CURRICULUM = (
    ((-5.0, 5.0), (-0.20, 0.20), (-2.0, 2.0)),
    ((-10.0, 10.0), (-0.40, 0.40), (-3.0, 3.0)),
    ((-16.0, 16.0), (-0.65, 0.65), (-4.0, 4.0)),
    ((-24.0, 24.0), (-0.90, 0.90), (-6.0, 6.0)),
    ((-32.0, 32.0), (-1.20, 1.20), (-8.0, 8.0)),
    ((-40.0, 40.0), (-1.50, 1.50), (-10.0, 10.0)),
)

_BASE_TRAINING_ANCHORS = tuple(
    (
        scenario.speed_mps,
        scenario.heading_degrees,
        scenario.lateral_speed_mps,
        scenario.roll_degrees,
    )
    for scenario in SCENARIOS
)

# The final two cases combine heading, lateral motion, and roll with opposing
# signs.  They are sparse corners of a four-dimensional reset distribution and
# exposed the two remaining failure modes (left-hand release and shallow pick
# entry), so deliberately oversample their neighborhoods during the finishing
# stage without removing uniform full-envelope resets.
TRAINING_ANCHORS = _BASE_TRAINING_ANCHORS + tuple(
    anchor for anchor in _BASE_TRAINING_ANCHORS[-2:] for _ in range(4)
)


class VelocityCurriculumCallback(BaseCallback):
    """Increase reset velocity only after the contact skill has time to form."""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self._stage = -1

    def _on_step(self) -> bool:
        stage = max(i for i, (threshold, _) in enumerate(CURRICULUM) if self.num_timesteps >= threshold)
        if stage != self._stage:
            speed_range = CURRICULUM[stage][1]
            heading_range, lateral_range, roll_range = DIVERSITY_CURRICULUM[stage]
            self.training_env.env_method("set_initial_speed_range", speed_range)
            self.training_env.env_method(
                "set_reset_diversity", heading_range, lateral_range, roll_range
            )
            print(
                f"Curriculum stage {stage + 1}: speed {speed_range[0]}-{speed_range[1]} m/s, "
                f"heading {heading_range[0]}..{heading_range[1]} deg, "
                f"lateral {lateral_range[0]}..{lateral_range[1]} m/s, "
                f"roll {roll_range[0]}..{roll_range[1]} deg"
            )
            self._stage = stage
        return True


def make_env(
    rank: int,
    seed: int,
    randomize: bool = True,
    initial_speed_range: tuple[float, float] = (4.0, 5.0),
    heading_range_degrees: tuple[float, float] = (0.0, 0.0),
    lateral_speed_range: tuple[float, float] = (0.0, 0.0),
    roll_range_degrees: tuple[float, float] = (0.0, 0.0),
    anchor_probability: float = 0.0,
):
    def _init():
        env = G1SelfArrestEnv(
            randomize_reset=randomize,
            initial_speed_range=initial_speed_range,
            heading_range_degrees=heading_range_degrees,
            lateral_speed_range=lateral_speed_range,
            roll_range_degrees=roll_range_degrees,
            anchor_resets=TRAINING_ANCHORS,
            anchor_probability=anchor_probability,
        )
        env = Monitor(
            env,
            info_keywords=(
                "success",
                "v_slope",
                "f_pick",
                "stopping_distance",
                "pick_contact_fraction",
                "left_grasp_contact_fraction",
                "right_grasp_contact_fraction",
                "grip_score",
                "valid_learned_plant_motion",
                "stroke_at_first_contact",
                "blade_angle_at_first_contact_deg",
            ),
        )
        env.reset(seed=seed + rank)
        return env

    return _init


def train(
    total_timesteps: int = 600_000,
    num_envs: int = 12,
    save_dir: str = "models/ppo_self_arrest",
    seed: int = 7,
    device: str = "auto",
    resume: str | None = None,
    restart_timesteps: bool = False,
    learning_rate: float = 3e-4,
    engaged_action_init: bool = True,
    anchor_probability: float = 0.0,
    checkpoint_interval: int = 50_000,
) -> Path:
    output = Path(save_dir)
    output.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    set_random_seed(seed)

    if num_envs > 1:
        env = SubprocVecEnv(
            [
                make_env(
                    i,
                    seed,
                    randomize=True,
                    initial_speed_range=CURRICULUM[0][1],
                    heading_range_degrees=DIVERSITY_CURRICULUM[0][0],
                    lateral_speed_range=DIVERSITY_CURRICULUM[0][1],
                    roll_range_degrees=DIVERSITY_CURRICULUM[0][2],
                    anchor_probability=anchor_probability,
                )
                for i in range(num_envs)
            ]
        )
    else:
        env = DummyVecEnv(
            [
                make_env(
                    0,
                    seed,
                    randomize=True,
                    initial_speed_range=CURRICULUM[0][1],
                    heading_range_degrees=DIVERSITY_CURRICULUM[0][0],
                    lateral_speed_range=DIVERSITY_CURRICULUM[0][1],
                    roll_range_degrees=DIVERSITY_CURRICULUM[0][2],
                    anchor_probability=anchor_probability,
                )
            ]
        )
    eval_env = DummyVecEnv(
        [
            make_env(
                10_000,
                seed,
                randomize=True,
                initial_speed_range=CURRICULUM[-1][1],
                heading_range_degrees=DIVERSITY_CURRICULUM[-1][0],
                lateral_speed_range=DIVERSITY_CURRICULUM[-1][1],
                roll_range_degrees=DIVERSITY_CURRICULUM[-1][2],
            )
        ]
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(checkpoint_interval // num_envs, 1),
        save_path=str(output / "checkpoints"),
        name_prefix="g1_self_arrest",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output / "best"),
        log_path=str(output / "evaluation"),
        eval_freq=max(100_000 // num_envs, 1),
        n_eval_episodes=3,
        deterministic=True,
        render=False,
    )

    selected_device = device
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training PPO with {num_envs} MuJoCo workers; network device={selected_device}")

    if resume:
        print(f"Continuing PPO from {resume}")
        model = PPO.load(
            resume,
            env=env,
            device=selected_device,
            tensorboard_log="logs/tb",
        )
        model.learning_rate = learning_rate
        model.lr_schedule = get_schedule_fn(learning_rate)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs={
                "net_arch": {"pi": [256, 256], "vf": [256, 256]},
                "activation_fn": torch.nn.Tanh,
                "log_std_init": -0.35,
            },
            learning_rate=learning_rate,
            n_steps=512,
            batch_size=512,
            n_epochs=8,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.002,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log="logs/tb",
            seed=seed,
            device=selected_device,
        )
        if engaged_action_init:
            reference_env = G1SelfArrestEnv(randomize_reset=False)
            engaged_action = (
                reference_env.engaged_qpos[
                    reference_env.POLICY_ACTUATOR_IDS
                ]
                - reference_env.ready_qpos[
                    reference_env.POLICY_ACTUATOR_IDS
                ]
            ) / reference_env.action_scale
            reference_env.close()
            with torch.no_grad():
                model.policy.action_net.weight.mul_(0.01)
                model.policy.action_net.bias.copy_(
                    torch.as_tensor(
                        engaged_action,
                        dtype=model.policy.action_net.bias.dtype,
                        device=model.policy.action_net.bias.device,
                    )
                )
            print(
                "Initialized the stochastic policy near the two-arm engaged "
                "reference; PPO remains free to update every action channel."
            )
    model.learn(
        total_timesteps=total_timesteps,
        callback=[VelocityCurriculumCallback(), checkpoint_callback, eval_callback],
        progress_bar=True,
        reset_num_timesteps=resume is None or restart_timesteps,
    )

    last_path = output / "g1_self_arrest_last"
    model.save(last_path)
    best_path = output / "best" / "best_model.zip"
    final_path = output / "g1_self_arrest_final.zip"
    if best_path.exists():
        shutil.copy2(best_path, final_path)
        selection_source = str(best_path)
    else:
        shutil.copy2(last_path.with_suffix(".zip"), final_path)
        selection_source = str(last_path.with_suffix(".zip"))
    selected_model = PPO.load(final_path, device="cpu")
    metadata = {
        "algorithm": "PPO",
        "training_run_target_timesteps": total_timesteps,
        "selected_checkpoint_timesteps": int(selected_model.num_timesteps),
        "num_parallel_envs": num_envs,
        "seed": seed,
        "device": selected_device,
        "torch_build": torch.__version__,
        "cloud_compute_used": False,
        "observation_dim": int(model.observation_space.shape[0]),
        "action_dim": int(model.action_space.shape[0]),
        "policy_network": [256, 256],
        "controlled_actuator_ids": G1SelfArrestEnv.POLICY_ACTUATOR_IDS.tolist(),
        "controlled_joints": [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
        "action_scale_radians": G1SelfArrestEnv.ACTION_SCALE.tolist(),
        "velocity_curriculum": CURRICULUM,
        "reset_diversity_curriculum": DIVERSITY_CURRICULUM,
        "resumed_from": resume,
        "restarted_timestep_counter": restart_timesteps,
        "learning_rate": learning_rate,
        "engaged_action_initialization": bool(
            engaged_action_init and resume is None
        ),
        "training_anchor_probability": anchor_probability,
        "checkpoint_interval": checkpoint_interval,
        "training_anchor_jitter": {
            "downhill_speed_mps": 0.10,
            "heading_degrees": 2.0,
            "lateral_speed_mps": 0.10,
            "roll_degrees": 1.0
        },
        "selection_source": selection_source,
        "last_checkpoint": str(last_path.with_suffix(".zip")),
        "selection_rule": (
            "best deterministic EvalCallback checkpoint, followed by the "
            "contact-based gates in evaluate_policy.py"
        ),
    }
    (output / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    env.close()
    eval_env.close()
    print(f"Selected learned policy from {selection_source} and saved it to {final_path}")
    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=600_000)
    parser.add_argument("--num-envs", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--save-dir", default="models/ppo_self_arrest")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume")
    parser.add_argument(
        "--restart-timesteps",
        action="store_true",
        help="Restart the curriculum clock at zero while retaining resumed weights.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--anchor-probability",
        type=float,
        default=0.0,
        help=(
            "Probability of sampling a jittered named-scenario reset instead "
            "of a continuous-uniform reset."
        ),
    )
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument(
        "--no-engaged-action-init",
        action="store_true",
        help="Do not initialize a new policy near the kinematic engaged-action reference.",
    )
    args = parser.parse_args()
    train(
        args.timesteps,
        args.num_envs,
        args.save_dir,
        args.seed,
        args.device,
        args.resume,
        args.restart_timesteps,
        args.learning_rate,
        not args.no_engaged_action_init,
        args.anchor_probability,
        args.checkpoint_interval,
    )
