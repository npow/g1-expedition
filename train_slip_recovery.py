"""Fine-tune the fixed-line policy to survive slips without the crutch.

`evaluate_slip_recovery.py` measured why this is needed. On the shipped
checkpoint:

    balance assist x1.0   100% success, 1.517 m ascent
    balance assist x0.5     0% success, 0.226 m ascent
    balance assist x0.0     0% success, 0.029 m ascent

Halving the external stabilizer costs 85% of the ascent, and the scheduled slip
never even fires because the robot has already collapsed. So the published
result is co-adapted to an uncapped orientation PD (gain 420) and an uncapped
lateral spring (700 N/m). No recovery number measured at x1.0 says anything
about the policy -- the assist performs the recovery.

The fix is not to switch the assist off, which just makes the task impossible
from a policy that never trained without it. It is to ANNEAL it: start where
the shipped checkpoint already works, and withdraw the support as the policy
takes over. Disturbances ramp in on their own schedule, so the robot is not
asked to learn balance and slip recovery in the same step.

Because `G1SlipRecoveryEnv` keeps the 2-D action space, obs stays 113 wide and
the shipped checkpoint loads directly -- this is a fine-tune from 259,896
interactions, not a fresh run. That is the whole reason the action space was
left alone.

    python train_slip_recovery.py \
        --resume models/ppo_fixed_line_slope/g1_fixed_line_final.zip \
        --timesteps 400000 --num-envs 12
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn, set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import slip_recovery_env


def make_env(rank: int, seed: int, slip_mode: str):
    def _init():
        env = slip_recovery_env.load(
            # Both start disabled. The curriculum callback switches them on;
            # having them on at step 0 would ask a policy that cannot yet stand
            # unaided to also recover from a shove.
            disturb=False,
            slip_mode=slip_mode,
            balance_assist_scale=1.0,
        )
        env = Monitor(
            env,
            info_keywords=(
                "success",
                "recovered",
                "slip_triggered",
                "slip_depth_m",
                "ascent",
                "upright_score",
                "balance_assist_scale",
            ),
        )
        env.reset(seed=seed + rank)
        return env

    return _init


class GatedAssistCallback(BaseCallback):
    """Withdraw the stabilizer only while the robot is still walking.

    A fixed schedule does not work here, and the failure is instructive. The
    first attempt annealed 1.0 -> 0.25 linearly over 740k steps regardless of
    how the policy was doing. It marched straight through the x0.5 regime --
    where the shipped checkpoint manages 0.226 m of a 1.5 m ascent -- and never
    came back. `ep_rew_mean` fell 205 -> -22.7 and the resulting checkpoint
    scored 0.003 m of ascent on the UNDISTURBED parent env: not a policy that
    failed to adapt, a policy that forgot how to walk.

    So the curriculum is gated on measured performance instead of on a clock.
    Assist steps down only when recent episodes are still climbing, steps back
    up if they collapse, and disturbances wait until the robot is stable at a
    lower assist. The floor is a floor, not a target -- ending at x0.6 with a
    policy that walks is a real result; ending at x0.25 with one that cannot is
    not.
    """

    def __init__(
        self,
        assist_floor: float,
        ascent_target: float,
        check_every: int,
        step_down: float,
        step_up: float,
        disturb_below: float,
        min_episodes: int,
    ) -> None:
        super().__init__(verbose=0)
        self.assist_floor = assist_floor
        self.ascent_target = ascent_target
        self.check_every = check_every
        self.step_down = step_down
        self.step_up = step_up
        self.disturb_below = disturb_below
        self.min_episodes = min_episodes
        self.scale = 1.0
        self._next_check = 0
        self._disturb_on = False

    def _recent_ascent(self) -> float | None:
        buf = self.model.ep_info_buffer
        if buf is None or len(buf) < self.min_episodes:
            return None
        vals = [e["ascent"] for e in buf if "ascent" in e]
        return float(np.mean(vals)) if vals else None

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_check:
            return True
        self._next_check = self.num_timesteps + self.check_every
        ascent = self._recent_ascent()
        if ascent is None:
            return True

        if ascent >= self.ascent_target:
            new = max(self.scale - self.step_down, self.assist_floor)
            verdict = "walking"
        elif ascent < 0.5 * self.ascent_target:
            # Back off. Holding a scale the policy cannot cope with is exactly
            # how the first run destroyed the gait.
            new = min(self.scale + self.step_up, 1.0)
            verdict = "struggling"
        else:
            new = self.scale
            verdict = "holding"

        if abs(new - self.scale) > 1e-9:
            self.scale = round(new, 3)
            self.training_env.env_method("set_balance_assist_scale", self.scale)
        print(f"[{self.num_timesteps:>9,}] ascent {ascent:5.2f} m ({verdict}) "
              f"-> assist {self.scale:.2f}", flush=True)

        # Disturbances only once the robot is stable well below full assist.
        # Learning to stand unaided and to recover from a shove at the same
        # time is the reliable way to learn neither.
        if not self._disturb_on and self.scale <= self.disturb_below \
                and ascent >= self.ascent_target:
            self.training_env.set_attr("disturb", True)
            print(f"[{self.num_timesteps:>9,}] slip disturbance ENABLED "
                  f"at assist {self.scale:.2f}", flush=True)
            self._disturb_on = True
        return True


def train(args) -> pathlib.Path:
    output = pathlib.Path(args.save_dir)
    output.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)

    constructors = [make_env(i, args.seed, args.slip_mode) for i in range(args.num_envs)]
    env = SubprocVecEnv(constructors) if args.num_envs > 1 else DummyVecEnv(constructors)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PPO with {args.num_envs} MuJoCo workers; device={device}", flush=True)

    if args.resume:
        print(f"fine-tuning from {args.resume}", flush=True)
        model = PPO.load(args.resume, env=env, device=device, tensorboard_log="logs/tb")
        model.learning_rate = args.learning_rate
        model.lr_schedule = get_schedule_fn(args.learning_rate)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs={
                "net_arch": {"pi": [256, 256], "vf": [256, 256]},
                "activation_fn": torch.nn.Tanh,
            },
            learning_rate=args.learning_rate,
            n_steps=512,
            batch_size=512,
            n_epochs=8,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.002,
            verbose=1,
            tensorboard_log="logs/tb",
            seed=args.seed,
            device=device,
        )

    callbacks = [
        GatedAssistCallback(
            args.assist_final,
            args.ascent_target,
            args.check_every,
            args.step_down,
            args.step_up,
            args.disturb_below,
            args.min_episodes,
        ),
        CheckpointCallback(
            save_freq=max(args.checkpoint_interval // args.num_envs, 1),
            save_path=str(output / "checkpoints"),
            name_prefix="g1_slip_recovery",
        ),
    ]
    # progress_bar=True needs tqdm+rich, which requirements.txt does not
    # declare -- train.py:272 and train_fixed_line.py:132 both hardcode it, so
    # a clean `pip install -r requirements.txt` followed by training dies on an
    # ImportError before the first step. Degrade instead of crashing.
    learn_kwargs = dict(
        total_timesteps=args.timesteps,
        callback=callbacks,
        reset_num_timesteps=args.resume is None,
    )
    try:
        import tqdm  # noqa: F401
        import rich  # noqa: F401
        learn_kwargs["progress_bar"] = True
    except ImportError:
        print("tqdm/rich not installed; training without a progress bar", flush=True)
    model.learn(**learn_kwargs)

    final = output / "g1_slip_recovery_final.zip"
    model.save(final.with_suffix(""))
    (output / "training_metadata.json").write_text(json.dumps({
        "algorithm": "PPO",
        "task": "fixed-line ascent with induced slip and recovery",
        "resumed_from": args.resume,
        "new_timesteps": args.timesteps,
        "resumed_from_step": 259_896 if args.resume else 0,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "device": device,
        "slip_mode": args.slip_mode,
        "assist_final": args.assist_final,
        "assist_floor": args.assist_final,
        "curriculum": "performance-gated",
        "ascent_target_m": args.ascent_target,
        "disturb_below_assist": args.disturb_below,
        "learning_rate": args.learning_rate,
        "action_dim": 2,
        "note": (
            "Action space deliberately unchanged from fixed_line_slope_env so "
            "the shipped checkpoint remains loadable. Report every recovery "
            "number together with the balance assist scale it was measured at."
        ),
    }, indent=2) + "\n")
    env.close()
    print(f"saved {final}", flush=True)
    return final


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timesteps", type=int, default=400_000,
                   help="NEW steps to train, not a total. On resume SB3 does "
                        "`total_timesteps += self.num_timesteps` "
                        "(BaseAlgorithm._setup_learn), so a run resumed from "
                        "the 259,896-step checkpoint with --timesteps 1200000 "
                        "ends at 1,459,896.")
    p.add_argument("--num-envs", type=int, default=min(12, os.cpu_count() or 1))
    p.add_argument("--save-dir", default="models/ppo_slip_recovery")
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--resume", default="models/ppo_fixed_line_slope/g1_fixed_line_final.zip")
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--slip-mode", choices=("friction", "impulse"), default="friction")
    p.add_argument("--assist-final", type=float, default=0.25,
                   help="FLOOR for the assist, not a target. The gate decides how far it actually gets; stopping high with a policy that walks beats reaching the floor with one that cannot.")
    p.add_argument("--ascent-target", type=float, default=1.10,
                   help="Mean recent ascent (m) required before assist steps down.")
    p.add_argument("--check-every", type=int, default=20_000)
    p.add_argument("--step-down", type=float, default=0.05)
    p.add_argument("--step-up", type=float, default=0.10,
                   help="Backs off faster than it advances, on purpose.")
    p.add_argument("--disturb-below", type=float, default=0.70,
                   help="Introduce slips only once assist is at or below this.")
    p.add_argument("--min-episodes", type=int, default=12)
    p.add_argument("--checkpoint-interval", type=int, default=20_000)
    train(p.parse_args())
