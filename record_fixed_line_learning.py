"""Render synchronized fixed-line rollouts across training checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from fixed_line_slope_env import G1FixedLineEnv
from fixed_line_visuals import alpine_backdrop, render_with_alpine_backdrop


@dataclass
class Stage:
    label: str
    interactions: int
    env: G1FixedLineEnv
    policy: PPO
    renderer: mujoco.Renderer
    camera: mujoco.MjvCamera
    observation: np.ndarray
    info: dict
    checkpoint: str | None = None
    done: bool = False
    last_frame: np.ndarray | None = None
    actions: list[np.ndarray] = field(default_factory=list)
    grasp_scores: list[float] = field(default_factory=list)
    left_grip_errors: list[float] = field(default_factory=list)
    right_grip_errors: list[float] = field(default_factory=list)
    hand_separations: list[float] = field(default_factory=list)


def initialized_policy() -> PPO:
    """Construct the exact network initialization used before PPO updates."""
    vec_env = DummyVecEnv(
        [lambda: G1FixedLineEnv(randomize_reset=False)]
    )
    model = PPO(
        "MlpPolicy",
        vec_env,
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
        seed=31,
        device="cpu",
        verbose=0,
    )
    # Prediction does not require the construction environment to remain open.
    vec_env.close()
    return model


def checkpoint_interactions(path: Path) -> int:
    match = re.search(r"_(\d+)_steps", path.name)
    if match:
        return int(match.group(1))
    return int(PPO.load(path, device="cpu").num_timesteps)


def parse_stages(values: list[str]) -> list[tuple[str, Path, int]]:
    parsed: list[tuple[str, Path, int]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Stage must be LABEL=MODEL_PATH, received: {value}")
        label, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        parsed.append((label, path, checkpoint_interactions(path)))
    if len(parsed) != 3:
        raise ValueError("Pass exactly three trained --stage entries")
    return parsed


@lru_cache(maxsize=None)
def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / filename
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def annotate(panel: np.ndarray, stage: Stage) -> np.ndarray:
    image = Image.fromarray(panel)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((12, 12, 448, 82), radius=10, fill=(8, 14, 22, 218))
    draw.text((26, 21), stage.label, font=font(20, bold=True), fill="white")
    ascent = float(stage.info.get("ascent", 0.0))
    cycles = int(stage.info.get("completed_cycles", 0))
    grasp = float(stage.info.get("grasp_score", 0.0))
    draw.text(
        (26, 51),
        f"slope {ascent:0.2f} m   steps {cycles}   grip {grasp:0.2f}",
        font=font(15),
        fill=(224, 233, 241, 255),
    )
    bar_left, bar_top, bar_right, bar_bottom = 598, 34, 620, 326
    draw.rounded_rectangle(
        (bar_left, bar_top, bar_right, bar_bottom),
        radius=8,
        fill=(14, 22, 32, 210),
        outline=(238, 245, 250, 210),
        width=2,
    )
    fraction = float(np.clip(ascent / 1.5, 0.0, 1.0))
    fill_top = bar_bottom - int((bar_bottom - bar_top) * fraction)
    color = (71, 207, 120, 245) if stage.info.get("success") else (58, 164, 229, 245)
    if fill_top < bar_bottom - 3:
        draw.rounded_rectangle(
            (bar_left + 3, fill_top, bar_right - 3, bar_bottom - 3),
            radius=5,
            fill=color,
        )
    draw.text((548, 329), "1.5 m", font=font(14, bold=True), fill="white")
    if stage.info.get("success"):
        draw.rounded_rectangle((472, 14, 622, 48), radius=9, fill=(23, 128, 69, 235))
        draw.text((489, 21), "POLICY PASS", font=font(15, bold=True), fill="white")
    return np.asarray(image)


def make_stage(
    label: str,
    interactions: int,
    policy: PPO,
    checkpoint: str | None,
    seed: int,
) -> Stage:
    env = G1FixedLineEnv(randomize_reset=False)
    observation, info = env.reset(seed=seed, options={"randomize": False})
    renderer = mujoco.Renderer(env.model, height=360, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = env.pelvis_body_id
    camera.distance = 2.85
    camera.azimuth = 120
    camera.elevation = -7
    return Stage(
        label=label,
        interactions=interactions,
        env=env,
        policy=policy,
        renderer=renderer,
        camera=camera,
        observation=observation,
        info=dict(info),
        checkpoint=checkpoint,
    )


def record_learning_progress(
    trained_stages: list[tuple[str, Path, int]],
    output_video: str = "videos/g1_fixed_line_learning_progress.mp4",
    seed: int = 2027,
    fps: int = 50,
) -> dict:
    output = Path(output_video)
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_dir = Path("videos/snapshots")
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    stages = [
        make_stage(
            "Initialized policy · 0 interactions",
            0,
            initialized_policy(),
            None,
            seed,
        )
    ]
    for label, path, interactions in trained_stages:
        stages.append(
            make_stage(
                f"{label} · {interactions:,} interactions",
                interactions,
                PPO.load(path, device="cpu"),
                str(path),
                seed,
            )
        )

    saved_snapshots: dict[int, np.ndarray] = {}
    max_steps = max(stage.env.max_episode_steps for stage in stages)
    snapshot_steps = (0, max_steps // 2, max_steps - 1)
    with imageio.get_writer(
        output, fps=fps, codec="libx264", quality=8, macro_block_size=1
    ) as writer:
        for step in range(max_steps):
            panels: list[np.ndarray] = []
            for stage in stages:
                if not stage.done:
                    action, _state = stage.policy.predict(
                        stage.observation, deterministic=True
                    )
                    stage.observation, _reward, terminated, truncated, info = (
                        stage.env.step(action)
                    )
                    stage.info = dict(info)
                    stage.actions.append(np.asarray(action, dtype=np.float32))
                    stage.grasp_scores.append(float(info["grasp_score"]))
                    stage.left_grip_errors.append(float(info["left_grip_error"]))
                    stage.right_grip_errors.append(float(info["right_grip_error"]))
                    stage.hand_separations.append(float(info["hand_separation"]))
                    stage.done = bool(terminated or truncated)
                    stage.last_frame = render_with_alpine_backdrop(
                        stage.renderer,
                        stage.env.model,
                        stage.env.data,
                        stage.camera,
                        alpine_backdrop(640, 360),
                    )
                if stage.last_frame is None:
                    raise RuntimeError("Learning stage produced no render frame")
                panels.append(annotate(stage.last_frame, stage))
            montage = np.vstack(
                [np.hstack(panels[:2]), np.hstack(panels[2:])]
            )
            writer.append_data(montage)
            if step in snapshot_steps:
                saved_snapshots[step] = montage.copy()
        for _ in range(fps):
            writer.append_data(montage)

    for step, name in zip(snapshot_steps, ("start", "mid", "final")):
        imageio.imwrite(
            snapshot_dir / f"fixed_line_learning_{name}.png", saved_snapshots[step]
        )

    stage_reports = []
    for stage in stages:
        action_array = np.asarray(stage.actions)
        stage_reports.append(
            {
                "label": stage.label,
                "interactions": stage.interactions,
                "checkpoint": stage.checkpoint,
                "success": bool(stage.info.get("success", False)),
                "ascent_m": float(stage.info.get("ascent", 0.0)),
                "completed_cycles": int(stage.info.get("completed_cycles", 0)),
                "policy_steps": len(stage.actions),
                "minimum_grasp_score": float(min(stage.grasp_scores)),
                "maximum_left_grip_error_m": float(max(stage.left_grip_errors)),
                "maximum_right_grip_error_m": float(max(stage.right_grip_errors)),
                "minimum_hand_separation_m": float(min(stage.hand_separations)),
                "mean_action": action_array.mean(axis=0).tolist(),
                "action_std": action_array.std(axis=0).tolist(),
            }
        )
        stage.renderer.close()
        stage.env.close()

    report = {
        "video": str(output),
        "same_deterministic_reset_seed": seed,
        "synchronized_rollouts": True,
        "stages": stage_reports,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        metavar="LABEL=MODEL_PATH",
        help="Exactly three trained checkpoints, in chronological order.",
    )
    parser.add_argument(
        "--output", default="videos/g1_fixed_line_learning_progress.mp4"
    )
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    record_learning_progress(parse_stages(args.stage), args.output, args.seed)
