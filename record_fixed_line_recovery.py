"""Record climb, physical fall, learned get-up, re-grasp, and continued climb."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO

from evaluate_fixed_line import resolve_fixed_line_model
from evaluate_mountain_recovery import resolve_recovery_model
from fixed_line_slope_env import G1FixedLineEnv
from fixed_line_visuals import alpine_backdrop, render_with_alpine_backdrop
from mountain_recovery import FixedLineRecoveryController


def _label_frame(
    frame: np.ndarray,
    phase: str,
    detail: str,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((28, 28, 620, 104), radius=12, fill=(5, 12, 18, 195))
    draw.text((48, 42), phase, fill=(255, 240, 178, 255), stroke_width=1)
    draw.text((48, 72), detail, fill=(235, 242, 248, 255))
    return np.asarray(image)


def record(
    recovery_model: str | None = None,
    climbing_model: str | None = None,
    output_video: str = "videos/g1_fixed_line_fall_recovery.mp4",
    seed: int = 2027,
    fps: int = 25,
) -> dict:
    recovery_checkpoint = resolve_recovery_model(recovery_model)
    climbing_checkpoint = resolve_fixed_line_model(climbing_model)
    recovery_policy = PPO.load(recovery_checkpoint, device="cpu")
    climbing_policy = PPO.load(climbing_checkpoint, device="cpu")
    env = G1FixedLineEnv(render_mode="rgb_array", randomize_reset=False)
    observation, _ = env.reset(seed=seed, options={"randomize": False})
    controller = FixedLineRecoveryController(env)

    output = Path(output_video)
    output.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(env.model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    # A fixed above-slope camera makes world translation visible and cannot
    # follow the pelvis below the terrain during a fall.  The target is a
    # mountain-level point near the middle of the full route, not the robot.
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.trackbodyid = -1
    camera.lookat[:] = (
        env.data.xpos[env.pelvis_body_id] + 0.72 * env.uphill
    )
    camera.distance = 4.35
    camera.azimuth = 120
    camera.elevation = -9
    backdrop = alpine_backdrop(1280, 720)
    snapshots: dict[str, np.ndarray] = {}
    frame_count = 0
    render_call_count = 0
    last_frame: np.ndarray | None = None
    simulation_hz = int(round(1.0 / (env.model.opt.timestep * env.frame_skip)))
    if fps <= 0 or fps > simulation_hz or simulation_hz % fps:
        raise ValueError(
            f"fps must evenly divide the {simulation_hz} Hz policy rate"
        )
    render_stride = simulation_hz // fps

    def append(
        writer,
        phase: str,
        detail: str,
        *,
        force: bool = False,
    ) -> np.ndarray:
        nonlocal frame_count, render_call_count, last_frame
        should_render = (
            force
            or last_frame is None
            or render_call_count % render_stride == 0
        )
        render_call_count += 1
        if not should_render:
            return last_frame
        frame = render_with_alpine_backdrop(
            renderer, env.model, env.data, camera, backdrop
        )
        frame = _label_frame(frame, phase, detail)
        writer.append_data(frame)
        frame_count += 1
        last_frame = frame
        return frame

    pre_fall_progress = env._progress()
    with imageio.get_writer(
        output, fps=fps, codec="libx264", quality=8, macro_block_size=1
    ) as writer:
        snapshots["start"] = append(
            writer,
            "LEARNED FIXED-LINE ASCENT",
            "fixed camera: world-frame uphill motion",
            force=True,
        )
        pre_fall_info: dict = {}
        for _ in range(112):
            action, _state = climbing_policy.predict(
                observation, deterministic=True
            )
            observation, _reward, terminated, truncated, pre_fall_info = env.step(
                action
            )
            frame = append(
                writer,
                "LEARNED FIXED-LINE ASCENT",
                f"uphill progress {pre_fall_info['ascent']:.2f} m",
            )
            if terminated or truncated:
                raise RuntimeError("Climbing policy ended before the fall trigger")
        for _ in range(40):
            if env._swing_side is None:
                break
            action, _state = climbing_policy.predict(
                observation, deterministic=True
            )
            observation, _reward, terminated, truncated, pre_fall_info = env.step(
                action
            )
            frame = append(
                writer,
                "LEARNED FIXED-LINE ASCENT",
                f"uphill progress {pre_fall_info['ascent']:.2f} m",
            )
        frame = append(
            writer,
            "LEARNED FIXED-LINE ASCENT",
            f"uphill progress {pre_fall_info['ascent']:.2f} m",
            force=True,
        )
        snapshots["pre_fall"] = frame.copy()
        pre_fall_progress = env._progress()

        controller.start_fall()
        for fall_frame in range(controller.fall_frames):
            fall_info = controller.step_fall(fall_frame, lateral_bias_n=4.0)
            frame = append(
                writer,
                "PHYSICAL CRAMPON SLIP + FALL",
                f"cam load {fall_info['line_load_n']:.0f} N | no root teleport",
            )
            if fall_frame == controller.push_frames + 10:
                frame = append(
                    writer,
                    "PHYSICAL CRAMPON SLIP + FALL",
                    (
                        f"cam load {fall_info['line_load_n']:.0f} N | "
                        "camera stays above slope"
                    ),
                    force=True,
                )
                snapshots["fall"] = frame.copy()
        grounded_progress = env._progress()
        peak_arrest_line_load_n = controller.peak_line_load_n
        frame = append(
            writer,
            "PHYSICAL FALL ARRESTED",
            "fixed camera never follows the pelvis under terrain",
            force=True,
        )
        snapshots["grounded"] = frame.copy()

        for ready_frame in range(controller.floor_ready_frames):
            ready_info = controller.step_floor_ready(ready_frame)
            frame = append(
                writer,
                "GROUNDED PREPARATION",
                "joint torques align the recovery pose on the 28-degree slope",
            )

        recovery_actions: list[np.ndarray] = []
        for _ in range(controller.recovery_frames):
            action, _state = recovery_policy.predict(
                controller.policy_observation(), deterministic=True
            )
            recovery_actions.append(np.asarray(action, dtype=np.float64))
            recovery_info = controller.step_recovery(action)
            frame = append(
                writer,
                "LEARNED SLOPE GET-UP",
                (
                    f"PPO brake actions + 1.1 m lanyard | upright "
                    f"{recovery_info['torso_upright']:.2f}"
                ),
            )
            if controller.recovered:
                break
        if not controller.recovered:
            raise RuntimeError("Recovery policy failed the two-foot transfer gate")
        frame = append(
            writer,
            "LEARNED SLOPE GET-UP",
            (
                f"PPO + lanyard | upright {recovery_info['torso_upright']:.2f}"
            ),
            force=True,
        )
        snapshots["recovered"] = frame.copy()

        controller.start_regrasp(100)
        while controller.phase == "regrasp":
            regrasp_info = controller.step_regrasp()
            frame = append(
                writer,
                "PHYSICAL RE-GRASP",
                "cam closes; right hand returns outside the braided rope",
            )
        regrasp_progress = env._progress()
        regrasp_metrics = env._metrics()
        regrasp_linear_speed = float(np.linalg.norm(env.data.qvel[:3]))
        regrasp_angular_speed = float(np.linalg.norm(env.data.qvel[3:6]))

        final_info: dict = {}
        climb_segments = 2
        completed_climb_segments = 0
        total_post_recovery_ascent = 0.0
        total_post_recovery_cycles = 0
        total_post_climb_rope_collisions = 0
        total_post_climb_hand_penetrations = 0
        for segment in range(climb_segments):
            observation, _ = env.rebase_climb_progress()
            final_info = {}
            for _ in range(env.max_episode_steps):
                action, _state = climbing_policy.predict(
                    observation, deterministic=True
                )
                (
                    observation,
                    _reward,
                    terminated,
                    truncated,
                    final_info,
                ) = env.step(action)
                displayed_ascent = (
                    total_post_recovery_ascent
                    + float(final_info.get("ascent", 0.0))
                )
                frame = append(
                    writer,
                    f"ASCENT RESUMED — SEGMENT {segment + 1}/{climb_segments}",
                    f"world-frame uphill progress {displayed_ascent:.2f} m",
                )
                if terminated or truncated:
                    break
            total_post_recovery_ascent += float(
                final_info.get("ascent", 0.0)
            )
            total_post_recovery_cycles += int(
                final_info.get("completed_cycles", 0)
            )
            total_post_climb_rope_collisions += int(
                final_info.get("rope_core_collision_steps", 0)
            )
            total_post_climb_hand_penetrations += int(
                final_info.get("hand_rope_penetration_steps", 0)
            )
            if not final_info.get("success", False):
                raise RuntimeError(
                    f"Resumed ascent segment {segment + 1} did not finish"
                )
            completed_climb_segments += 1
        frame = append(
            writer,
            "ASCENT RESUMED — COMPLETE",
            f"world-frame uphill progress {total_post_recovery_ascent:.2f} m",
            force=True,
        )
        snapshots["final"] = frame.copy()
        for _ in range(fps):
            writer.append_data(frame)

    snapshot_dir = output.parent / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in snapshots.items():
        imageio.imwrite(snapshot_dir / f"{output.stem}_{name}.png", frame)
    actions = np.asarray(recovery_actions)
    report = {
        "video": str(output),
        "climbing_checkpoint": str(climbing_checkpoint),
        "recovery_checkpoint": str(recovery_checkpoint),
        "success": True,
        "pre_fall_ascent_m": float(pre_fall_info["ascent"]),
        "fall_arrest_progress_loss_m": float(
            max(pre_fall_progress - grounded_progress, 0.0)
        ),
        "fall_grounded_nonfoot_contact": bool(
            fall_info["nonfoot_ground_contact"]
        ),
        "recovery_policy_steps": len(recovery_actions),
        "recovery_progress_change_m": float(
            regrasp_progress - grounded_progress
        ),
        "net_post_fall_progress_m": float(
            env._progress() - grounded_progress
        ),
        "mean_recovery_action": actions.mean(axis=0).tolist(),
        "regrasp_pelvis_normal_height_m": float(
            regrasp_metrics["pelvis_normal_height"]
        ),
        "regrasp_torso_upright": float(regrasp_metrics["upright_score"]),
        "regrasp_linear_speed_mps": regrasp_linear_speed,
        "regrasp_angular_speed_radps": regrasp_angular_speed,
        "post_recovery_climb_segments": completed_climb_segments,
        "post_recovery_ascent_m": total_post_recovery_ascent,
        "post_recovery_cycles": total_post_recovery_cycles,
        "peak_fall_arrest_line_load_n": peak_arrest_line_load_n,
        "peak_total_safety_line_load_n": controller.peak_line_load_n,
        "grounded_recovery_lanyard_slack_m": 1.1,
        "peak_recovery_side_guide_load_n": (
            controller.peak_lateral_guide_load_n
        ),
        "peak_motor_torque_ratio": controller.peak_motor_torque_ratio,
        "maximum_contact_penetration_m": (
            controller.maximum_contact_penetration_m
        ),
        "root_teleports_after_fall_start": 0,
        "rope_core_collision_frames": controller.rope_core_collision_frames,
        "hand_rope_penetration_frames": (
            controller.hand_rope_penetration_frames
        ),
        "post_climb_rope_core_collision_steps": int(
            total_post_climb_rope_collisions
        ),
        "post_climb_hand_rope_penetration_steps": int(
            total_post_climb_hand_penetrations
        ),
        "frame_count": frame_count,
        "simulation_policy_hz": simulation_hz,
        "video_fps": fps,
        "camera": {
            "mode": "fixed_above_slope",
            "lookat": camera.lookat.tolist(),
            "distance_m": camera.distance,
            "azimuth_degrees": camera.azimuth,
            "elevation_degrees": camera.elevation,
        },
        "physics_claim": (
            "No floating-base animation: finite disturbance, gravity, MuJoCo "
            "contacts, torque-capped motors, deformable rope reactions, learned "
            "get-up transfer, then the saved climbing PPO."
        ),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    renderer.close()
    controller.close()
    env.close()
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-model")
    parser.add_argument("--climbing-model")
    parser.add_argument(
        "--output", default="videos/g1_fixed_line_fall_recovery.mp4"
    )
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()
    record(
        args.recovery_model,
        args.climbing_model,
        args.output,
        args.seed,
        args.fps,
    )
