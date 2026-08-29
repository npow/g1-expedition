"""Record a fixed-line rollout driven exclusively by a saved PPO policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from stable_baselines3 import PPO

from evaluate_fixed_line import resolve_fixed_line_model
from fixed_line_slope_env import G1FixedLineEnv
from fixed_line_visuals import alpine_backdrop, render_with_alpine_backdrop


def record_policy_demo(
    model_path: str | None = None,
    output_video: str = "videos/g1_fixed_line_learned.mp4",
    seed: int = 2027,
    fps: int = 50,
) -> dict:
    checkpoint = resolve_fixed_line_model(model_path)
    output = Path(output_video)
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_dir = Path("videos/snapshots")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_prefix = output.stem

    policy = PPO.load(checkpoint, device="cpu")
    env = G1FixedLineEnv(render_mode="rgb_array", randomize_reset=False)
    observation, _ = env.reset(seed=seed, options={"randomize": False})
    renderer = mujoco.Renderer(env.model, height=720, width=1280)
    equipment_renderer = mujoco.Renderer(env.model, height=260, width=430)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = env.pelvis_body_id
    camera.distance = 2.85
    camera.azimuth = 120
    camera.elevation = -7
    close_camera = mujoco.MjvCamera()
    close_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    close_camera.distance = 0.95
    close_camera.azimuth = 92
    close_camera.elevation = -5
    main_backdrop = alpine_backdrop(1280, 720)
    inset_backdrop = alpine_backdrop(430, 260)

    actions: list[list[float]] = []
    heights: list[float] = []
    chest_forces: list[float] = []
    ground_forces: list[float] = []
    grounded_fractions: list[float] = []
    left_contact_fractions: list[float] = []
    right_contact_fractions: list[float] = []
    grasp_scores: list[float] = []
    left_grip_errors: list[float] = []
    right_grip_errors: list[float] = []
    hand_separations: list[float] = []
    cross_hand_collisions: list[float] = []
    wall_hand_collisions: list[float] = []
    final_info: dict = {}
    snapshots: dict[str, np.ndarray] = {}
    frame_count = 0
    with imageio.get_writer(
        output, fps=fps, codec="libx264", quality=8, macro_block_size=1
    ) as writer:
        for step in range(env.max_episode_steps):
            action, _state = policy.predict(observation, deterministic=True)
            observation, _, terminated, truncated, final_info = env.step(action)
            actions.append(np.asarray(action).tolist())
            heights.append(float(final_info["ascent"]))
            chest_forces.append(float(final_info["line_load_n"]))
            ground_forces.append(float(final_info["ground_load_n"]))
            grounded_fractions.append(float(final_info["grounded_fraction"]))
            left_contact_fractions.append(
                float(final_info["left_boot_contact_fraction"])
            )
            right_contact_fractions.append(
                float(final_info["right_boot_contact_fraction"])
            )
            grasp_scores.append(float(final_info["grasp_score"]))
            left_grip_errors.append(float(final_info["left_grip_error"]))
            right_grip_errors.append(float(final_info["right_grip_error"]))
            hand_separations.append(float(final_info["hand_separation"]))
            cross_hand_collisions.append(float(final_info["cross_hand_collision"]))
            wall_hand_collisions.append(float(final_info["wall_hand_collision"]))

            frame = render_with_alpine_backdrop(
                renderer, env.data, camera, main_backdrop
            )
            _chest_device, hand_device = env._device_points()
            close_camera.lookat[:] = hand_device
            close_frame = render_with_alpine_backdrop(
                equipment_renderer, env.data, close_camera, inset_backdrop
            )
            inset_y, inset_x = 18, 18
            inset_h, inset_w = close_frame.shape[:2]
            frame[
                inset_y - 4 : inset_y + inset_h + 4,
                inset_x - 4 : inset_x + inset_w + 4,
            ] = 30
            frame[inset_y : inset_y + inset_h, inset_x : inset_x + inset_w] = close_frame
            writer.append_data(frame)
            frame_count += 1
            if step == 0:
                snapshots["start"] = frame.copy()
            if step == env.max_episode_steps // 2:
                snapshots["mid"] = frame.copy()
            if terminated or truncated:
                break
        if frame_count:
            snapshots["final"] = frame.copy()
            snapshots.setdefault("mid", frame.copy())
            for _ in range(fps):
                writer.append_data(frame)

    if not frame_count:
        raise RuntimeError("Fixed-line policy rollout produced no video frames")
    imageio.imwrite(
        snapshot_dir / f"{snapshot_prefix}_start.png", snapshots["start"]
    )
    imageio.imwrite(
        snapshot_dir / f"{snapshot_prefix}_mid.png", snapshots["mid"]
    )
    imageio.imwrite(
        snapshot_dir / f"{snapshot_prefix}_final.png", snapshots["final"]
    )

    action_array = np.asarray(actions)
    report = {
        "checkpoint": str(checkpoint),
        "video": str(output),
        "success": bool(final_info.get("success", False)),
        "ascent_m": float(final_info.get("ascent", np.nan)),
        "along_slope_ascent_m": float(final_info.get("ascent", np.nan)),
        "high_water_ascent_m": float(
            final_info.get("high_water_ascent", np.nan)
        ),
        "completed_cycles": int(final_info.get("completed_cycles", 0)),
        "completed_alternating_steps": int(
            final_info.get("completed_cycles", 0)
        ),
        "descent_from_high_water_m": float(
            final_info.get("descent_from_high_water", np.nan)
        ),
        "line_load_fraction": float(final_info.get("line_load_fraction", 0.0)),
        "grounded_fraction": float(final_info.get("grounded_fraction", 0.0)),
        "left_boot_contact_fraction": float(
            final_info.get("left_boot_contact_fraction", 0.0)
        ),
        "right_boot_contact_fraction": float(
            final_info.get("right_boot_contact_fraction", 0.0)
        ),
        "double_support_fraction": float(
            final_info.get("double_support_fraction", 0.0)
        ),
        "maximum_airborne_streak": int(
            final_info.get("maximum_airborne_streak", 0)
        ),
        "vertical_gain_m": float(final_info.get("vertical_gain_m", 0.0)),
        "peak_line_load_n": float(max(chest_forces, default=0.0)),
        "peak_ground_load_n": float(max(ground_forces, default=0.0)),
        "minimum_grasp_score": float(min(grasp_scores, default=0.0)),
        "mean_grasp_score": float(np.mean(grasp_scores)),
        "maximum_left_grip_error_m": float(max(left_grip_errors, default=np.nan)),
        "maximum_right_grip_error_m": float(max(right_grip_errors, default=np.nan)),
        "minimum_hand_separation_m": float(min(hand_separations, default=np.nan)),
        "cross_hand_collision_steps": int(np.sum(cross_hand_collisions)),
        "wall_hand_collision_steps": int(np.sum(wall_hand_collisions)),
        "policy_steps": len(actions),
        "mean_action": action_array.mean(axis=0).tolist(),
        "action_std": action_array.std(axis=0).tolist(),
        "minimum_action": action_array.min(axis=0).tolist(),
        "maximum_action": action_array.max(axis=0).tolist(),
        "first_action": action_array[0].tolist(),
        "final_action": action_array[-1].tolist(),
        "minimum_ascent_m": float(min(heights, default=np.nan)),
        "live_equipment_closeup_inset": True,
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    renderer.close()
    equipment_renderer.close()
    env.close()
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--output", default="videos/g1_fixed_line_learned.mp4")
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    record_policy_demo(args.model, args.output, args.seed)
