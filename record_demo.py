"""Record a rollout driven exclusively by a saved PPO policy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

from himalaya_env import G1SelfArrestEnv


def resolve_model_path(requested: str | None) -> Path:
    candidates = [
        Path(requested) if requested else None,
        Path("models/ppo_self_arrest/g1_self_arrest_final.zip"),
        Path("models/ppo_self_arrest_refined/g1_self_arrest_final.zip"),
        Path("models/ppo_self_arrest_refined/best/best_model.zip"),
        Path("models/ppo_self_arrest/best/best_model.zip"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No trained PPO checkpoint found. Run `python train.py` first or pass --model."
    )


def record_policy_demo(
    model_path: str | None = None,
    output_video: str = "videos/g1_self_arrest_learned.mp4",
    seed: int = 2026,
    fps: int = 50,
) -> dict[str, float | bool | str]:
    checkpoint = resolve_model_path(model_path)
    output = Path(output_video)
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_dir = Path("videos/snapshots")
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    model = PPO.load(checkpoint, device="cpu")
    env = G1SelfArrestEnv(render_mode="rgb_array", randomize_reset=False)
    # Record the strongest causal demonstration: retain normal slope support,
    # but remove tangential friction from every non-axe slope contact.  A stop
    # in this rollout must therefore come from the physical pick/snow path.
    env.set_body_slope_friction_enabled(False)
    observation, _ = env.reset(seed=seed, options={"randomize": False, "speed": 4.5})
    renderer = mujoco.Renderer(env.model, height=1080, width=1920)
    hand_renderer = mujoco.Renderer(env.model, height=480, width=720)
    camera = mujoco.MjvCamera()
    # A stationary side view makes downhill travel and deceleration visible:
    # the robot enters uphill/right and stops downhill/left instead of being
    # pinned to the centre of a tracking camera.
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Centre the *traversal*, not the robot: the start sits uphill/right and
    # the arrest finishes downhill/left while both endpoints remain in frame.
    camera.lookat[:] = [-0.3, 0.0, -0.2]
    camera.distance = 10.5
    camera.azimuth = 90
    camera.elevation = -15
    hand_camera = mujoco.MjvCamera()
    hand_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    hand_camera.distance = 0.62
    # The reverse chest-side view makes the dark pick silhouette and both
    # physical finger wraps visible instead of hiding the pick behind a palm.
    hand_camera.azimuth = 235
    hand_camera.elevation = -28
    left_wrist_body_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"
    )
    right_wrist_body_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"
    )
    font_path = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
    status_font = ImageFont.truetype(font_path, 30)
    detail_font = ImageFont.truetype(font_path, 24)

    speeds: list[float] = []
    forces: list[float] = []
    snow_drag_forces: list[float] = []
    ventral_margins: list[float] = []
    axe_head_heights: list[float] = []
    rigid_pick_contact_steps = 0
    actions: list[list[float]] = []
    pick_strokes: list[float] = []
    blade_angles: list[float] = []
    final_info: dict = {}
    start_frame: np.ndarray | None = None
    middle_frame: np.ndarray | None = None
    final_frame: np.ndarray | None = None
    writer = imageio.get_writer(
        output,
        fps=fps,
        codec="libx264",
        quality=9,
        macro_block_size=1,
    )
    # One policy step is 0.01 s; render every other step for a 50 FPS video.
    try:
        for step in range(env.max_episode_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, final_info = env.step(action)
            speeds.append(float(final_info["v_slope"]))
            forces.append(float(final_info["f_pick"]))
            snow_drag_forces.append(float(final_info["snow_drag_force"]))
            ventral_margins.append(float(final_info["ventral_placement_margin"]))
            axe_head_heights.append(float(final_info["axe_head_torso_z"]))
            rigid_pick_contact_steps += int(final_info["rigid_pick_contact"] > 0.5)
            actions.append(np.asarray(action).tolist())
            pick_strokes.append(float(final_info["pick_stroke_displacement"]))
            blade_angles.append(
                float(final_info["pick_blade_into_slope_angle_deg"])
            )
            if step % 2 == 0:
                contact_now = final_info["rigid_pick_contact"] > 0.5
                # This diagnostic sphere is centred on the physical pick site,
                # not on an independently animated point. Green means a rigid
                # pick/slope contact exists at the rendered instant.
                # Keep the contact marker small enough that it cannot hide the
                # actual pick geometry in the close-up.
                env.model.site_size[env.axe_pick_site_id, 0] = 0.012
                env.model.site_rgba[env.axe_pick_site_id] = (
                    [0.05, 1.0, 0.10, 1.0]
                    if contact_now
                    else [1.0, 0.12, 0.05, 0.80]
                )
                renderer.update_scene(env.data, camera=camera)
                frame = renderer.render().copy()
                hand_camera.lookat[:] = 0.5 * (
                    env.data.xpos[left_wrist_body_id]
                    + env.data.xpos[right_wrist_body_id]
                )
                hand_renderer.update_scene(env.data, camera=hand_camera)
                hand_frame = hand_renderer.render()
                # Large live close-up from the chest-facing side.  It exposes
                # the orange shaft, pick/head, palms, and articulated digits.
                inset_y, inset_x = 24, 24
                inset_h, inset_w = hand_frame.shape[:2]
                frame[
                    inset_y - 6 : inset_y + inset_h + 6,
                    inset_x - 6 : inset_x + inset_w + 6,
                ] = 25
                frame[
                    inset_y : inset_y + inset_h,
                    inset_x : inset_x + inset_w,
                ] = hand_frame
                annotated = Image.fromarray(frame)
                draw = ImageDraw.Draw(annotated, "RGBA")
                draw.rounded_rectangle(
                    (24, 934, 1896, 1056),
                    radius=14,
                    fill=(10, 18, 28, 218),
                    outline=(125, 180, 220, 235),
                    width=2,
                )
                draw.text(
                    (48, 948),
                    "LEARNED PPO POLICY  |  35° SLOPE  |  NON-AXE BODY FRICTION: DISABLED",
                    font=status_font,
                    fill=(245, 248, 252, 255),
                )
                contact_label = "YES" if contact_now else "NO"
                contact_color = (
                    (80, 255, 110, 255) if contact_now else (255, 125, 90, 255)
                )
                draw.text(
                    (48, 1000),
                    f"speed: {final_info['v_slope']:.2f} m/s     rigid pick contact: {contact_label}",
                    font=detail_font,
                    fill=contact_color,
                )
                draw.text(
                    (730, 1000),
                    f"plant travel: {100.0 * max(final_info['pick_stroke_displacement'], final_info['stroke_at_first_contact']):.1f} cm     "
                    f"blade: {final_info['pick_blade_into_slope_angle_deg']:.1f}° into slope",
                    font=detail_font,
                    fill=(225, 232, 240, 255),
                )
                slow_motion_plant = step <= 60
                phase_label = (
                    "0.25x SLOW MOTION — LEARNED PICK PLANT"
                    if slow_motion_plant
                    else "REAL TIME — SUSTAINED AXE ARREST"
                )
                draw.rounded_rectangle(
                    (1110, 28, 1888, 82),
                    radius=10,
                    fill=(10, 18, 28, 218),
                    outline=(245, 248, 252, 235),
                    width=2,
                )
                draw.text(
                    (1132, 39),
                    phase_label,
                    font=detail_font,
                    fill=(245, 248, 252, 255),
                )
                draw.rounded_rectangle(
                    (24, 24, 744, 510),
                    radius=8,
                    outline=(245, 248, 252, 255),
                    width=3,
                )
                draw.text(
                    (42, 522),
                    "LIVE CLOSE-UP: FINGERS, SHAFT, HEAD, AND PICK",
                    font=detail_font,
                    fill=(245, 248, 252, 255),
                    stroke_width=2,
                    stroke_fill=(10, 18, 28, 255),
                )
                frame = np.asarray(annotated)
                # The learned plant takes only ~0.29 s in real time. Repeat
                # its rendered frames four times so the pick motion and angle
                # change are visually auditable; the remainder is real time.
                for _ in range(4 if slow_motion_plant else 1):
                    writer.append_data(frame)
                start_frame = frame.copy() if start_frame is None else start_frame
                if step >= 90 and middle_frame is None:
                    middle_frame = frame.copy()
                final_frame = frame.copy()
            if terminated or truncated:
                break

        if final_frame is None:
            raise RuntimeError("Policy rollout produced no video frames")
        # Hold on the physical final state so the completed stop is visible.
        for _ in range(fps):
            writer.append_data(final_frame)
    finally:
        writer.close()

    if middle_frame is None:
        middle_frame = final_frame.copy()
    imageio.imwrite(snapshot_dir / "learned_arrest_start.png", start_frame)
    imageio.imwrite(snapshot_dir / "learned_arrest_mid.png", middle_frame)
    imageio.imwrite(snapshot_dir / "learned_arrest_final.png", final_frame)

    action_array = np.asarray(actions)
    report: dict[str, float | bool | str] = {
        "checkpoint": str(checkpoint.resolve()),
        "video": str(output.resolve()),
        "success": bool(final_info.get("success", False)),
        "initial_speed_mps": 4.5,
        "final_speed_mps": float(final_info.get("v_slope", np.nan)),
        "stopping_distance_m": float(final_info.get("stopping_distance", np.nan)),
        "pick_contact_fraction": float(final_info.get("pick_contact_fraction", 0.0)),
        "left_grasp_contact_fraction": float(
            final_info.get("left_grasp_contact_fraction", 0.0)
        ),
        "right_grasp_contact_fraction": float(
            final_info.get("right_grasp_contact_fraction", 0.0)
        ),
        "peak_sampled_rigid_pick_force_n": float(max(forces, default=0.0)),
        "peak_substep_rigid_pick_force_n": float(
            final_info.get("peak_rigid_pick_force", 0.0)
        ),
        "rigid_pick_contact_seen": bool(
            final_info.get("rigid_pick_contact_seen", False)
        ),
        "mean_snow_drag_force_n": float(np.mean(snow_drag_forces)),
        "rigid_pick_contact_fraction": rigid_pick_contact_steps / len(actions),
        "rigid_pick_contact_substep_fraction": float(
            final_info.get("rigid_pick_contact_substep_fraction", 0.0)
        ),
        "terminal_rolling_rigid_pick_contact_fraction": float(
            final_info.get("rolling_rigid_pick_contact_fraction", 0.0)
        ),
        "terminal_rolling_mean_snow_drag_force_n": float(
            final_info.get("rolling_mean_snow_drag_force", 0.0)
        ),
        "grip_score": float(final_info.get("grip_score", 0.0)),
        "minimum_ventral_placement_margin_m": float(min(ventral_margins)),
        "axe_head_torso_z_range_m": [
            float(min(axe_head_heights)),
            float(max(axe_head_heights)),
        ],
        "policy_steps": len(actions),
        "mean_action": action_array.mean(axis=0).tolist(),
        "action_std": action_array.std(axis=0).tolist(),
        "minimum_action": action_array.min(axis=0).tolist(),
        "maximum_action": action_array.max(axis=0).tolist(),
        "first_action": action_array[0].tolist(),
        "final_action": action_array[-1].tolist(),
        "minimum_speed_mps": float(min(speeds, default=np.nan)),
        "first_rigid_contact_step": int(
            final_info.get("first_rigid_contact_step", -1)
        ),
        "stroke_at_first_contact_m": float(
            final_info.get("stroke_at_first_contact", 0.0)
        ),
        "lowering_at_first_contact_m": float(
            final_info.get("lowering_at_first_contact", 0.0)
        ),
        "blade_angle_at_first_contact_deg": float(
            final_info.get("blade_angle_at_first_contact_deg", 0.0)
        ),
        "terminal_rolling_blade_angle_deg": float(
            final_info.get("rolling_pick_blade_into_slope_angle_deg", 0.0)
        ),
        "maximum_visible_pick_stroke_m": float(max(pick_strokes, default=0.0)),
        "blade_angle_range_deg": [
            float(min(blade_angles, default=np.nan)),
            float(max(blade_angles, default=np.nan)),
        ],
        "valid_learned_plant_motion": bool(
            final_info.get("valid_learned_plant_motion", False)
        ),
        "live_hand_closeup_inset": True,
        "physical_pick_contact_marker": "green during current rigid contact; red otherwise",
        "non_axe_body_friction_enabled": False,
        "main_camera": "fixed 10.5 m wide side view; robot traverses frame",
        "video_timeline": (
            "first 0.61 policy-seconds shown at 0.25x for the learned plant; "
            "remaining arrest shown in real time; final state held for 1 s"
        ),
        "output_resolution": "1920x1080",
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    renderer.close()
    hand_renderer.close()
    env.close()
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--output", default="videos/g1_self_arrest_learned.mp4")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    record_policy_demo(args.model, args.output, args.seed)
