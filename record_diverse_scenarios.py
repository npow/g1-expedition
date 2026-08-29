"""Record diverse oblique-fall rollouts driven by one saved PPO policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

from diverse_scenarios import SCENARIOS, FallScenario
from himalaya_env import G1SelfArrestEnv
from record_demo import resolve_model_path


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf", size
    )


def _bordered_inset(
    frame: np.ndarray, inset: np.ndarray, *, x: int, y: int
) -> None:
    height, width = inset.shape[:2]
    frame[y - 5 : y + height + 5, x - 5 : x + width + 5] = 22
    frame[y : y + height, x : x + width] = inset


def _annotate(
    frame: np.ndarray,
    scenario: FallScenario,
    info: dict[str, Any],
    *,
    slow_motion: bool,
    scenario_index: int,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = _font(30)
    detail_font = _font(23)
    small_font = _font(19)
    draw.rounded_rectangle(
        (655, 22, 1360, 134),
        radius=12,
        fill=(9, 17, 27, 226),
        outline=(230, 240, 250, 235),
        width=2,
    )
    draw.text(
        (680, 35),
        f"SCENARIO {scenario_index + 1}/{len(SCENARIOS)}  |  {scenario.label}",
        font=title_font,
        fill=(250, 252, 255, 255),
    )
    draw.text(
        (680, 82),
        f"heading {scenario.heading_degrees:+.0f}°   "
        f"cross-slope {scenario.lateral_speed_mps:+.2f} m/s   "
        f"roll {scenario.roll_degrees:+.0f}°",
        font=detail_font,
        fill=(190, 225, 250, 255),
    )
    phase = "0.5x LEARNED PLANT" if slow_motion else "REAL-TIME ARREST"
    draw.rounded_rectangle(
        (1470, 350, 1890, 402),
        radius=9,
        fill=(9, 17, 27, 220),
        outline=(230, 240, 250, 230),
        width=2,
    )
    draw.text((1490, 361), phase, font=detail_font, fill=(250, 252, 255, 255))
    draw.text(
        (38, 455),
        "LIVE HAND/PICK VIEW",
        font=small_font,
        fill=(250, 252, 255, 255),
        stroke_width=2,
        stroke_fill=(9, 17, 27, 255),
    )
    draw.text(
        (1470, 315),
        "LIVE SLOPE-NORMAL VIEW",
        font=small_font,
        fill=(250, 252, 255, 255),
        stroke_width=2,
        stroke_fill=(9, 17, 27, 255),
    )
    draw.rounded_rectangle(
        (24, 912, 1896, 1056),
        radius=14,
        fill=(9, 17, 27, 224),
        outline=(115, 180, 225, 240),
        width=2,
    )
    contact = info["rigid_pick_contact"] > 0.5
    status_colour = (75, 255, 110, 255) if contact else (255, 120, 85, 255)
    draw.text(
        (48, 930),
        "ONE LEARNED PPO  |  35° SLOPE  |  NON-AXE BODY FRICTION DISABLED",
        font=title_font,
        fill=(245, 248, 252, 255),
    )
    plant_travel = max(
        float(info["pick_stroke_displacement"]),
        float(info["stroke_at_first_contact"]),
    )
    draw.text(
        (48, 990),
        f"speed {info['v_slope']:.2f} m/s   pick contact {'YES' if contact else 'NO'}",
        font=detail_font,
        fill=status_colour,
    )
    draw.text(
        (665, 990),
        f"plant travel {100.0 * plant_travel:.1f} cm   "
        f"blade {info['pick_blade_into_slope_angle_deg']:.1f}° into slope   "
        f"snow load {info['snow_drag_force']:.0f} N",
        font=detail_font,
        fill=(225, 234, 242, 255),
    )
    return np.asarray(image)


def _summary_frame(
    reports: list[dict[str, Any]], width: int, height: int
) -> np.ndarray:
    image = Image.new("RGB", (width, height), (9, 17, 27))
    draw = ImageDraw.Draw(image)
    draw.text(
        (100, 72),
        "DIVERSE OBLIQUE-FALL SUITE",
        font=_font(46),
        fill=(245, 248, 252),
    )
    draw.text(
        (100, 145),
        "Same PPO checkpoint · non-axe body friction disabled in every rollout",
        font=_font(27),
        fill=(175, 215, 245),
    )
    y = 235
    for index, report in enumerate(reports):
        passed = report["success"]
        colour = (85, 245, 120) if passed else (255, 110, 85)
        draw.text(
            (120, y),
            f"{index + 1}. {report['label']}",
            font=_font(25),
            fill=(235, 240, 246),
        )
        draw.text(
            (1120, y),
            f"{'ARRESTED' if passed else 'FAILED'}  ·  {report['final_speed_mps']:.2f} m/s",
            font=_font(25),
            fill=colour,
        )
        y += 92
    return np.asarray(image)


def record_suite(
    model_path: str | None,
    output_video: str,
    seed: int,
    fps: int = 50,
) -> dict[str, Any]:
    checkpoint = resolve_model_path(model_path)
    policy = PPO.load(checkpoint, device="cpu")
    output = Path(output_video)
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_dir = output.parent / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    width, height = 1920, 1080
    writer = imageio.get_writer(
        output,
        fps=fps,
        codec="libx264",
        quality=9,
        macro_block_size=1,
    )
    reports: list[dict[str, Any]] = []
    try:
        for scenario_index, scenario in enumerate(SCENARIOS):
            env = G1SelfArrestEnv(randomize_reset=False)
            env.set_body_slope_friction_enabled(False)
            observation, reset_info = env.reset(
                seed=seed + scenario_index, options=scenario.reset_options()
            )
            main_renderer = mujoco.Renderer(env.model, height=height, width=width)
            hand_renderer = mujoco.Renderer(env.model, height=420, width=620)
            top_renderer = mujoco.Renderer(env.model, height=285, width=420)
            main_camera = mujoco.MjvCamera()
            main_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            main_camera.lookat[:] = [-0.4, 0.0, -0.2]
            # This fixed, wider view lets the robot traverse the frame while
            # terrain relief and shadows establish the fall line.
            main_camera.distance = 12.5
            main_camera.azimuth = 108
            main_camera.elevation = -18
            hand_camera = mujoco.MjvCamera()
            hand_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            hand_camera.distance = 0.62
            hand_camera.azimuth = 235
            hand_camera.elevation = -28
            top_camera = mujoco.MjvCamera()
            top_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            top_camera.distance = 4.0
            top_camera.azimuth = 90
            top_camera.elevation = -89
            left_wrist = mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"
            )
            right_wrist = mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"
            )
            final_info: dict[str, Any] = {}
            start_frame: np.ndarray | None = None
            contact_frame: np.ndarray | None = None
            final_frame: np.ndarray | None = None
            for step in range(env.max_episode_steps):
                action, _ = policy.predict(observation, deterministic=True)
                observation, _, terminated, truncated, final_info = env.step(action)
                if step % 2 == 0:
                    contact_now = final_info["rigid_pick_contact"] > 0.5
                    env.model.site_size[env.axe_pick_site_id, 0] = 0.012
                    env.model.site_rgba[env.axe_pick_site_id] = (
                        [0.05, 1.0, 0.10, 1.0]
                        if contact_now
                        else [1.0, 0.12, 0.05, 0.80]
                    )
                    main_renderer.update_scene(env.data, camera=main_camera)
                    frame = main_renderer.render().copy()
                    hand_camera.lookat[:] = 0.5 * (
                        env.data.xpos[left_wrist] + env.data.xpos[right_wrist]
                    )
                    hand_renderer.update_scene(env.data, camera=hand_camera)
                    hand_frame = hand_renderer.render()
                    top_camera.lookat[:] = env.data.xpos[env.pelvis_body_id]
                    top_renderer.update_scene(env.data, camera=top_camera)
                    top_frame = top_renderer.render()
                    _bordered_inset(frame, hand_frame, x=24, y=24)
                    _bordered_inset(frame, top_frame, x=1470, y=24)
                    slow_motion = step <= 60
                    frame = _annotate(
                        frame,
                        scenario,
                        final_info,
                        slow_motion=slow_motion,
                        scenario_index=scenario_index,
                    )
                    if start_frame is None:
                        start_frame = frame.copy()
                        for _ in range(fps // 2):
                            writer.append_data(frame)
                    if contact_now and contact_frame is None:
                        contact_frame = frame.copy()
                    for _ in range(2 if slow_motion else 1):
                        writer.append_data(frame)
                    final_frame = frame.copy()
                if terminated or truncated:
                    break
            if final_frame is None or start_frame is None:
                raise RuntimeError(f"No frames recorded for {scenario.name}")
            for _ in range(fps // 2):
                writer.append_data(final_frame)
            if contact_frame is None:
                contact_frame = final_frame
            imageio.imwrite(snapshot_dir / f"{scenario.name}_start.png", start_frame)
            imageio.imwrite(
                snapshot_dir / f"{scenario.name}_contact.png", contact_frame
            )
            imageio.imwrite(snapshot_dir / f"{scenario.name}_final.png", final_frame)
            reports.append(
                {
                    **scenario.to_dict(),
                    "success": bool(final_info.get("success", False)),
                    "policy_steps": step + 1,
                    "initial_total_slope_speed_mps": float(reset_info["v_slope"]),
                    "final_speed_mps": float(final_info["v_slope"]),
                    "stopping_distance_m": float(final_info["stopping_distance"]),
                    "first_contact_step": int(final_info["first_rigid_contact_step"]),
                    "stroke_at_first_contact_m": float(
                        final_info["stroke_at_first_contact"]
                    ),
                    "lowering_at_first_contact_m": float(
                        final_info["lowering_at_first_contact"]
                    ),
                    "first_contact_blade_angle_deg": float(
                        final_info["blade_angle_at_first_contact_deg"]
                    ),
                    "terminal_rolling_blade_angle_deg": float(
                        final_info["rolling_pick_blade_into_slope_angle_deg"]
                    ),
                    "terminal_rigid_pick_contact_fraction": float(
                        final_info["rolling_rigid_pick_contact_fraction"]
                    ),
                    "terminal_mean_snow_drag_force_n": float(
                        final_info["rolling_mean_snow_drag_force"]
                    ),
                    "left_grasp_contact_fraction": float(
                        final_info["left_grasp_contact_fraction"]
                    ),
                    "right_grasp_contact_fraction": float(
                        final_info["right_grasp_contact_fraction"]
                    ),
                    "grip_score": float(final_info["grip_score"]),
                    "valid_learned_plant_motion": bool(
                        final_info["valid_learned_plant_motion"]
                    ),
                }
            )
            main_renderer.close()
            hand_renderer.close()
            top_renderer.close()
            env.close()
        summary = _summary_frame(reports, width, height)
        for _ in range(2 * fps):
            writer.append_data(summary)
    finally:
        writer.close()
    report: dict[str, Any] = {
        "checkpoint": str(checkpoint.resolve()),
        "video": str(output.resolve()),
        "one_checkpoint_for_every_scenario": True,
        "non_axe_body_friction_enabled": False,
        "all_scenarios_succeeded": all(row["success"] for row in reports),
        "scenarios": reports,
        "playback": (
            "Each first 0.61 simulation-seconds is shown at 0.5x; the "
            "remaining arrest is real time. Actions and physics are unchanged."
        ),
        "output_resolution": "1920x1080",
        "fps": fps,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["all_scenarios_succeeded"]:
        failed = [row["name"] for row in reports if not row["success"]]
        raise RuntimeError(f"Video suite contains failed scenarios: {failed}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument(
        "--output", default="videos/g1_self_arrest_diverse_suite.mp4"
    )
    parser.add_argument("--seed", type=int, default=81_000)
    arguments = parser.parse_args()
    record_suite(arguments.model, arguments.output, arguments.seed)
