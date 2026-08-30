"""Validate inclined handline grip, boot contact, and alternating steps."""

from __future__ import annotations

from pathlib import Path
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixed_line_slope_env import G1FixedLineEnv
from fixed_line_visuals import add_braided_rope_visual


def render_view(
    env: G1FixedLineEnv,
    output: Path,
    *,
    azimuth: float,
) -> None:
    renderer = mujoco.Renderer(env.model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = env.pelvis_body_id
    camera.distance = 2.45
    camera.azimuth = azimuth
    camera.elevation = -8
    renderer.update_scene(env.data, camera=camera)
    add_braided_rope_visual(renderer, env.model, env.data)
    imageio.imwrite(output, renderer.render())
    renderer.close()


def reference_action(env: G1FixedLineEnv) -> np.ndarray:
    if env._progress() - env._start_progress >= env.target_ascent:
        return np.zeros(env.action_dim, dtype=np.float32)
    side = env._swing_side if env._swing_side is not None else env._expected_side
    return (
        np.asarray([1.0, -1.0, 1.0], dtype=np.float32)
        if side == 0
        else np.asarray([-1.0, 1.0, 1.0], dtype=np.float32)
    )


def validate() -> dict[str, float]:
    output = Path("videos/snapshots")
    output.mkdir(parents=True, exist_ok=True)
    env = G1FixedLineEnv(randomize_reset=False)
    _observation, metrics = env.reset(options={"randomize": False})
    rows = [metrics]
    swing_rendered = False
    for _step in range(env.max_episode_steps):
        _observation, _reward, terminated, truncated, metrics = env.step(
            reference_action(env)
        )
        rows.append(metrics)
        if not swing_rendered and 0.47 < env._swing_phase < 0.56:
            render_view(
                env,
                output / "fixed_line_slope_swing.png",
                azimuth=112,
            )
            swing_rendered = True
        if terminated or truncated:
            break

    result = {
        "slope_angle_degrees": float(np.rad2deg(env.slope_angle)),
        "success": float(metrics["success"]),
        "along_slope_ascent_m": float(metrics["ascent"]),
        "vertical_gain_m": float(metrics["vertical_gain_m"]),
        "completed_alternating_steps": float(metrics["completed_cycles"]),
        "rope_core_collision_steps": float(
            sum(row["rope_core_collision"] for row in rows)
        ),
        "maximum_rope_extension_m": float(
            max(row["rope_extension_m"] for row in rows)
        ),
        "maximum_rope_deformation_m": float(
            max(row["rope_max_displacement_m"] for row in rows)
        ),
        "maximum_rope_contacts": float(
            max(row["rope_contact_count"] for row in rows)
        ),
        "maximum_hand_rope_penetration_m": float(
            max(row["hand_rope_max_penetration_m"] for row in rows)
        ),
        "hand_rope_contact_steps": float(
            sum(row["hand_rope_contact_count"] > 0.0 for row in rows)
        ),
        "maximum_rope_guide_load_n": float(
            max(row["rope_guide_load_n"] for row in rows)
        ),
        "maximum_arm_pull_load_n": float(
            max(row["arm_pull_load_n"] for row in rows)
        ),
        "arm_pull_impulse_ns": float(metrics["arm_pull_impulse_ns"]),
        "minimum_grasp_score": float(min(row["grasp_score"] for row in rows)),
        "maximum_right_grip_error_m": float(
            max(row["right_grip_error"] for row in rows)
        ),
        "minimum_hand_separation_m": float(
            min(row["hand_separation"] for row in rows)
        ),
        "grounded_fraction": float(metrics["grounded_fraction"]),
        "left_boot_contact_fraction": float(
            metrics["left_boot_contact_fraction"]
        ),
        "right_boot_contact_fraction": float(
            metrics["right_boot_contact_fraction"]
        ),
        "maximum_airborne_streak": float(metrics["maximum_airborne_streak"]),
        "cross_hand_collision_steps": float(
            sum(row["cross_hand_collision"] for row in rows)
        ),
        "slope_hand_collision_steps": float(
            sum(row["wall_hand_collision"] for row in rows)
        ),
    }
    assert result["slope_angle_degrees"] > 20.0, result
    assert result["success"] == 1.0, result
    assert result["along_slope_ascent_m"] >= 1.50, result
    assert result["vertical_gain_m"] > 0.60, result
    assert result["completed_alternating_steps"] >= 8.0, result
    assert result["rope_core_collision_steps"] == 0.0, result
    assert 0.005 < result["maximum_rope_extension_m"] < 0.10, result
    assert result["maximum_rope_deformation_m"] > 0.03, result
    assert result["maximum_hand_rope_penetration_m"] <= 8e-4, result
    assert result["minimum_grasp_score"] > 0.32, result
    assert result["maximum_right_grip_error_m"] < 0.14, result
    assert result["minimum_hand_separation_m"] > 0.12, result
    assert result["grounded_fraction"] > 0.90, result
    assert result["left_boot_contact_fraction"] > 0.30, result
    assert result["right_boot_contact_fraction"] > 0.30, result
    assert result["maximum_airborne_streak"] <= 4.0, result
    assert result["cross_hand_collision_steps"] == 0.0, result
    assert result["slope_hand_collision_steps"] == 0.0, result

    render_view(env, output / "fixed_line_grasp_side.png", azimuth=112)
    render_view(env, output / "fixed_line_grasp_reverse.png", azimuth=248)
    env.close()
    print(result)
    return result


if __name__ == "__main__":
    validate()
