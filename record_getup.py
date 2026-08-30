"""Render an evidence-first physical fall and learned G1 get-up video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from getup_controller import DEFAULT_FALL, G1PhysicalGetup


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def _phase_label(info: dict[str, Any]) -> tuple[str, str]:
    phase = info["phase"]
    if phase == "pre_fall":
        return "SETTLED STAND", "motors supporting stance · disturbance queued"
    if phase == "fall":
        return "PHYSICAL FALL", "finite shove → gravity → impact"
    if phase == "floor_ready":
        return "GROUNDED ALIGNMENT", "joint-space only · floating base remains free"
    if phase == "recovery":
        return "LEARNED GET-UP", "pretrained RL WBC inference at 50 Hz"
    return "RECOVERY COMPLETE", "stable upright stance"


def _annotate(frame: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    phase_title, phase_detail = _phase_label(info)
    width, height = image.size

    draw.rounded_rectangle(
        (28, 24, 696, 146), radius=14,
        fill=(8, 15, 25, 224), outline=(208, 225, 238, 235), width=2,
    )
    draw.text((52, 42), phase_title, font=_font(31, bold=True), fill=(246, 250, 253, 255))
    draw.text((52, 91), phase_detail, font=_font(21), fill=(176, 216, 242, 255))

    draw.rounded_rectangle(
        (width - 536, 24, width - 28, 184), radius=14,
        fill=(8, 15, 25, 224), outline=(208, 225, 238, 235), width=2,
    )
    draw.text((width - 510, 42), "CAUSAL PHYSICS", font=_font(25, bold=True), fill=(246, 250, 253, 255))
    rows = (
        f"gravity  9.81 m/s²     solver  Newton",
        f"push  {info['applied_push_force_n']:.0f} N     base teleports  {info['root_teleports_after_fall_start']}",
        f"contacts  {info['contact_count']}     penetration  {1000 * info['maximum_contact_penetration_m']:.1f} mm",
    )
    for index, row in enumerate(rows):
        draw.text((width - 510, 82 + 30 * index), row, font=_font(18), fill=(218, 228, 236, 255))

    draw.rounded_rectangle(
        (28, height - 132, width - 28, height - 24), radius=14,
        fill=(8, 15, 25, 228), outline=(102, 174, 220, 240), width=2,
    )
    policy_colour = (91, 241, 130, 255) if info["policy_inference_active"] else (190, 201, 210, 255)
    draw.text(
        (52, height - 112),
        f"pelvis {info['pelvis_height_m']:.2f} m     upright {info['torso_upright']:.3f}     "
        f"linear {info['base_linear_speed_mps']:.2f} m/s     angular {info['base_angular_speed_radps']:.2f} rad/s     "
        f"motor {100 * info['peak_motor_torque_ratio']:.0f}%",
        font=_font(23, bold=True), fill=(242, 247, 251, 255),
    )
    draw.text(
        (52, height - 70),
        "POLICY INFERENCE ACTIVE" if info["policy_inference_active"] else "POLICY INFERENCE WAITING",
        font=_font(20, bold=True), fill=policy_colour,
    )
    contacts = ", ".join(info["ground_contact_bodies"][:4]) or "none"
    draw.text(
        (430, height - 70), f"ground contacts: {contacts}",
        font=_font(19), fill=(201, 221, 235, 255),
    )
    return np.asarray(image)


def _title_frame(width: int, height: int) -> np.ndarray:
    image = Image.new("RGB", (width, height), (8, 15, 25))
    draw = ImageDraw.Draw(image)
    draw.text((82, 92), "FALL → GROUND → GET UP", font=_font(49, bold=True), fill=(244, 249, 252))
    draw.text((84, 166), "Unitree G1 · contact-rich MuJoCo · learned whole-body recovery", font=_font(27), fill=(173, 215, 242))
    lines = (
        "1  A finite 100 N shove initiates the fall from a settled stand",
        "2  Gravity and 2 ms contact dynamics produce and settle the impact",
        "3  The pretrained RL WBC acts at 50 Hz; torque limits remain enforced",
        "No floating-base pose is written after the fall starts",
    )
    for index, line in enumerate(lines):
        colour = (90, 239, 132) if index == 3 else (222, 230, 237)
        draw.text((112, 274 + 65 * index), line, font=_font(24, bold=index == 3), fill=colour)
    draw.text(
        (84, height - 68),
        "Policy: wbc-mjlab/wbc-g1-deploy (Apache-2.0) · adapted here, not trained here",
        font=_font(19), fill=(158, 180, 197),
    )
    return np.asarray(image)


def _summary_frame(width: int, height: int, report: dict[str, Any]) -> np.ndarray:
    image = Image.new("RGB", (width, height), (8, 15, 25))
    draw = ImageDraw.Draw(image)
    passed = report["success"]
    draw.text((86, 86), "PHYSICAL RECOVERY VERIFIED" if passed else "RECOVERY FAILED", font=_font(46, bold=True), fill=(86, 242, 127) if passed else (255, 112, 90))
    metrics = (
        ("final pelvis height", f"{report['final_pelvis_height_m']:.3f} m"),
        ("final torso upright", f"{report['final_torso_upright']:.3f}"),
        ("final linear speed", f"{report['final_base_linear_speed_mps']:.3f} m/s"),
        ("final angular speed", f"{report['final_base_angular_speed_radps']:.3f} rad/s"),
        ("peak motor limit", f"{100 * report['peak_motor_torque_ratio']:.1f}%"),
        ("maximum penetration", f"{1000 * report['maximum_contact_penetration_m']:.1f} mm"),
        ("floating-base teleports", str(report["root_teleports_after_fall_start"])),
    )
    for index, (label, value) in enumerate(metrics):
        y = 202 + index * 62
        draw.text((120, y), label, font=_font(24), fill=(188, 207, 220))
        draw.text((780, y), value, font=_font(25, bold=True), fill=(240, 246, 250))
    draw.text((86, height - 66), "Causal ablations and perturbed-fall results: videos/g1_physical_getup.json", font=_font(19), fill=(158, 180, 197))
    return np.asarray(image)


def record(output_video: str, report_path: str, fps: int = 50) -> dict[str, Any]:
    output = Path(output_video)
    report_output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1600, 900
    controller = G1PhysicalGetup()
    controller.reset(condition=DEFAULT_FALL, mode="policy")
    renderer = mujoco.Renderer(controller.model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [-0.28, 0.08, 0.52]
    camera.distance = 2.75
    camera.azimuth = 128
    camera.elevation = -17
    writer = imageio.get_writer(
        output, fps=fps, codec="libx264", quality=9, macro_block_size=1
    )
    title = _title_frame(width, height)
    for _ in range(int(1.4 * fps)):
        writer.append_data(title)
    last_info = controller.telemetry()
    try:
        for info in controller.rollout():
            last_info = info
            renderer.update_scene(controller.data, camera)
            writer.append_data(_annotate(renderer.render(), info))
        report = controller.report()
        summary = _summary_frame(width, height, report)
        for _ in range(int(2.0 * fps)):
            writer.append_data(summary)
    finally:
        writer.close()
        renderer.close()
        controller.close()
    report.update(
        {
            "video": str(output),
            "video_fps": fps,
            "video_resolution": [width, height],
            "policy_provenance": {
                "source": "https://github.com/wbc-mjlab/wbc-g1-deploy",
                "commit": "6dabf86fddc2b7b429b09e74999732fcde3441f9",
                "license": "Apache-2.0",
                "trained_here": False,
            },
            "last_frame_policy_inference_active": last_info["policy_inference_active"],
        }
    )
    report_output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="videos/g1_physical_getup.mp4")
    parser.add_argument("--report", default="videos/g1_physical_getup.json")
    args = parser.parse_args()
    result = record(args.output, args.report)
    print(json.dumps(result, indent=2))
    if not result["success"]:
        raise SystemExit("Rendered rollout did not finish in a stable stand")


if __name__ == "__main__":
    main()
