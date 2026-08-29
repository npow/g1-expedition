"""Shared alpine presentation helpers for the fixed-line videos."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageOps


ALPINE_BACKDROP = (
    Path(__file__).resolve().parent
    / "assets"
    / "environments"
    / "fixed_line_himalaya.png"
)


@lru_cache(maxsize=8)
def alpine_backdrop(width: int, height: int) -> np.ndarray:
    """Load and crop the distant mountain plate for a render resolution."""
    with Image.open(ALPINE_BACKDROP) as source:
        fitted = ImageOps.fit(
            source.convert("RGB"),
            (int(width), int(height)),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.48),
        )
        return np.asarray(fitted).copy()


def render_with_alpine_backdrop(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera: mujoco.MjvCamera,
    backdrop: np.ndarray,
) -> np.ndarray:
    """Render the 3-D scene while replacing only the synthetic skybox.

    The physical slope, its moving texture, contact shadows, the robot, and
    all climbing equipment stay native MuJoCo pixels.  Only pixels at the
    renderer's far plane are replaced, so the mountain range reads as a
    distant environment instead of a foreground photograph glued to the feet.
    """
    renderer.update_scene(data, camera=camera)
    frame = renderer.render().copy()
    renderer.enable_depth_rendering()
    try:
        depth = renderer.render().copy()
    finally:
        renderer.disable_depth_rendering()
    frame[depth > 100.0] = backdrop[depth > 100.0]
    return frame
