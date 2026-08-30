"""Offscreen rendering with a telemetry overlay.

The overlay is the point. A video of two humanoids picking up a log shows
that something happened; it does not show that the load was weighed before
it was lifted, that it stayed within a degree of level, or that the two
machines split it evenly. Those are the claims, so they go on screen next
to the thing they describe.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .controller import PHASE_NAMES, Telemetry
from .scene import ROBOTS

# --- palette (dark, legible over an alpine scene) --------------------------
INK = (238, 242, 248)
DIM = (150, 162, 178)
PANEL = (14, 18, 26, 205)
ACCENT = (108, 196, 255)
GOOD = (104, 214, 148)
WARN = (245, 190, 90)
BAD = (240, 108, 96)
SLING = (236, 110, 70)


@dataclass
class CameraShot:
    name: str
    lookat: tuple[float, float, float]
    distance: float
    azimuth: float
    elevation: float
    track_payload: bool = False


SHOTS: dict[str, CameraShot] = {
    "wide":   CameraShot("wide", (0.0, 0.0, 0.52), 3.5, 132.0, -11.0),
    "front":  CameraShot("front", (0.0, 0.0, 0.58), 2.9, 90.0, -6.0),
    "hands":  CameraShot("hands", (0.0, 0.52, 0.48), 1.5, 150.0, -12.0, True),
    "low":    CameraShot("low", (0.0, 0.0, 0.40), 3.0, 200.0, -4.0),
    "top":    CameraShot("top", (0.0, 0.0, 0.45), 3.6, 128.0, -46.0),
}


def _font(size: int):
    for path in (
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


class Overlay:
    """Draws the telemetry panel onto a rendered frame."""

    def __init__(self, width: int, height: int, title: str = ""):
        self.w, self.h = width, height
        self.title = title
        s = height / 720.0
        self.s = s
        self.f_big = _font(int(30 * s))
        self.f_mid = _font(int(19 * s))
        self.f_small = _font(int(15 * s))
        self.f_tiny = _font(int(13 * s))

    # -- primitives ---------------------------------------------------------
    def _panel(self, dr, xy, wh, radius=10):
        x, y = xy
        w, h = wh
        dr.rounded_rectangle([x, y, x + w, y + h], radius=radius * self.s, fill=PANEL)

    def _bar(self, dr, x, y, w, h, frac, colour, bg=(46, 54, 68, 255)):
        dr.rounded_rectangle([x, y, x + w, y + h], radius=h / 2, fill=bg)
        f = float(np.clip(frac, 0.0, 1.0))
        if f > 0.01:
            dr.rounded_rectangle([x, y, x + w * f, y + h], radius=h / 2, fill=colour)

    # -- main ---------------------------------------------------------------
    def draw(self, frame: np.ndarray, t: Telemetry, extra: dict | None = None) -> np.ndarray:
        img = Image.fromarray(frame).convert("RGBA")
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(layer)
        s = self.s
        extra = extra or {}

        # ---------------- header ----------------
        self._panel(dr, (int(24 * s), int(20 * s)), (int(520 * s), int(76 * s)))
        dr.text((int(40 * s), int(30 * s)), self.title or "ALPINE COORDINATED LIFT",
                font=self.f_mid, fill=ACCENT)
        phase = t.phase
        pcol = BAD if t.aborted else (GOOD if phase in ("HOLD", "SHIFT") else INK)
        dr.text((int(40 * s), int(56 * s)), f"{t.t:6.2f}s   {phase}",
                font=self.f_big, fill=pcol)

        # phase ribbon
        y = int(108 * s)
        x = int(24 * s)
        for name in PHASE_NAMES:
            wseg = int(62 * s)
            done = (not t.aborted) and PHASE_NAMES.index(name) < t.phase_idx
            cur = (not t.aborted) and name == phase
            col = ACCENT if cur else (GOOD if done else (52, 60, 74, 255))
            dr.rounded_rectangle([x, y, x + wseg - int(4 * s), y + int(7 * s)],
                                 radius=3 * s, fill=col)
            x += wseg
        if t.aborted:
            dr.text((int(24 * s), y + int(14 * s)), f"ABORT - {t.abort_reason}",
                    font=self.f_small, fill=BAD)

        # ---------------- left telemetry ----------------
        px, py = int(24 * s), int(150 * s)
        self._panel(dr, (px, py), (int(330 * s), int(268 * s)))
        cx, cy = px + int(18 * s), py + int(16 * s)

        def row(label, value, colour=INK, dy=26):
            nonlocal cy
            dr.text((cx, cy), label, font=self.f_small, fill=DIM)
            dr.text((cx + int(168 * s), cy), value, font=self.f_small, fill=colour)
            cy += int(dy * s)

        row("lift", f"{t.payload_lift * 100:+6.1f} cm", GOOD if t.payload_lift > 0.02 else INK)
        row("offset", f"{t.payload_x * 100:+6.1f} cm")
        tcol = GOOD if abs(t.tilt_deg) < 4 else (WARN if abs(t.tilt_deg) < 12 else BAD)
        row("tilt", f"{t.tilt_deg:+6.2f} deg", tcol)
        row("mass est", f"{t.mass_est:6.1f} kg", ACCENT)
        row("pace", f"{t.pace * 100:5.0f} %", WARN if t.pace < 0.7 else INK)
        cy += int(6 * s)

        # load share bar
        dr.text((cx, cy), "load share   A", font=self.f_small, fill=DIM)
        dr.text((cx + int(250 * s), cy), "B", font=self.f_small, fill=DIM)
        cy += int(22 * s)
        bw, bh = int(292 * s), int(14 * s)
        share = float(np.clip(t.share, 0.0, 1.0))
        bal = abs(share - 0.5)
        scol = GOOD if bal < 0.08 else (WARN if bal < 0.2 else BAD)
        self._bar(dr, cx, cy, bw, bh, share, scol)
        mid = cx + bw / 2
        dr.line([mid, cy - 3 * s, mid, cy + bh + 3 * s], fill=INK, width=max(1, int(2 * s)))
        cy += int(bh + 8 * s)
        dr.text((cx, cy), f"{share * 100:.0f} / {100 - share * 100:.0f}",
                font=self.f_tiny, fill=scol)
        cy += int(24 * s)

        gcol = GOOD if t.go else BAD
        dr.text((cx, cy), "GO" if t.go else "NO-GO", font=self.f_mid, fill=gcol)
        reason = extra.get("go_reason", "")
        if reason:
            dr.text((cx + int(58 * s), cy + int(4 * s)), reason[:34],
                    font=self.f_tiny, fill=DIM)

        # ---------------- right: per-hand force ----------------
        pw = int(268 * s)
        px2 = self.w - pw - int(24 * s)
        py2 = int(150 * s)
        self._panel(dr, (px2, py2), (pw, int(214 * s)))
        cx2, cy2 = px2 + int(18 * s), py2 + int(16 * s)
        dr.text((cx2, cy2), "SLING LOAD", font=self.f_small, fill=ACCENT)
        cy2 += int(28 * s)
        limit = extra.get("sling_limit", 340.0)
        for p in ROBOTS:
            for side in ("left", "right"):
                key = f"{p}{side}"
                f = t.hand_force.get(key, 0.0)
                ok = extra.get("sling_ok", {}).get(key, True)
                frac = f / max(limit, 1e-6)
                col = BAD if not ok else (
                    GOOD if frac < 0.5 else (WARN if frac < 0.8 else BAD)
                )
                lbl = f"{p[0]}-{side[0].upper()}"
                dr.text((cx2, cy2), lbl, font=self.f_tiny, fill=DIM if ok else BAD)
                self._bar(dr, cx2 + int(44 * s), cy2 + int(3 * s),
                          int(140 * s), int(10 * s), frac, col)
                txt = "SLIP" if not ok else f"{f:5.0f}N"
                dr.text((cx2 + int(194 * s), cy2), txt, font=self.f_tiny, fill=col)
                cy2 += int(24 * s)
        cy2 += int(6 * s)
        dr.text((cx2, cy2), "BALANCE MARGIN", font=self.f_small, fill=ACCENT)
        cy2 += int(24 * s)
        for p in ROBOTS:
            m = t.cp_margin.get(p, 0.0)
            col = GOOD if m > 0.02 else (WARN if m > -0.02 else BAD)
            dr.text((cx2, cy2), f"robot {p[0]}", font=self.f_tiny, fill=DIM)
            dr.text((cx2 + int(90 * s), cy2), f"{m * 100:+5.1f} cm", font=self.f_tiny, fill=col)
            cy2 += int(20 * s)

        # ---------------- events ----------------
        if t.events:
            ex, ey = int(24 * s), self.h - int(30 * s) - int(22 * s) * min(len(t.events), 4)
            self._panel(dr, (ex - int(10 * s), ey - int(10 * s)),
                        (int(470 * s), int(22 * s) * min(len(t.events), 4) + int(20 * s)))
            for et, msg in t.events[-4:]:
                col = BAD if ("ABORT" in msg or "slip" in msg) else DIM
                dr.text((ex, ey), f"{et:6.2f}s  {msg}", font=self.f_tiny, fill=col)
                ey += int(22 * s)

        out = Image.alpha_composite(img, layer).convert("RGB")
        return np.asarray(out)


class Recorder:
    """Renders a mission to frames, optionally from several cameras."""

    def __init__(self, model, width=1280, height=720, shot="wide", title=""):
        self.model = model
        self.w, self.h = width, height
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.overlay = Overlay(width, height, title)
        self.cam = mujoco.MjvCamera()
        self.set_shot(shot)
        # Aerial perspective. MuJoCo's <visual><map fogstart/fogend> only
        # takes effect with the FOG render flag set on the scene, and without
        # it a 200 m peak renders as crisply as the robot in front of it,
        # which flattens the whole backdrop into grey slabs.
        self._rndflags = {
            mujoco.mjtRndFlag.mjRND_SHADOW: 1,
            mujoco.mjtRndFlag.mjRND_FOG: 1,
            mujoco.mjtRndFlag.mjRND_HAZE: 1,
            mujoco.mjtRndFlag.mjRND_REFLECTION: 1,
        }
        for k, v in self._rndflags.items():
            self.renderer.scene.flags[k] = v
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[3] = 0     # hide collision primitives
        self.opt.geomgroup[4] = 0     # hide sites/handles

    def set_shot(self, shot: str | CameraShot) -> None:
        sh = SHOTS[shot] if isinstance(shot, str) else shot
        self.shot = sh
        self.cam.lookat[:] = sh.lookat
        self.cam.distance = sh.distance
        self.cam.azimuth = sh.azimuth
        self.cam.elevation = sh.elevation

    def frame(self, data, tele: Telemetry, extra: dict | None = None,
              hud: bool = True) -> np.ndarray:
        if self.shot.track_payload:
            pid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "payload")
            self.cam.lookat[:] = data.xpos[pid]
        self.renderer.update_scene(data, self.cam, self.opt)
        for k, v in self._rndflags.items():
            self.renderer.scene.flags[k] = v
        img = self.renderer.render()
        return self.overlay.draw(img, tele, extra) if hud else img

    def close(self):
        self.renderer.close()


def write_video(path: str, frames, fps: int = 50) -> str:
    import imageio.v2 as imageio

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                            macro_block_size=8) as w:
        for f in frames:
            w.append_data(f)
    return path
