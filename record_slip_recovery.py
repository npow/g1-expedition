"""Render walk -> slip -> recover -> move as one legible sequence.

Fixed camera, deliberately
--------------------------
`G1FixedLineSlopeEnv.render()` uses `mjCAMERA_TRACKING` locked to the pelvis,
so the robot sits dead centre and translation is invisible -- you cannot see it
resume climbing because the frame climbs with it. npow already solved this for
the self-arrest suite ("a fixed 12.5 m camera, so the robot crosses the image
instead of being tracked at the center") and the same fix applies here: park
the camera and let the robot traverse.

Phase banner
------------
A colour bar across the top names the phase the state machine is actually in,
so the four beats are readable without narration:

    slate   WALK      ascending, undisturbed
    red     SLIP      impulse live, sliding back down the line
    amber   RECOVER   shove over, regaining stance, not yet confirmed
    green   MOVING    recovery confirmed -- stance held and distance added

Confirmation is deliberately strict. Regaining footing for an instant is not
recovery; the bar only turns green once the robot has held it and put real
distance on the board.

    MUJOCO_GL=glfw python record_slip_recovery.py --impulse 700
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "glfw")  # macOS has no egl/osmesa

import mujoco
import numpy as np
from stable_baselines3 import PPO

import slip_recovery_env

PHASES = {
    "walk": ((70, 84, 96), "WALK"),
    "slip": ((198, 58, 48), "SLIP"),
    "recover": ((214, 148, 40), "RECOVER"),
    "moving": ((58, 168, 108), "MOVING"),
}


class Snowfall:
    """Falling snow, composited onto rendered frames.

    Deliberately NOT simulated. Adding real particle geoms would put hundreds
    of bodies in the model, change `njmax`, and risk contacts with the robot --
    i.e. it would alter the physics of the very rollout being filmed. npow's
    scene takes the same line with its wind-crust relief and route wands:
    visual-only geoms with contype=0/conaffinity=0. This goes one step further
    and never enters the model at all.

    Three parallax layers. Near flakes are larger, brighter and fall faster;
    far flakes are small and dim. Wind shear is a function of depth, so the
    layers drift apart rather than moving as one sheet -- that separation is
    what sells depth in an otherwise flat overlay.
    """

    # (count, radius px, fall px/frame, drift px/frame, alpha, rgb)
    # Colour matters as much as alpha here. Additive white worked against the
    # sky and vanished completely over the slope -- half the frame got no snow,
    # because brightening near-white pixels does nothing. Alpha-blending toward
    # a per-layer colour fixes it: far flakes stay near-white and read against
    # the sky, while the near layer is a cooler blue-grey that holds contrast
    # over bright snow the way an out-of-focus flake actually does.
    LAYERS = (
        (150, 1, 2.2, 0.7, 0.34, (252, 254, 255)),
        (90, 2, 4.0, 1.3, 0.55, (232, 241, 250)),
        (45, 3, 6.5, 2.1, 0.80, (203, 220, 238)),
    )

    def __init__(self, width: int, height: int, seed: int = 0,
                 intensity: float = 1.0, gust: float = 0.35) -> None:
        self.w, self.h = width, height
        self.gust = float(gust)
        rng = np.random.default_rng(seed)
        self.layers = []
        for count, radius, speed, drift, alpha, rgb in self.LAYERS:
            n = max(int(count * intensity), 0)
            self.layers.append({
                "xy": rng.uniform([0, 0], [width, height], size=(n, 2)),
                "radius": radius,
                "speed": speed,
                "drift": drift,
                "alpha": alpha,
                "rgb": np.asarray(rgb, dtype=np.float32),
                # Per-flake phase so they do not sway in lockstep.
                "phase": rng.uniform(0.0, 2 * np.pi, size=n),
            })
        self.t = 0.0

    def step(self) -> None:
        self.t += 0.06
        for layer in self.layers:
            xy = layer["xy"]
            sway = np.sin(self.t + layer["phase"]) * self.gust
            xy[:, 0] += layer["drift"] + sway
            xy[:, 1] += layer["speed"]
            # Recycle off the top once a flake leaves the bottom or the side,
            # so density stays constant instead of thinning out over the clip.
            gone = xy[:, 1] >= self.h
            xy[gone, 1] -= self.h
            xy[gone, 0] = np.random.default_rng().uniform(0, self.w, gone.sum())
            xy[:, 0] %= self.w

    def draw(self, frame: np.ndarray) -> np.ndarray:
        out = frame.astype(np.float32)
        for layer in self.layers:
            r = layer["radius"]
            alpha = layer["alpha"]
            colour = layer["rgb"]
            xs = layer["xy"][:, 0].astype(np.int32)
            ys = layer["xy"][:, 1].astype(np.int32)
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    d2 = dx * dx + dy * dy
                    if d2 > r * r:
                        continue
                    # Soften the rim so flakes are discs, not squares.
                    a = alpha * (1.0 if d2 <= (r - 1) ** 2 else 0.45)
                    px = np.clip(xs + dx, 0, self.w - 1)
                    py = np.clip(ys + dy, 0, self.h - 1)
                    out[py, px] = out[py, px] * (1.0 - a) + colour * a
        return np.clip(out, 0, 255).astype(np.uint8)


def phase_of(info: dict) -> str:
    if info.get("slip_active", 0) > 0.5:
        return "slip"
    if info.get("recovered", 0) > 0.5:
        return "moving"
    if info.get("slip_triggered", 0) > 0.5:
        return "recover"
    return "walk"


def glyphs(name: str, height: int) -> np.ndarray:
    """Block-letter phase name. Avoids a font dependency for four words."""
    font = {
        "W": ["X   X", "X   X", "X X X", "XX XX", "X   X"],
        "A": [" XXX ", "X   X", "XXXXX", "X   X", "X   X"],
        "L": ["X    ", "X    ", "X    ", "X    ", "XXXXX"],
        "K": ["X   X", "X  X ", "XXX  ", "X  X ", "X   X"],
        "S": [" XXXX", "X    ", " XXX ", "    X", "XXXX "],
        "I": ["XXXXX", "  X  ", "  X  ", "  X  ", "XXXXX"],
        "P": ["XXXX ", "X   X", "XXXX ", "X    ", "X    "],
        "R": ["XXXX ", "X   X", "XXXX ", "X  X ", "X   X"],
        "E": ["XXXXX", "X    ", "XXXX ", "X    ", "XXXXX"],
        "C": [" XXXX", "X    ", "X    ", "X    ", " XXXX"],
        "O": [" XXX ", "X   X", "X   X", "X   X", " XXX "],
        "V": ["X   X", "X   X", "X   X", " X X ", "  X  "],
        "M": ["X   X", "XX XX", "X X X", "X   X", "X   X"],
        "N": ["X   X", "XX  X", "X X X", "X  XX", "X   X"],
        "G": [" XXXX", "X    ", "X  XX", "X   X", " XXX "],
        " ": ["     "] * 5,
    }
    scale = max(height // 9, 2)
    cols = []
    for ch in name:
        pattern = font.get(ch, font[" "])
        block = np.array([[1 if c == "X" else 0 for c in row] for row in pattern])
        cols.append(np.kron(block, np.ones((scale, scale))))
        cols.append(np.zeros((5 * scale, scale)))
    return np.concatenate(cols, axis=1) if cols else np.zeros((5 * scale, 1))


def banner(width: int, height: int, phase: str) -> np.ndarray:
    rgb, label = PHASES[phase]
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    strip[:] = rgb
    mask = glyphs(label, height)
    mh, mw = mask.shape
    y, x = (height - mh) // 2, 26
    if 0 <= y and y + mh <= height and x + mw <= width:
        region = strip[y : y + mh, x : x + mw]
        region[mask > 0] = (255, 255, 255)
    return strip


def bar(width: int, height: int, value: float, lo: float, hi: float, rgb,
        ticks: float = 0.0) -> np.ndarray:
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    strip[:] = (20, 24, 28)
    filled = int(np.clip((value - lo) / max(hi - lo, 1e-9), 0.0, 1.0) * width)
    strip[:, :filled] = rgb
    if ticks > 0:
        # Fixed marks on the slope give the eye something stationary to judge
        # travel against. Without them a slow walk on featureless snow reads as
        # standing still no matter how the camera is placed.
        t = ticks
        while t < hi:
            x = int((t - lo) / max(hi - lo, 1e-9) * width)
            if 0 <= x < width:
                strip[:, max(x - 1, 0) : x + 1] = (150, 160, 172)
            t += ticks
    return strip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/ppo_fixed_line_slope/g1_fixed_line_final.zip")
    p.add_argument("--impulse", type=float, default=700.0)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--slip-at", type=int, default=300)
    p.add_argument("--lead-in", type=int, default=150, help="Steps of walking to show first.")
    p.add_argument("--tail", type=int, default=900,
                   help="Steps to keep after recovery. The walk is slow -- roughly "
                        "680 steps per metre climbed -- so a short tail shows a "
                        "recovery but not a resumed ascent.")
    p.add_argument("--stride", type=int, default=4,
                   help="Render every Nth step. Keeps a long walk to a short clip.")
    p.add_argument("--span", type=float, default=2.6,
                   help="Metres of ascent the camera and progress bar are framed for.")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=640)
    p.add_argument("--device-lead", type=float, default=0.36,
                   help="Ascender lead along the rope. The shipped 0.36 m goes out "
                        "of arm reach under load (48.6 cm grip error at 12 kg); "
                        "0.24 restores it to 11.9 cm.")
    p.add_argument("--snow", type=float, default=1.0,
                   help="Falling-snow density. 0 disables the overlay.")
    p.add_argument("--out", default="videos/slip_recovery.mp4")
    a = p.parse_args()

    model = PPO.load(a.model, device="cpu")
    env = slip_recovery_env.load(
        disturb=True, slip_mode="impulse",
        slip_impulse_range=(a.impulse, a.impulse),
        slip_duration_range=(24, 25),
        slip_step_range=(a.slip_at, a.slip_at + 1),
        device_lead_m=a.device_lead,
    )
    obs, _ = env.reset(seed=a.seed)

    renderer = mujoco.Renderer(env.model, height=a.height, width=a.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Parked just past the middle of the run, so the robot enters low-left and
    # climbs across frame. This is the whole point of not tracking.
    # Frame for the travel, not for the robot. MuJoCo's default ~45 deg fovy
    # means the visible width is roughly 1.9x the camera distance at this
    # aspect, so distance sets what fraction of the frame the ascent crosses:
    #   9.1 m -> 2.4 m of travel is ~17% of the width, and it reads as static
    #   5.0 m -> ~32%, and the traverse is unmistakable
    # Half the span from the midpoint still leaves margin at 5 m, so the robot
    # never walks out of frame.
    camera.lookat[:] = env.uphill * (a.span * 0.5) + np.array([0.0, 0.0, 0.42])
    camera.distance = 2.6 + 0.9 * a.span
    camera.azimuth = 118
    camera.elevation = -14

    frames, phases, info = [], [], {}
    ascent_at_recovery = 0.0
    start = max(a.slip_at - a.lead_in, 0)
    stop_at = None
    for i in range(env.max_episode_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(action)
        step = i + 1
        if step >= start and step % a.stride == 0:
            renderer.update_scene(env.data, camera=camera)
            frames.append(renderer.render())
            phases.append(phase_of(info))
        if stop_at is None and info.get("recovered", 0) > 0.5:
            stop_at = step + a.tail
            ascent_at_recovery = float(info.get("ascent", 0.0))
        # `terminated` here is usually SUCCESS (target ascent held for 8
        # steps), which would cut the clip at exactly the moment the robot has
        # proved it is walking again. Keep stepping: the parent resets nothing
        # on success, so the rollout simply continues. target_ascent is left at
        # its default on purpose -- it feeds the observation
        # (fixed_line_slope_env.py:631), so raising it to postpone success
        # would push the policy out of distribution instead.
        ended = trunc or (term and bool(info.get("failure", False)))
        if ended or (stop_at is not None and step >= stop_at):
            break
    renderer.close()

    print(f"slip {info.get('slip_depth_m',0):.3f} m | "
          f"recovered={bool(info.get('recovered',0))} | "
          f"ascent {info.get('ascent',0):.3f} m | {len(frames)} frames", flush=True)
    print(f"  ascent after recovery: {info.get('ascent',0) - ascent_at_recovery:+.3f} m",
          flush=True)
    counts = {k: phases.count(k) for k in PHASES if phases.count(k)}
    print("  phase frames:", counts, flush=True)
    if "moving" not in counts:
        print("  WARNING: no confirmed-recovery frames. The clip does not show "
              "the fourth beat; lower --impulse or raise --tail.", flush=True)

    snowfall = Snowfall(a.width, a.height, seed=a.seed, intensity=a.snow) \
        if a.snow > 0 else None
    composed = []
    for frame, phase in zip(frames, phases):
        if snowfall is not None:
            snowfall.step()
            frame = snowfall.draw(frame)
        composed.append(np.concatenate([
            banner(a.width, 34, phase),
            frame,
            bar(a.width, 12, info.get("slip_depth_m", 0.0), 0.0, 1.0, (198, 58, 48)),
            bar(a.width, 16, info.get("ascent", 0.0), 0.0, a.span, (70, 130, 200),
                ticks=0.5),
        ], axis=0))
    video = np.asarray(composed)
    # yuv420p subsamples 2x2: both dimensions must be even or ffmpeg silently
    # closes the pipe and imageio reports only "Broken pipe".
    h, w = video.shape[1], video.shape[2]
    video = video[:, : h - (h % 2), : w - (w % 2)]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    # Pad to a multiple of 16, not just to even. yuv420p only requires even,
    # but hardware decoders commonly want 16, and 1280x702 refused to play in
    # QuickTime until it was padded to 1280x704. `+faststart` moves the moov
    # atom to the front so a player can start without reading the whole file.
    h, w = video.shape[1], video.shape[2]
    ph, pw = (-h) % 16, (-w) % 16
    if ph or pw:
        video = np.pad(video, ((0, 0), (0, ph), (0, pw), (0, 0)))
    import imageio.v2 as imageio
    # `profile` is not a kwarg of imageio's ffmpeg writer (valid ones are
    # fps/codec/bitrate/pixelformat/quality/macro_block_size/*_params) -- it
    # has to go through output_params as a raw flag.
    imageio.mimwrite(a.out, video, fps=25, codec="libx264", quality=8,
                     macro_block_size=1,
                     output_params=["-profile:v", "main", "-pix_fmt", "yuv420p",
                                    "-movflags", "+faststart"])
    print(f"wrote {a.out} ({video.shape[0]} frames, {video.shape[2]}x{video.shape[1]})",
          flush=True)


if __name__ == "__main__":
    main()
