"""Draw the fixed line as kernmantle climbing rope.

The scene ships the handline as one smooth 26 mm capsule, which reads as steel
cable. A first attempt replaced it with three helical strands -- but that is
hawser-laid rope, the marine/utility construction. Nobody fixes a Himalayan
route with laid rope; fixed lines are kernmantle: a load-bearing core inside a
tightly braided sheath, typically 9-11 mm.

Why this is a texture and not geometry
--------------------------------------
A real sheath braid repeats roughly every 25 mm. Over an 11.4 m line that is
~450 repeats, and drawing it as capsules needs tens of thousands of geoms. The
laid-rope attempt already cost 1,680 capsules for a much coarser pattern. A
procedural texture gives a finer, more accurate braid on a SINGLE geom.

Why a mesh and not a capsule
----------------------------
MuJoCo projects textures onto primitives by axis direction; it does not
UV-wrap them. A braid texture on a capsule therefore renders as banding along
the rope, never as a weave around it -- verified against `2d`/`cube` and
`texuniform` true/false. So the rope is emitted as a tube MESH carrying real
texture coordinates, which is the only way the sheath wraps correctly.

The UV scale matters as much as the texture. One tile is mapped to a square
patch of rope surface -- tile length along the axis equals the circumference --
so the braid's diagonals meet the axis at the angle they were drawn at. Map a
tile to a long thin patch instead and the weave shears until it reads as
axial stripes.

Safe by construction: the rope geom is `contype="0" conaffinity="0"` and the
ascender's position comes from `_rope_point()` in Python, not from the geom.

Diameter is corrected to 11 mm, which is a realistic fixed line. The shipped
26 mm is nearer to a ship's hawser.

    python scripts/generate_rope_visual.py
"""

from __future__ import annotations

import pathlib
import re

import mujoco
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENE = ROOT / "assets/unitree_g1/scene_fixed_line_slope.xml"
TEXTURE = ROOT / "assets/unitree_g1/assets/rope_kernmantle.png"
MESH = ROOT / "assets/unitree_g1/assets/rope_tube.obj"
RING = 20            # vertices around the circumference
SPANS = 48           # rings ALONG the span, for the catenary
# Sag at midspan as a fraction of span. A tensioned rope still sags;
# drawing it dead straight is the single biggest reason it reads as cable.
SAG_FRACTION = 0.0   # see SLIP_RECOVERY.md: sag desynced the ascender

BEGIN = "<!-- BEGIN generated rope (scripts/generate_rope_visual.py) -->"
END = "<!-- END generated rope -->"

# 14 mm. A real fixed line is 9-11 mm, but at the demo camera's 5 m that is
# 2-3 px and reads as a hairline. 14 mm stays in range for a heavy static
# line and is legible on video.
ROPE_RADIUS = 0.0070
SHEATH_STRANDS = 8            # per handedness
TEX = 512                     # texture is TEX x TEX, tiles seamlessly
# Sheath colours: a used fixed line is not showroom orange. Two tones plus
# grime, because a uniformly bright rope is the other way to look synthetic.
WARP = np.array([1.00, 0.50, 0.12])
WEFT = np.array([1.00, 0.90, 0.38])
GRIME = np.array([0.46, 0.40, 0.34])


def braid_texture() -> Image.Image:
    """Seamless over-under diamond braid, the way a sheath actually looks."""
    u = np.linspace(0.0, 1.0, TEX, endpoint=False)[None, :]
    v = np.linspace(0.0, 1.0, TEX, endpoint=False)[:, None]

    # Two counter-rotating strand families. Integer coefficients keep both
    # seamless across the tile edges.
    a = (u * SHEATH_STRANDS + v * SHEATH_STRANDS)
    b = (u * SHEATH_STRANDS - v * SHEATH_STRANDS)
    # `% 1.0`, NOT np.modf: modf keeps the sign of its input, and b goes
    # negative wherever v exceeds u. A negative fractional part then hits
    # `sin(pi*f) ** 0.55` as a negative base under a fractional exponent, which
    # is NaN, and the NaNs cast to black speckle in the saved PNG.
    fa, fb = a % 1.0, b % 1.0

    # Over-under: which family sits on top alternates cell by cell, which is
    # what makes a braid read as woven rather than as crosshatch.
    over_a = (np.floor(a) + np.floor(b)).astype(int) % 2 == 0

    # Round each strand across its width so it reads as a cord, not a stripe.
    round_a = np.sin(np.pi * fa) ** 0.55
    round_b = np.sin(np.pi * fb) ** 0.55

    shade = np.where(over_a, round_a, round_b * 0.72)
    colour = np.where(over_a[..., None], WARP[None, None, :], WEFT[None, None, :])

    # Fibre noise, then a little grime in the valleys where dirt collects.
    rng = np.random.default_rng(7)
    fibre = 1.0 + 0.10 * rng.standard_normal((TEX, TEX, 1))
    # Mean luminance matters as much as the pattern. At the demo's 5 m camera
    # an 11 mm rope is 2-3 px wide, so the mip average IS the rope colour --
    # deep valleys and heavy grime made it average to a dark line rather than
    # an orange one. Shallower shading and lighter grime keep the braid legible
    # close up while reading as rope at distance.
    # Shallow shading on purpose. At 5 m a 14 mm rope is a few pixels and the
# MIP AVERAGE is the rope's colour; deep valleys averaged it to a dark
# hairline even though the braid was correct in close-up.
    img = colour * (0.88 + 0.22 * shade[..., None]) * fibre
    valley = np.clip(1.0 - shade, 0.0, 1.0)[..., None]
    img = img * (1.0 - 0.05 * valley) + GRIME[None, None, :] * 0.05 * valley

    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def tube_obj(length: float, radius: float) -> str:
    """Tube along LOCAL +Z, centred, sagging in LOCAL +X, with texcoords.

    The sag is a parabola -- the standard small-sag approximation to a catenary,
    indistinguishable at this span. Canonical frame on purpose: MuJoCo
    re-centres every mesh and rotates it onto its principal axes, so a tube
    authored in world coordinates comes back translated and re-aimed.
    """
    circumference = 2.0 * np.pi * radius
    repeats = length / circumference
    half = length / 2.0
    sag = SAG_FRACTION * length

    lines = ["# generated by scripts/generate_rope_visual.py"]
    for si in range(SPANS + 1):
        t = si / SPANS
        z = -half + t * length
        drop = sag * 4.0 * t * (1.0 - t)   # zero at anchors, max at midspan
        for k in range(RING + 1):
            ang = 2.0 * np.pi * k / RING
            lines.append(
                f"v {radius*np.cos(ang)+drop:.6f} {radius*np.sin(ang):.6f} {z:.6f}"
            )
    for si in range(SPANS + 1):
        for k in range(RING + 1):
            lines.append(f"vt {si/SPANS*repeats:.5f} {k/RING:.5f}")
    for k in range(RING + 1):
        ang = 2.0 * np.pi * k / RING
        lines.append(f"vn {np.cos(ang):.5f} {np.sin(ang):.5f} 0.00000")

    cols = RING + 1
    for si in range(SPANS):
        for k in range(RING):
            a0 = 1 + si * cols + k
            b0 = a0 + 1
            c0 = 1 + (si + 1) * cols + k
            d0 = c0 + 1
            na, nb = 1 + k, 1 + k + 1
            lines.append(f"f {a0}/{a0}/{na} {c0}/{c0}/{na} {d0}/{d0}/{nb}")
            lines.append(f"f {a0}/{a0}/{na} {d0}/{d0}/{nb} {b0}/{b0}/{nb}")
    return "\n".join(lines) + "\n"


def main() -> None:
    TEXTURE.parent.mkdir(parents=True, exist_ok=True)
    braid_texture().save(TEXTURE)

    xml = SCENE.read_text()
    xml = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), "", xml, flags=re.S)

    match = re.search(r'[ \t]*<geom name="fixed_rope".*?/>', xml, flags=re.S)
    if match is None:
        raise SystemExit(
            "fixed_rope geom not found -- restore "
            "assets/unitree_g1/scene_fixed_line_slope.xml first."
        )
    original = match.group(0)
    coords = [float(x) for x in re.search(r'fromto="([^"]+)"', original).group(1).split()]
    start, end = np.array(coords[:3]), np.array(coords[3:])
    length = float(np.linalg.norm(end - start))

    MESH.write_text(tube_obj(length, ROPE_RADIUS))
    # texrepeat stays 1 1 -- the mesh's own UVs carry the tiling.
    assets = (
        f'    {BEGIN}\n'
        f'    <texture type="2d" name="rope_kernmantle" file="rope_kernmantle.png"/>\n'
        f'    <material name="rope_kernmantle_mat" texture="rope_kernmantle" '
        f'texuniform="false" texrepeat="1 1" '
        f'specular="0.06" shininess="0.03" reflectance="0.0"/>\n'
        f'    <mesh name="rope_tube" file="rope_tube.obj"/>\n'
        f'    {END}'
    )
    anchor = '<material name="rope_material"'
    idx = xml.index(anchor)
    xml = xml[:idx] + assets.strip() + "\n    " + xml[idx:]

    mid = (start + end) / 2.0
    quat = np.zeros(4)
    axis = (end - start) / length
    # Local +X must follow gravity's component perpendicular to the rope, so the
    # sag hangs downward rather than sideways.
    down = np.array([0.0, 0.0, -1.0])
    x_hat = down - np.dot(down, axis) * axis
    x_hat /= np.linalg.norm(x_hat)
    rot = np.column_stack([x_hat, np.cross(axis, x_hat), axis])
    mujoco.mju_mat2Quat(quat, rot.flatten())
    rope = (
        f'<geom name="fixed_rope" type="mesh" mesh="rope_tube"\n'
        f'      pos="{mid[0]:.4f} {mid[1]:.4f} {mid[2]:.4f}"\n'
        f'      quat="{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}"\n'
        f'      material="rope_kernmantle_mat" '
        f'contype="0" conaffinity="0" group="2"/>'
    )
    xml = xml.replace(original, "\n    " + rope)
    SCENE.write_text(xml)

    print(f"kernmantle rope: {ROPE_RADIUS*2000:.0f} mm diameter, "
          f"{SHEATH_STRANDS} sheath strands per handedness")
    circ = 2 * np.pi * ROPE_RADIUS
    print(f"  tube mesh: {RING} sides, UV tiles {length/circ:.0f}x along "
          f"({circ*1000:.0f} mm per braid repeat, isotropic)")
    print(f"  texture {TEX}x{TEX} -> {TEXTURE.name}")
    print(f"  1 geom (was 1680 capsules for the laid-rope attempt)")


if __name__ == "__main__":
    main()
