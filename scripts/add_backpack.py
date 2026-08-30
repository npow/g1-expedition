"""Put a real load on the robot's back.

A climber on a fixed line carries a pack, and that pack is not decoration: on a
44 kg robot a 12 kg load is a 27% mass increase carried high and behind the
spine, which shifts the centre of mass aft and raises the pitching inertia.
Adding it as a visual-only geom would look right and test nothing.

So this attaches a `haul_pack` body to `torso_link` with real mass and inertia.
It adds **no degrees of freedom** -- a body with no joint is welded into the
kinematic tree -- so nq/nv/nu/neq are unchanged and everything downstream still
composes.

Contact is deliberately OFF (`contype=0 conaffinity=0`). The point is to
isolate the effect of MASS: a colliding pack would also catch the rope and the
arms, and any change in behaviour could then be either the load or a new
contact. Mass alone is the clean experiment. Turn it on with --collide if you
want the messier, more realistic version.

Inertia is computed as a solid box of the given mass rather than left to
MuJoCo's default, so the pitching term is right.

    python scripts/add_backpack.py --mass 12
    python scripts/add_backpack.py --remove
"""

from __future__ import annotations

import argparse
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
ROBOT = ROOT / "assets/unitree_g1/g1_with_hands.xml"

BEGIN = "<!-- BEGIN generated haul pack (scripts/add_backpack.py) -->"
END = "<!-- END generated haul pack -->"

# Half-extents (m): a ~45 litre alpine pack. Taller than wide, hugging the
# spine, rather than the wide crate a naive box gives you.
HALF = (0.098, 0.135, 0.205)
# Behind the spine, centred between shoulder and waist. The first placement
# (z=0.215) rode high enough to swallow the head from behind.
POS = (-0.135, 0.0, 0.105)


def block(mass: float, collide: bool) -> str:
    hx, hy, hz = HALF
    # Solid-box inertia about the body centre.
    ixx = mass * (hy * hy + hz * hz) / 3.0
    iyy = mass * (hx * hx + hz * hz) / 3.0
    izz = mass * (hx * hx + hy * hy) / 3.0
    con = "" if collide else ' contype="0" conaffinity="0"'
    return (
        f'            {BEGIN}\n'
        f'            <body name="haul_pack" pos="{POS[0]} {POS[1]} {POS[2]}">\n'
        f'              <inertial pos="0 0 0" mass="{mass:g}" '
        f'diaginertia="{ixx:.6f} {iyy:.6f} {izz:.6f}"/>\n'
        f'              <geom name="haul_pack_body" type="box" '
        f'size="{hx} {hy} {hz}" rgba="0.16 0.19 0.24 1"{con} group="2"/>\n'
        f'              <geom name="haul_pack_lid" type="box" pos="{-hx*0.15:.3f} 0 {hz*0.99:.3f}" '
        f'size="{hx*0.80:.3f} {hy*0.86:.3f} {hz*0.13:.3f}" '
        f'rgba="0.62 0.24 0.10 1" contype="0" conaffinity="0" group="2"/>\n'
        f'              <geom name="haul_pack_strap_l" type="capsule" '
        f'fromto="{hx*0.6:.3f} -0.085 {hz*0.55:.3f} {hx*0.9:.3f} -0.075 {-hz*0.55:.3f}" '
        f'size="0.016" rgba="0.10 0.11 0.13 1" contype="0" conaffinity="0" group="2"/>\n'
        f'              <geom name="haul_pack_strap_r" type="capsule" '
        f'fromto="{hx*0.6:.3f} 0.085 {hz*0.55:.3f} {hx*0.9:.3f} 0.075 {-hz*0.55:.3f}" '
        f'size="0.016" rgba="0.10 0.11 0.13 1" contype="0" conaffinity="0" group="2"/>\n'
        f'            </body>\n'
        f'            {END}'
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mass", type=float, default=12.0, help="Pack mass in kg.")
    p.add_argument("--collide", action="store_true",
                   help="Let the pack collide. Off by default so the experiment "
                        "isolates mass rather than mixing in new contacts.")
    p.add_argument("--remove", action="store_true")
    a = p.parse_args()

    xml = ROBOT.read_text()
    xml = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?", "", xml, flags=re.S)

    if not a.remove:
        anchor = '            <geom class="collision" mesh="torso_link"/>\n'
        if anchor not in xml:
            raise SystemExit("torso_link collision geom not found -- XML changed?")
        xml = xml.replace(anchor, anchor + block(a.mass, a.collide) + "\n", 1)
    ROBOT.write_text(xml)

    if a.remove:
        print(f"pack removed from {ROBOT.name}")
    else:
        print(f"haul pack: {a.mass:g} kg on torso_link at {POS}, "
              f"{'colliding' if a.collide else 'mass-only (no contact)'}")


if __name__ == "__main__":
    main()
