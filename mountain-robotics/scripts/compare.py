"""Side-by-side video: scripted baseline vs scripted + learned residual.

Both panels run the same seed, the same log, the same gust, the same patch
of verglas. The only difference between them is whether the policy is in the
loop, which is the only way the comparison means anything.

    python scripts/compare.py --policy out/policy.npz --wind 45
"""
import argparse, sys
sys.path.insert(0, ".")
import numpy as np
from PIL import Image

from alpine_lift.env import AlpineLiftEnv
from alpine_lift.mission import Mission, Disturbance
from alpine_lift.policy import load_if_present
from alpine_lift.render import Recorder, write_video
from alpine_lift.scene import SceneConfig

ap = argparse.ArgumentParser()
ap.add_argument("--policy", default="out/policy.npz")
ap.add_argument("--out", default="out/compare.mp4")
ap.add_argument("--wind", type=float, default=45.0)
ap.add_argument("--ice", type=float, default=0.0)
ap.add_argument("--push", type=float, default=0.0)
ap.add_argument("--mass", type=float, default=11.0)
ap.add_argument("--com", type=float, default=0.0)
ap.add_argument("--width", type=int, default=760)
ap.add_argument("--height", type=int, default=640)
a = ap.parse_args()

pol = load_if_present(a.policy)
if pol is None:
    sys.exit(f"no policy at {a.policy} -- train one first")


def build():
    scfg = SceneConfig(payload_mass=a.mass, payload_com_offset=(0.0, a.com, 0.0))
    dist = Disturbance(wind_gust=a.wind, ice_friction=a.ice, ice_at=7.5,
                       push_force=a.push, push_at=8.0, seed=11)
    return Mission(scene=scfg, disturbance=dist)


runs = []
for use_policy in (False, True):
    mi = build()
    env = None
    if use_policy:
        env = AlpineLiftEnv(randomize=False, seed=0)
        env.reset()
        env._mi = mi
    rec = Recorder(mi.model, a.width, a.height, "wide",
                   "SCRIPTED + LEARNED RESIDUAL" if use_policy else "SCRIPTED ONLY")
    frames = []
    while not mi.ctrl.done:
        act = pol(env._obs(mi.ctrl.tele)) if use_policy else None
        t = mi.step(residual=act)
        frames.append(rec.frame(mi.data, t,
                                {"go_reason": mi.ctrl.coord.go_reason,
                                 "sling_ok": mi.ctrl.coord.sling_ok,
                                 "sling_limit": mi.scfg.sling_strength}))
    rec.close()
    runs.append((frames, mi.result()))
    print(("residual" if use_policy else "scripted "),
          "->", "SUCCESS" if runs[-1][1].success else
          f"FAILED ({runs[-1][1].abort_reason or 'goal not met'})", flush=True)

left, right = runs[0][0], runs[1][0]
n = max(len(left), len(right))
out = []
for i in range(n):
    l = left[min(i, len(left) - 1)]
    r = right[min(i, len(right) - 1)]
    canvas = Image.new("RGB", (a.width * 2 + 6, a.height), (10, 13, 19))
    canvas.paste(Image.fromarray(l), (0, 0))
    canvas.paste(Image.fromarray(r), (a.width + 6, 0))
    out.append(np.asarray(canvas))

print("wrote", write_video(a.out, out, 50))
