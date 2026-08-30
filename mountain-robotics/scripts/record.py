"""Render a mission to an mp4 with the telemetry overlay.

    python scripts/record.py --out out/lift.mp4 --shot wide
    python scripts/record.py --mass 30 --out out/nogo.mp4     # refuses the lift
    python scripts/record.py --wind 55 --out out/gust.mp4
"""
import argparse, sys, time
sys.path.insert(0, ".")
from alpine_lift.mission import Mission, Disturbance
from alpine_lift.scene import SceneConfig
from alpine_lift.render import Recorder, SHOTS, write_video

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out/lift.mp4")
ap.add_argument("--shot", default="wide", choices=list(SHOTS))
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--fps", type=int, default=50)
ap.add_argument("--mass", type=float, default=None)
ap.add_argument("--com", type=float, default=0.0)
ap.add_argument("--payload", default=None)
ap.add_argument("--wind", type=float, default=0.0)
ap.add_argument("--ice", type=float, default=0.0)
ap.add_argument("--push", type=float, default=0.0)
ap.add_argument("--title", default="")
ap.add_argument("--no-hud", action="store_true")
ap.add_argument("--no-scenery", action="store_true")
a = ap.parse_args()

kw = dict(payload_com_offset=(0.0, a.com, 0.0), scenery=not a.no_scenery)
if a.payload: kw["payload"] = a.payload
if a.mass: kw["payload_mass"] = a.mass
scfg = SceneConfig(**kw)
dist = Disturbance(wind_gust=a.wind, ice_friction=a.ice, ice_at=7.5,
                   push_force=a.push, push_at=8.0)
mi = Mission(scene=scfg, disturbance=dist)
rec = Recorder(mi.model, a.width, a.height, a.shot, a.title)

frames = []
t0 = time.time()
for tele in mi.run():
    extra = {"go_reason": mi.ctrl.coord.go_reason,
             "sling_ok": mi.ctrl.coord.sling_ok,
             "sling_limit": scfg.sling_strength}
    frames.append(rec.frame(mi.data, tele, extra, hud=not a.no_hud))
rec.close()

r = mi.result()
print("rendered %d frames in %.1fs" % (len(frames), time.time() - t0))
print("  success=%s lift=%.3f shift=%.3f maxTilt=%.1f maxN=%.0f %s" % (
    r.success, r.lift_peak, r.shift, r.max_tilt, r.max_hand_force,
    ("ABORT: " + r.abort_reason) if r.aborted else ""))
print("wrote", write_video(a.out, frames, a.fps))
