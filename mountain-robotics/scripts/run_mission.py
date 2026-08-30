"""Headless mission run with a text trace -- the fastest way to see what happened."""
import argparse, sys
sys.path.insert(0, ".")
from alpine_lift.mission import Mission, Disturbance
from alpine_lift.scene import SceneConfig

ap = argparse.ArgumentParser()
ap.add_argument("--mass", type=float, default=None)
ap.add_argument("--radius", type=float, default=None)
ap.add_argument("--roll", type=float, default=None, help="rolling resistance")
ap.add_argument("--wind", type=float, default=0.0)
ap.add_argument("--ice", type=float, default=0.0)
ap.add_argument("--push", type=float, default=0.0)
ap.add_argument("--every", type=float, default=0.5)
a = ap.parse_args()

kw = dict(scenery=False)
if a.mass: kw["log_mass"] = a.mass
if a.radius: kw["log_radius"] = a.radius
if a.roll is not None: kw["log_roll_friction"] = a.roll
mi = Mission(scene=SceneConfig(**kw),
             disturbance=Disturbance(wind_gust=a.wind, ice_friction=a.ice,
                                     ice_at=6.0, push_force=a.push))

print("%-6s %-10s %7s %7s %6s %7s %7s %6s  %s" % (
    "t", "phase", "travel", "toEdge", "skew", "push", "speed", "pace", "note"))
last = -9.0
for tele in mi.run():
    if tele.t - last >= a.every or tele.phase != getattr(mi, "_lp", None):
        last = tele.t; mi._lp = tele.phase
        print("%-6.2f %-10s %7.3f %7.3f %6.1f %7.1f %7.2f %6.2f  %s" % (
            tele.t, tele.phase, tele.log_travel, tele.to_edge, tele.skew_deg,
            tele.push_total, tele.speed, tele.pace,
            mi.ctrl.coord.go_reason if tele.engaged else ""))
r = mi.result()
print("\n=== RESULT ===")
for k, v in r.__dict__.items():
    if k != "events":
        print(f"  {k:16s} {v}")
print("  events:")
for t, e in r.events:
    print(f"     {t:6.2f}s  {e}")
