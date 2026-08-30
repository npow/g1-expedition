"""Interactive viewer -- this is the one to run on stage.

macOS needs ``mjpython`` rather than ``python`` here: MuJoCo's passive viewer
has to own the main thread for Cocoa, and plain python exits with a thread
error. Everything else in this repo runs under normal python.

    mjpython scripts/live.py                    # nominal lift
    mjpython scripts/live.py --mass 30          # team weighs it and declines
    mjpython scripts/live.py --wind 45          # gust during the carry
    mjpython scripts/live.py --ice 0.3          # verglas underfoot
    mjpython scripts/live.py --policy out/policy.npz --wind 45

Runs at wall-clock speed with a live telemetry line in the terminal, and
loops so you can talk over it. Press Escape or close the window to stop.
"""
import argparse, sys, time
sys.path.insert(0, ".")
import mujoco
import mujoco.viewer

from alpine_lift.mission import Mission, Disturbance
from alpine_lift.policy import load_if_present
from alpine_lift.scene import SceneConfig

ap = argparse.ArgumentParser()
ap.add_argument("--mass", type=float, default=None)
ap.add_argument("--com", type=float, default=0.0)
ap.add_argument("--payload", default=None)
ap.add_argument("--wind", type=float, default=0.0)
ap.add_argument("--ice", type=float, default=0.0)
ap.add_argument("--push", type=float, default=0.0)
ap.add_argument("--policy", default=None)
ap.add_argument("--speed", type=float, default=1.0)
ap.add_argument("--once", action="store_true")
a = ap.parse_args()

kw = dict(payload_com_offset=(0.0, a.com, 0.0))
if a.payload: kw["payload"] = a.payload
if a.mass: kw["payload_mass"] = a.mass
scfg = SceneConfig(**kw)
dist = Disturbance(wind_gust=a.wind, ice_friction=a.ice, ice_at=7.5,
                   push_force=a.push, push_at=8.0)
mi = Mission(scene=scfg, disturbance=dist)
pol = load_if_present(a.policy)
print(f"policy: {a.policy if pol else 'none (scripted baseline)'}")

need_obs = pol is not None
env = None
if need_obs:
    from alpine_lift.env import AlpineLiftEnv
    env = AlpineLiftEnv(randomize=False, seed=0)
    env.reset()
    env._mi = mi   # drive the same mission we are rendering

with mujoco.viewer.launch_passive(mi.model, mi.data,
                                  show_left_ui=False, show_right_ui=False) as v:
    v.cam.azimuth, v.cam.elevation, v.cam.distance = 132.0, -11.0, 3.5
    v.cam.lookat[:] = [0.0, 0.0, 0.52]
    # Hide collision primitives and the sling-loop sites. The passive
    # viewer's Handle exposes cam/opt/user_scn but no render-flag scene, so
    # fog and haze are left at the viewer's own defaults here; record.py sets
    # them explicitly for the offscreen renders.
    v.opt.geomgroup[3] = 0
    v.opt.geomgroup[4] = 0

    while v.is_running():
        mi.reset()
        if env is not None:
            env._mi = mi
        last = time.time()
        line = 0
        while v.is_running() and not mi.ctrl.done:
            act = None
            if pol is not None:
                act = pol(env._obs(mi.ctrl.tele))
            t = mi.step(residual=act)
            v.sync()
            line += 1
            if line % 10 == 0:
                gr = mi.ctrl.coord.go_reason if t.gripped else ""
                sys.stdout.write(
                    f"\r{t.t:6.2f}s {t.phase:<12} lift {t.payload_lift * 100:+6.1f}cm  "
                    f"tilt {t.tilt_deg:+6.2f}d  share {t.share * 100:3.0f}/{100 - t.share * 100:<3.0f}  "
                    f"{'GO ' if t.go else 'NOGO'}  {gr[:38]:<38}")
                sys.stdout.flush()
            dt = mi.ctrl.n_sub * mi.model.opt.timestep / max(a.speed, 1e-3)
            sleep = last + dt - time.time()
            if sleep > 0:
                time.sleep(sleep)
            last = time.time()
        r = mi.result()
        print(f"\n--> {'SUCCESS' if r.success else 'FAILED'}  "
              f"lift {r.lift_peak * 100:.0f}cm  shift {r.shift * 100:.0f}cm  "
              f"maxTilt {r.max_tilt:.1f}deg  peak {r.max_hand_force:.0f}N"
              + (f"  [{r.abort_reason}]" if r.aborted else ""))
        if a.once:
            break
        time.sleep(1.5)
