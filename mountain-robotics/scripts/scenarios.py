"""Run every demo scenario headless and report. Use this before pitching.

Each row is a claim made in the README or on stage; this is what checks that
the claim is still true after a change.
"""
import sys, time
sys.path.insert(0, ".")
from alpine_lift.mission import Mission, Disturbance
from alpine_lift.scene import SceneConfig

SCENARIOS = [
    ("nominal lift",        dict(), dict(), "completes"),
    ("heavy - no go",       dict(payload_mass=30.0), dict(), "declines"),
    ("off-centre CoM",      dict(payload_com_offset=(0.0, 0.14, 0.0)), dict(), "completes"),
    ("gust 20N",            dict(), dict(wind_gust=20.0), "either"),
    ("gust 45N",            dict(), dict(wind_gust=45.0), "either"),
    ("verglas mu=0.45",     dict(), dict(ice_friction=0.45, ice_at=7.0), "either"),
    ("verglas mu=0.30",     dict(), dict(ice_friction=0.30, ice_at=7.0), "either"),
    ("push 40N on A",       dict(), dict(push_force=40.0, push_at=8.0), "either"),
    ("push 70N on A",       dict(), dict(push_force=70.0, push_at=8.0), "either"),
    ("light log 8kg",       dict(payload_mass=8.0), dict(), "completes"),
    ("boulder payload",     dict(payload="boulder", payload_mass=12.0,
                                 payload_half=(0.16, 0.44, 0.16)), dict(), "either"),
]

print("%-22s %8s %8s %8s %9s  %s" % (
    "scenario", "lift(cm)", "tilt", "peak(N)", "mass_est", "outcome"))
print("-" * 88)
t0 = time.time()
for name, skw, dkw, expect in SCENARIOS:
    mi = Mission(scene=SceneConfig(scenery=False, **skw), disturbance=Disturbance(**dkw))
    est = 0.0
    for tele in mi.run():
        # The number that matters is the one the weigh-in decided on, not
        # whatever the filter reads after the load has been set back down.
        if mi.ctrl.coord.decided and est == 0.0:
            est = tele.mass_est
    r = mi.result()
    if r.aborted and "no-go" in (r.abort_reason or ""):
        outcome = "DECLINED - " + mi.ctrl.coord.go_reason
    elif r.aborted:
        outcome = "aborted: " + r.abort_reason
    elif r.success:
        outcome = "completed"
    else:
        outcome = "finished, goal not met"
    print("%-22s %8.1f %8.1f %8.0f %9.1f  %s" % (
        name, 100 * r.lift_peak, r.max_tilt, r.max_hand_force, est, outcome))
print("-" * 88)
print("%d scenarios in %.0fs" % (len(SCENARIOS), time.time() - t0))
