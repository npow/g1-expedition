"""Scripted baseline vs scripted + learned residual, on identical episodes.

Same seeds, same masses, same gusts, same patch of verglas -- the only
difference is whether the policy is in the loop. Anything else is not a
comparison, it is two anecdotes.
"""
import argparse, sys, time
sys.path.insert(0, ".")
import numpy as np
from alpine_lift.env import ACT_DIM, AlpineLiftEnv
from alpine_lift.policy import load_if_present

ap = argparse.ArgumentParser()
ap.add_argument("--policy", default="out/policy.npz")
ap.add_argument("--episodes", type=int, default=40)
ap.add_argument("--seed0", type=int, default=9000)
a = ap.parse_args()

pol = load_if_present(a.policy)
if pol is None:
    print(f"no policy at {a.policy}; evaluating the baseline only")

def run(policy, seeds):
    env = AlpineLiftEnv(seed=0)
    out = []
    for s in seeds:
        o = env.reset(seed=s)
        ret = 0.0
        while True:
            act = policy(o) if policy is not None else np.zeros(ACT_DIM)
            o, r, d, _ = env.step(act)
            ret += r
            if d:
                break
        res = env.mission.result()
        out.append((res.success, res.lift_peak, res.max_tilt, res.max_hand_force,
                    ret, res.abort_reason))
    return out

seeds = [a.seed0 + i for i in range(a.episodes)]
t0 = time.time()
base = run(None, seeds)
rows = [("scripted only", base)]
if pol is not None:
    rows.append(("scripted + residual", run(pol, seeds)))

print("\n%-22s %8s %9s %9s %9s %9s" % ("controller", "success", "lift(cm)", "tilt(deg)", "peak(N)", "return"))
print("-" * 72)
for name, r in rows:
    sc = np.mean([x[0] for x in r])
    print("%-22s %7.0f%% %9.1f %9.1f %9.0f %9.1f" % (
        name, 100 * sc, 100 * np.mean([x[1] for x in r]), np.mean([x[2] for x in r]),
        np.mean([x[3] for x in r]), np.mean([x[4] for x in r])))

if len(rows) == 2:
    b = {s: r for s, r in zip(seeds, rows[0][1])}
    p = {s: r for s, r in zip(seeds, rows[1][1])}
    fixed = [s for s in seeds if not b[s][0] and p[s][0]]
    broke = [s for s in seeds if b[s][0] and not p[s][0]]
    print("\nepisodes the policy rescued: %d   episodes it lost: %d" % (len(fixed), len(broke)))
    if fixed:
        print("  rescued seeds:", fixed[:12])
    if broke:
        print("  lost seeds:   ", broke[:12])
print("\n%.0fs for %d episodes x %d controllers" % (time.time() - t0, a.episodes, len(rows)))
