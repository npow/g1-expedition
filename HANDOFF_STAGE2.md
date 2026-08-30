# Stage 2 handoff — what a PR to this repo would contain

Everything below is additive. `fixed_line_slope_env.py`, `himalaya_env.py` and
all published checkpoints are untouched, and the parent env still reproduces
its own 100%-over-ten-resets result (verified: 3/3, ascent 1.500-1.515 m).

## New files

| File | Role |
|---|---|
| `slip_recovery_env.py` | `G1SlipRecoveryEnv` — induced slip, recovery phase, and a `set_balance_assist_scale/enabled` toggle |
| `evaluate_slip_recovery.py` | Recovery rate as a function of balance-assist scale, with a no-disturbance control |
| `train_slip_recovery.py` | Fine-tune with the assist withdrawn on a performance-gated curriculum |
| `record_slip_recovery.py` | Fixed-camera recorder: phase banner, snowfall, player-compatible mp4 |
| `scripts/generate_rope_visual.py` | Kernmantle rope: UV-mapped tube mesh + procedural braid texture |
| `SLIP_RECOVERY.md` | Full findings, every failed attempt included |

## Four measurements

**1. The balance assist carries the ascent.** An uncapped orientation PD
(gain 420) plus an uncapped lateral spring (700 N/m). Halving it takes the
shipped policy from 1.512 m to 0.226 m of ascent; the scheduled disturbance
never even fires. There was no toggle for it, so it was active in every row of
the published table, controls included.

**2. The policy is a metronome.** Two checkpoints differing 25% in mean weight
produce bit-identical trajectories. Both saturate the step-request gate, after
which the gait is a fixed 23-step period (`step_duration` 18 + `step_cooldown`
3 + ~2 for the 4-frame hold). Confirmed from the other direction: a 0.02 rad
open-loop leg residual moves the robot 0.105 m, while swapping the entire
policy moves it 0.000 m. The leg channel has ample authority; the policy does
not reach it.

**3. The robot cannot be toppled.** With the model's real joint torque limits,
the orientation assist capped at 20 Nm and the lateral spring removed, a 1400 N
shove degrades uprightness only to 0.656. `_leg_targets` (line 416) re-commands
a fixed stance every control step at kp=500, so the legs cannot buckle.

**4. The published result has 18 steps of margin.** Undisturbed, ascent crosses
the 1.5 m target at step 1075 and the 8-step hold completes at 1082, against a
1100-step budget.

Findings 1-3 are one cause: the policy commands 2 of 43 actuators.

## Two defects worth fixing regardless

- **Training is broken from a clean install.** `train.py:272` and
  `train_fixed_line.py:132` pass `progress_bar=True`; `requirements.txt`
  declares neither `tqdm` nor `rich`, so the documented reproduce path dies on
  ImportError before the first step.
- **`mediapy.write_video` needs a system ffmpeg** that `requirements.txt` never
  installs. `imageio.mimwrite(..., codec="libx264")` uses `imageio_ffmpeg`'s
  bundled binary and needs nothing external.

## Corrections made during this work

Recorded because they were wrong in intermediate states of this branch.

- Claimed the actuators had unlimited torque, from `actuator_forcerange` being
  `[0,0]`. MuJoCo enforces those per joint via `actuatorfrcrange`, and this
  model sets them correctly on 43 of 44 joints with the real G1 values. The
  claim was withdrawn; no conclusion depended on it.
- Two fine-tuning runs (fixed anneal, then performance-gated) were attempted
  before establishing that the action space cannot express the target
  behaviour. The first destroyed the gait (0.003 m ascent on the undisturbed
  env); the second preserved it but changed nothing measurable. Both were
  avoidable — checking whether a channel has authority costs 2 minutes, versus
  ~25 minutes per training run.
- A recovery metric that credited a momentary regain of footing. Under an
  800 N shove the robot cleared it 16 steps after the slip, then collapsed;
  that scored as a recovery. Recovery now requires holding stance 60 further
  steps and adding 10 cm.

## Not done, and why

Fall-and-recover cannot be demonstrated in this environment at any parameter
setting. It needs compliant or torque-controlled legs, a whole-body action
space, and a separate get-up policy — a new environment, not a change here.

## `.gitignore` is a whitelist

`/*` ignores everything at root and each file is re-admitted with `!`. Every
new module in this branch was therefore invisible to git until listed, which
would have produced a PR that silently omitted all of it. The entries are
added; keep the pattern in mind when adding anything else.

## Before merging

This checkout fetched LFS objects without `git-lfs` installed, so `git status`
reports ~21 model zips as modified. **Do not `git add -A`.** Add explicitly:

```bash
git add slip_recovery_env.py evaluate_slip_recovery.py train_slip_recovery.py \
        record_slip_recovery.py scripts/generate_rope_visual.py \
        assets/unitree_g1/scene_fixed_line_slope.xml \
        assets/unitree_g1/assets/rope_kernmantle.png \
        assets/unitree_g1/assets/rope_tube.obj \
        SLIP_RECOVERY.md HANDOFF_STAGE2.md
```
