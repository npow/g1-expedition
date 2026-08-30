# Stage 2 — Induced Slip and Recovery on the Fixed Line

Added on top of `855cef6`. Everything here is additive: `fixed_line_slope_env.py`
is untouched and still reproduces its published 100%-over-ten-resets result.

`FIXED_LINE.md` lists "recovery from a large fall" as out of scope. This stage
adds the disturbance that causes one, a phase that lets the episode continue
past it, and the measurement of whether the robot gets going again.

## Status

Environment, evaluation harness, trainer and recorder are built and exercised
against the shipped checkpoint. The headline finding is negative and it is the
reason the trainer exists: **at full assist the recovery is performed by the
external stabilizer, not by the policy.**

An assist-annealing fine-tune is running at the time of writing (1,000,000
total / 740,104 new steps, ~610 fps on 8 CPU workers, ~20 min). Its numbers are
NOT in this document.

## Files added

| File | Role |
|---|---|
| `slip_recovery_env.py` | `G1SlipRecoveryEnv` — disturbance, recovery phase, and the `set_balance_assist_scale/enabled` toggle the parent never had |
| `evaluate_slip_recovery.py` | Ablation harness: recovery rate as a function of assist scale |
| `train_slip_recovery.py` | Fine-tune from the shipped checkpoint with the assist annealed out |
| `record_slip_recovery.py` | Fixed-camera renderer with phase banner and snowfall overlay |

## The finding

Shipped checkpoint, friction-mode slip, 8 episodes per row:

| Condition | Success | Recovery | Mean slip | Mean ascent |
|---|---:|---:|---:|---:|
| parent env, no disturbance | 100% | n/a | 0.000 m | 1.512 m |
| slip, balance assist x1 | 100% | 100% | 0.008 m | 1.517 m |
| slip, balance assist x0.5 | 0% | n/a | 0.000 m | 0.226 m |
| slip, balance assist x0.25 | 0% | n/a | 0.000 m | 0.071 m |
| slip, balance assist x0 | 0% | n/a | 0.000 m | 0.029 m |

Read the `n/a` rows carefully: mean slip is 0.000 m because **the slip never
fires**. It is scheduled for step 140-420 and at half assist the robot has
already collapsed. Halving the stabilizer costs 85% of the ascent.

The assist is an uncapped orientation PD (gain 420) holding the pelvis at a
target quaternion, plus an uncapped lateral spring (700 N/m) pinning it to
y=0. Only the normal term is capped, at 100 N. `HANDOFF.md`'s "capped
low-level balance assist" refers to that one term.

Fairness: the policy trained at x1.0, so x0.5 is out of distribution. This does
not show the policy learned nothing — it shows policy and assist are
co-adapted, and that no recovery number measured at x1.0 is attributable to the
policy. Hence the anneal.

## Dose-response (impulse mode, assist x1, seed 5)

| Impulse | Slip depth | Recovered | Outcome |
|---:|---:|---|---|
| 200 N | 0.024 m | yes | success, 1.521 m |
| 400 N | 0.038 m | yes | success, 1.512 m |
| 700 N | 0.350 m | yes | success, 1.520 m |
| 800 N | 0.658 m | **no** | regained footing 16 steps, then collapsed; 1.031 m |
| 900 N | 0.985 m | no | slid off, -0.571 m |
| 1100 N | 1.022 m | no | slid off, -0.596 m |

At full assist this policy recovers from a 0.35 m slip and not from 0.66 m.

## Design decisions

**Action space stays 2-D.** `obs_dim` is a function of `action_dim`, so
widening it changes the observation width and the shipped checkpoint can no
longer load. Keeping it at 2 made the shipped policy testable with zero
training, which is how the assist finding surfaced in an afternoon. Widening
(leg-target residuals, or a third "recover" request) is the follow-on.

**`target_ascent` is left alone.** It feeds the observation
(`fixed_line_slope_env.py:631`), so raising it to postpone success would push
the policy out of distribution. The recorder instead keeps stepping past
success termination.

**Recovery is confirmed, not provisional.** Regaining double support plus 6 cm
is only a candidate; confirmation needs 60 further steps and another 10 cm.
Without that, the 800 N case above was credited as a recovery for something the
robot did not do.

**Horizon raised to 1500.** See "18 steps of margin" below.

## Bugs found in this repo

1. **Training is broken from a clean install.** `train.py:272` and
   `train_fixed_line.py:132` pass `progress_bar=True`; `requirements.txt`
   declares neither `tqdm` nor `rich`. The documented reproduce path dies on
   ImportError before the first step. `train_slip_recovery.py` degrades
   gracefully instead.
2. **The published ascent result has 18 steps of margin.** Undisturbed, ascent
   crosses the 1.5 m target at step 1075 and the 8-step hold completes at 1082,
   against a 1100-step budget. A disturbed run crossed at 1095 and truncated
   with 6 of 8 steps banked — scored a failure purely on the clock while
   climbing 1.521 m and recovering cleanly.
3. **The balance assist has no ablation toggle.** `set_line_enabled`,
   `set_traction_enabled`, `set_foot_ascender_enabled` exist; the assist is on
   in every row of the published table, including the neutral- and
   random-action controls.
4. **`render()` uses `mjCAMERA_TRACKING` on the pelvis**, so translation is
   invisible — the frame climbs with the robot. npow already solved this for
   the self-arrest suite with a fixed camera; the same fix applies here.

## Bugs found in this stage, by testing

Recorded because each one would have produced a wrong published number.

1. **The airborne gate made success unreachable.** The parent disqualifies an
   episode if the both-feet-off streak ever exceeds 3, as a whole-episode
   maximum. A 700 N slip hits 14. A robot that slipped 0.35 m, recovered, and
   climbed 2.66 m scored as a failure on that gate alone. Fixed by rewinding to
   the pre-slip maximum on confirmed recovery; instability after recovery still
   counts.
2. **Transient recovery was being credited.** See the 800 N row.
3. **The curriculum ran on the wrong clock.** On resume `num_timesteps` starts
   at 259,896, so progress measured against the absolute target started the run
   65% into the anneal. It now runs over the new span.
4. **Odd video width.** 1280 + 5 + 1280 = 2565, and `yuv420p` subsamples 2x2,
   so ffmpeg closed the pipe and imageio reported only "Broken pipe".
   Dimensions are trimmed to even.
5. **The clip was cut at the moment of recovery**, because `terminated` usually
   means success.

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# ablation table
.venv/bin/python evaluate_slip_recovery.py --episodes 8

# fine-tune with the assist annealed out
.venv/bin/python train_slip_recovery.py --timesteps 1000000 --num-envs 8 \
    --assist-final 0.25 --slip-mode impulse

# walk -> slip -> recover -> move, with snowfall
MUJOCO_GL=glfw .venv/bin/python record_slip_recovery.py \
    --impulse 700 --span 2.6 --tail 900 --snow 1.0
```

`MUJOCO_GL=egl` as documented elsewhere in this repo does not work on macOS;
`glfw` is the working backend. `mediapy.write_video` needs a system `ffmpeg`
that `requirements.txt` never installs — `imageio.mimwrite(..., codec="libx264")`
uses `imageio_ffmpeg`'s bundled binary and needs nothing.

## Snowfall

Visual only, composited at render time, never in the model. Real particle geoms
would add hundreds of bodies, push past `njmax`, and risk contacts with the
robot — altering the physics of the rollout being filmed. Three parallax layers
with per-flake sway phase; alpha-blended with a per-layer colour, because
additive white vanished over the snowfield.

## Load carriage

`scripts/add_backpack.py` attaches a `haul_pack` body to `torso_link` with real
mass and inertia -- a climber on a fixed line carries a pack, and a visual-only
geom would look right and test nothing. It adds **no degrees of freedom** (a
body with no joint welds into the tree), so nq 50 / nv 49 / nu 43 / neq 0 are
unchanged and everything downstream still composes. Contact is off by default
so the experiment isolates MASS; a colliding pack would also catch the rope and
arms, and any change could then be either.

Shipped policy, seed 5, mass-only:

| Pack | Total mass | Ascent | Success | min upright |
|---:|---:|---:|---|---:|
| 0 kg | 44.2 kg | **1.514 m** | yes | 0.965 |
| 6 kg | 50.2 kg | 1.060 m | no | 0.971 |
| 12 kg | 56.2 kg | 0.942 m | no | 0.966 |
| 18 kg | 62.2 kg | 0.461 m | no | 0.981 |
| 25 kg | 69.2 kg | 0.184 m | no | 0.986 |

**A 6 kg pack -- a 14% mass increase -- costs 30% of the ascent and fails the
task.** For a domain whose headline use case is carrying things up mountains,
that is the most operationally relevant number on this page.

Note `min upright` RISES with load. The robot never falls; it simply stops
making progress. That is the same root cause again: the 18-frame leg stroke has
fixed amplitude, so it does a fixed amount of work per step, and under load the
same stroke lifts less. The policy cannot lengthen or strengthen the stroke
because it does not control the legs -- `_step_power` is its only amplitude
influence and it is already saturated.

A whole-body policy would have the obvious remedy available: lean further,
push harder, take shorter steps. This one has none of them.

## Recovery metric: three corrections

The recovery criterion was wrong three times, each caught by a result that made
no physical sense. Recorded because each would have produced a published number
for something that did not happen.

**1. Momentary regain credited as recovery.** An 800 N shove cleared "double
support + 6 cm" 16 steps after the slip, was credited, then collapsed and ended
the episode at 1.031 m of a 1.5 m target. Confirmation now requires holding
stance 60 further steps and adding another 10 cm.

**2. Recovery demanded from a nudge.** Any disturbance armed a recovery
deadline. A 400 N shove displaces the robot 3.8 cm and it walks on untroubled
-- but with no confirmed recovery the episode was killed at step 545 and scored
a failure with 0.692 m of ascent, while a 700 N shove that genuinely slipped
0.35 m scored 1.520 m. **A smaller disturbance producing a worse result is the
signature of a metric bug, not a policy limit.** Recovery is now required only
for slips >= 8 cm.

**3. The airborne rewind missed absorbed slips.** The parent disqualifies an
episode if the both-feet-off streak ever exceeds 3, as a whole-episode maximum;
a real slip hits 14. The rewind that excuses this fired only on confirmed
recovery, so an absorbed nudge broke the gate with nothing to repair it. It now
fires for absorbed disturbances too.

Verified after all three, seed 5:

| Impulse | Slip | Recovery required | Recovered | Ascent | Success |
|---:|---:|---|---|---:|---|
| 0 N | 0.000 m | no | -- | 1.514 m | yes |
| 200 N | 0.024 m | no | -- | 1.521 m | yes |
| 400 N | 0.038 m | no | -- | 2.063 m | no (see below) |
| 700 N | 0.350 m | yes | yes | 1.520 m | yes |
| 800 N | 0.658 m | yes | no | 1.031 m | no |
| 1000 N | 0.982 m | yes | no | -0.568 m | no |

### A residual, in the parent's gate rather than mine

The 400 N row still fails despite climbing 2.063 m. The rewind fires correctly,
then the streak reaches 4 again later during ordinary walking.
`max_airborne_streak <= 3` is a **whole-episode maximum**, so a longer episode
has more chances to trip it: at 0 N success fires at ~1060 steps, while 400 N
runs the full 1500 and picks up a 4-step streak in the extra 440. The gate
penalises duration, not gait quality.

Left unchanged, because altering it would move the published success criterion.
A rolling window, or a rate per 100 steps, would measure what the gate is
actually for.

## Working-copy warning

`.gitattributes` LFS-tracks `*.mp4`, `*.zip` and `*.pt`. This checkout fetched
those objects through the LFS batch API without `git-lfs` installed, so `git
status` reports ~21 model zips as modified. **Do not `git add -A`** — it would
commit the binaries over their pointers. Add the new files explicitly:

```bash
git add slip_recovery_env.py evaluate_slip_recovery.py \
        train_slip_recovery.py record_slip_recovery.py SLIP_RECOVERY.md
```

## The fine-tune, and what it uncovered

Two attempts, both instructive.

**Attempt 1 — fixed anneal (1.0 -> 0.25 on a clock).** Destroyed the gait.
`ep_rew_mean` 205 -> -22.7; the checkpoint scored **0.003 m** of ascent on the
UNDISTURBED parent env, where the shipped policy gets 1.512 m. Catastrophic
forgetting: the schedule marched through the x0.5 regime the policy cannot
stand in, and never came back.

**Attempt 2 — performance-gated curriculum.** Assist steps down only while
recent episodes still climb >= 1.10 m, and steps back up if they collapse. This
worked as designed: it walked the assist down to x0.55 while still climbing
1.66 m, hit a cliff at x0.50 (ascent 0.23 m), backed off to x0.70, and held.
The gait survived -- control back to 100% / 1.512 m.

But the resulting policy is behaviourally **identical** to the shipped one:

| Condition | Shipped | Gated |
|---|---:|---:|
| parent, no disturbance | 100%, 1.512 m | 100%, 1.512 m |
| slip, assist x1 | 38%, 0.982 m | 38%, 0.982 m |
| slip, assist x0.7 | 0%, 0.978 m | 0%, 0.980 m |
| slip, assist x0.5 | 0%, 0.227 m | 0%, 0.227 m |
| slip, assist x0.25 | 0%, 0.072 m | 0%, 0.072 m |

Per-episode ascents are bit-identical across the two model files.

## Why: the policy is a metronome

The two checkpoints differ by 25% in mean absolute weight and emit visibly
different raw actions (mean |A-B| = 0.519 over a rollout). They still produce
**exactly the same trajectory**, because they produce exactly the same
step-start sequence:

```text
shipped  steps at 4, 27, 50, 73, 96, 119, 142, 165, 188, 211, ...
gated    steps at 4, 27, 50, 73, 96, 119, 142, 165, 188, 211, ...
identical: True
```

That is a fixed 23-step period: `step_duration` 18 + `step_cooldown` 3 + the
~2 frames needed to re-satisfy the 4-frame `step_request_hold`. The policy's
entire influence is whether `action[expected] > 0.22` and beats the other side
by 0.20. Any policy that saturates that gate produces this identical metronome;
`_step_power` saturates too.

So npow's ablation (neutral action -> 0.003 m, random action -> 0.107 m) is
correct but proves less than it appears: it shows the gate must be crossed, not
that anything was learned about how to walk. The learning-progress video's
"emerging gait" between 40k and 80k interactions is the policy learning to
reliably cross a threshold.

**Consequence for this stage:** fall recovery cannot be learned in this
environment, at any assist scale, under any curriculum. The policy has no
channel through which to express a recovery -- the legs follow a hardcoded
sinusoid, the timing is a state machine, and balance is the external
stabilizer's job. Widening the action space is not a follow-on nicety; it is
the precondition. Attempts 1 and 2 were both, in hindsight, tuning the
schedule of a variable that cannot matter.

## The fall is not a fall, and cannot be made into one

Measured on the shipped policy, `pelvis_normal_height` (nominal standing height
is 0.67, the value the normal-balance term targets) and `upright_score`:

| Disturbance | min pelvis height | min upright | Outcome |
|---|---:|---:|---|
| 700 N | 0.368 | 0.566 | deep crouch, recovers, +1.52 m |
| 1600 N | 0.636 | 0.905 | **never leaves upright**, slides off, -0.61 m |
| 2200 N | 0.636 | 0.905 | same, -0.63 m |
| 700 N, orientation PD zeroed | 0.417 | 0.790 | *less* disturbed, stalls at +0.03 m |

Increasing the shove does not increase the fall. Past a point the robot simply
translates downslope while standing bolt upright until it slides off the bottom
of the route. It never topples.

Five separate mechanisms hold it up, and only the last is the policy's:

1. **The legs are position-controlled to fixed stance targets** (hip -0.18,
   knee 0.38, ankle -0.52). A position-servoed leg is a rigid strut; it cannot
   buckle, which is what falling over actually is.
2. The uncapped orientation PD (gain 420) holds torso attitude.
3. The uncapped lateral spring (700 N/m) pins the robot to y=0.
4. The normal-balance term holds height (capped, 100 N).
5. The rope catches downslope motion.

And nothing could get up afterwards even if it did fall: the arms are IK-locked
to the ascender, the legs run a hardcoded stroke, and the policy is the 23-step
metronome described above. There is no actuator authority anywhere in the loop
that could produce a stand-up.

So **walk -> fall -> get up -> resume cannot be shown honestly in this
environment.** Producing that clip would require either scripting the get-up
(fabricating the capability) or letting the orientation PD haul the torso
upright and calling it recovery (filming the puppet string). What the current
demo shows -- a genuine 0.35 m slip with a crouch and a resumed climb -- is the
deepest real disturbance available here.

Getting the real thing needs, in order: torque or compliant leg control so the
robot can actually collapse; a whole-body action space; and a separate get-up
policy. That is a new environment, not a parameter change.

### Gate: can weaker actuators enable a fall? No -- and the premise was wrong.

Tested 2026-08-30, ~30 min, shipped policy.

**Correction first.** The gate was set up on the belief that the model had
unlimited actuator torque, read off `actuator_forcerange` being `[0, 0]`. That
was a misreading. MuJoCo enforces per-joint limits through `actuatorfrcrange`
on the JOINT, and this model sets them on 43 of 44 joints with the real Unitree
G1 values: knee +/-139, hip +/-88, ankle +/-50, shoulder +/-25 Nm. The torque
model was already correct; there was no fidelity win to claim.

The gate result stands regardless, and is stronger for it. With those real
limits already active, the orientation assist capped at 20 Nm and the lateral
spring removed entirely, a 1400 N shove degrades uprightness only to 0.656:

| Config | ascent | min pelvis | min upright | toppled |
|---|---:|---:|---:|---|
| baseline, no shove | 1.514 | 0.636 | 0.965 | no |
| 900 N shove, orient cap 60 Nm | 0.768 | 0.634 | 0.964 | no |
| + lateral spring 700 -> 0 | 0.759 | 0.621 | 0.956 | no |
| + orient cap 20 Nm, 1400 N shove | 0.046 | 0.395 | 0.656 | no |

So the robot cannot be toppled even with realistic actuator torque and every
external assist stripped. The cause is `_leg_targets`
(`fixed_line_slope_env.py:416`), which hardcodes
`hip, knee, ankle = -0.18, 0.38, -0.52` on EVERY control step. The legs are
actively driven back to a standing pose 50 times a second at kp=500. The robot
is not held up by the balance assist -- it is held up by its own leg
controller, continuously re-posing it. No parameter reachable from outside the
policy overrides that.

### Positive control: does the leg channel have authority?

The metronome finding rests on negative evidence -- different weights, identical
behaviour. This tests it from the other side by injecting open-loop leg
residuals and measuring trajectory divergence:

| Leg residual | Max trajectory divergence | min upright |
|---:|---:|---:|
| none (reference) | -- | 0.965 |
| **0.02 rad (1.1 deg)** | **0.105 m** | 0.962 |
| 0.05 rad | 0.169 m | 0.953 |
| 0.15 rad | 0.196 m | 0.901 |
| 0.30 rad | 0.269 m | 0.874 |
| **replacing the entire policy** | **0.000 m** | -- |

A one-degree open-loop leg perturbation moves the robot 10 cm. Swapping the
whole policy moves it nothing. The leg channel has ample authority; the policy
simply does not reach it. Uprightness also degrades monotonically with residual
amplitude, so that channel could plausibly carry balance -- which is what makes
widening the action space the indicated next step rather than a guess.

Cost: 2 minutes, versus the ~3 hours a retrain would have taken to answer the
same question with a 45% chance of an uninterpretable result.

## Rope appearance

`fixed_rope` shipped as a single 26 mm capsule -- constant diameter, specular
highlight, no surface detail -- which reads as steel cable.
`scripts/generate_rope_visual.py` replaces it with an 11 mm kernmantle rope: a
UV-mapped tube mesh carrying a procedurally generated braided-sheath texture.

Safe by construction: the geom is already `contype="0" conaffinity="0"` and the
ascender's position comes from `_rope_point()` in Python, not from the geom.
Verified nq 50, nv 49, nu 43, neq 0 unchanged, ngeom back to 140.

Four things learned getting there, each of which cost a render:

- **Laid rope is the wrong construction.** The first attempt drew three
  helical strands. That is hawser-laid rope -- marine/utility. Fixed lines are
  kernmantle: braided sheath over a core.
- **The lay aliases if undersampled.** 96 segments against 99 turns is just
  over one twist per capsule, and rendered as three straight parallel rails --
  looking *more* like cable than the thing it replaced.
- **MuJoCo does not UV-wrap primitives.** A braid texture on a capsule renders
  as banding along the axis, never as a weave around it; checked against
  `2d`/`cube` and `texuniform` both ways. Hence the tube mesh with real
  texture coordinates.
- **MuJoCo re-centres and re-orients every mesh onto its principal axes.** A
  tube authored in world coordinates comes back translated to the body origin
  and pointing down local z, i.e. through the slope. The mesh is therefore
  authored canonically (centred, +Z aligned) and placed by the geom's own
  `pos`/`quat`.

Two calibration notes. The texture tile maps to a *square* patch of rope
surface -- one tile length equals one circumference -- so the braid's diagonals
meet the axis at the angle they were drawn at; map a tile to a long thin patch
and the weave shears into axial stripes. And the diameter is now correct rather
than convenient: **an 11 mm line is 2-3 pixels wide at the demo's 5 m camera**,
so the braid is only legible in close-up. The shipped 26 mm was more visible
precisely because it was twice the size of a real fixed line.

## Load carriage

`scripts/add_backpack.py` attaches a `haul_pack` body to `torso_link` with real
mass and inertia -- a climber on a fixed line carries a pack, and a visual-only
geom would look right and test nothing. It adds **no degrees of freedom** (a
body with no joint welds into the tree), so nq 50 / nv 49 / nu 43 / neq 0 are
unchanged and everything downstream still composes. Contact is off by default
so the experiment isolates MASS; a colliding pack would also catch the rope and
arms, and any change could then be either.

Shipped policy, seed 5, mass-only:

| Pack | Total mass | Ascent | Success | min upright |
|---:|---:|---:|---|---:|
| 0 kg | 44.2 kg | **1.514 m** | yes | 0.965 |
| 6 kg | 50.2 kg | 1.060 m | no | 0.971 |
| 12 kg | 56.2 kg | 0.942 m | no | 0.966 |
| 18 kg | 62.2 kg | 0.461 m | no | 0.981 |
| 25 kg | 69.2 kg | 0.184 m | no | 0.986 |

**A 6 kg pack -- a 14% mass increase -- costs 30% of the ascent and fails the
task.** For a domain whose headline use case is carrying things up mountains,
that is the most operationally relevant number on this page.

Note `min upright` RISES with load. The robot never falls; it simply stops
making progress. That is the same root cause again: the 18-frame leg stroke has
fixed amplitude, so it does a fixed amount of work per step, and under load the
same stroke lifts less. The policy cannot lengthen or strengthen the stroke
because it does not control the legs -- `_step_power` is its only amplitude
influence and it is already saturated.

A whole-body policy would have the obvious remedy available: lean further,
push harder, take shorter steps. This one has none of them.

## Recovery metric: three corrections

The recovery criterion was wrong three times, each caught by a result that made
no physical sense. Recorded because each would have produced a published number
for something that did not happen.

**1. Momentary regain credited as recovery.** An 800 N shove cleared "double
support + 6 cm" 16 steps after the slip, was credited, then collapsed and ended
the episode at 1.031 m of a 1.5 m target. Confirmation now requires holding
stance 60 further steps and adding another 10 cm.

**2. Recovery demanded from a nudge.** Any disturbance armed a recovery
deadline. A 400 N shove displaces the robot 3.8 cm and it walks on untroubled
-- but with no confirmed recovery the episode was killed at step 545 and scored
a failure with 0.692 m of ascent, while a 700 N shove that genuinely slipped
0.35 m scored 1.520 m. **A smaller disturbance producing a worse result is the
signature of a metric bug, not a policy limit.** Recovery is now required only
for slips >= 8 cm.

**3. The airborne rewind missed absorbed slips.** The parent disqualifies an
episode if the both-feet-off streak ever exceeds 3, as a whole-episode maximum;
a real slip hits 14. The rewind that excuses this fired only on confirmed
recovery, so an absorbed nudge broke the gate with nothing to repair it. It now
fires for absorbed disturbances too.

Verified after all three, seed 5:

| Impulse | Slip | Recovery required | Recovered | Ascent | Success |
|---:|---:|---|---|---:|---|
| 0 N | 0.000 m | no | -- | 1.514 m | yes |
| 200 N | 0.024 m | no | -- | 1.521 m | yes |
| 400 N | 0.038 m | no | -- | 2.063 m | no (see below) |
| 700 N | 0.350 m | yes | yes | 1.520 m | yes |
| 800 N | 0.658 m | yes | no | 1.031 m | no |
| 1000 N | 0.982 m | yes | no | -0.568 m | no |

### A residual, in the parent's gate rather than mine

The 400 N row still fails despite climbing 2.063 m. The rewind fires correctly,
then the streak reaches 4 again later during ordinary walking.
`max_airborne_streak <= 3` is a **whole-episode maximum**, so a longer episode
has more chances to trip it: at 0 N success fires at ~1060 steps, while 400 N
runs the full 1500 and picks up a 4-step streak in the extra 440. The gate
penalises duration, not gait quality.

Left unchanged, because altering it would move the published success criterion.
A rolling window, or a rate per 100 steps, would measure what the gate is
actually for.

## Working-copy warning

`.gitattributes` LFS-tracks `*.mp4`, `*.zip` and `*.pt`. This checkout fetched
those objects through the LFS batch API without `git-lfs` installed, so `git
status` reports ~21 model zips as modified. **Do not `git add -A`** — it would
commit the binaries over their pointers. Add the new files explicitly:

```bash
git add slip_recovery_env.py evaluate_slip_recovery.py \
        train_slip_recovery.py record_slip_recovery.py SLIP_RECOVERY.md
```

## The fine-tune, and what it uncovered

Two attempts, both instructive.

**Attempt 1 — fixed anneal (1.0 -> 0.25 on a clock).** Destroyed the gait.
`ep_rew_mean` 205 -> -22.7; the checkpoint scored **0.003 m** of ascent on the
UNDISTURBED parent env, where the shipped policy gets 1.512 m. Catastrophic
forgetting: the schedule marched through the x0.5 regime the policy cannot
stand in, and never came back.

**Attempt 2 — performance-gated curriculum.** Assist steps down only while
recent episodes still climb >= 1.10 m, and steps back up if they collapse. This
worked as designed: it walked the assist down to x0.55 while still climbing
1.66 m, hit a cliff at x0.50 (ascent 0.23 m), backed off to x0.70, and held.
The gait survived -- control back to 100% / 1.512 m.

But the resulting policy is behaviourally **identical** to the shipped one:

| Condition | Shipped | Gated |
|---|---:|---:|
| parent, no disturbance | 100%, 1.512 m | 100%, 1.512 m |
| slip, assist x1 | 38%, 0.982 m | 38%, 0.982 m |
| slip, assist x0.7 | 0%, 0.978 m | 0%, 0.980 m |
| slip, assist x0.5 | 0%, 0.227 m | 0%, 0.227 m |
| slip, assist x0.25 | 0%, 0.072 m | 0%, 0.072 m |

Per-episode ascents are bit-identical across the two model files.

## Why: the policy is a metronome

The two checkpoints differ by 25% in mean absolute weight and emit visibly
different raw actions (mean |A-B| = 0.519 over a rollout). They still produce
**exactly the same trajectory**, because they produce exactly the same
step-start sequence:

```text
shipped  steps at 4, 27, 50, 73, 96, 119, 142, 165, 188, 211, ...
gated    steps at 4, 27, 50, 73, 96, 119, 142, 165, 188, 211, ...
identical: True
```

That is a fixed 23-step period: `step_duration` 18 + `step_cooldown` 3 + the
~2 frames needed to re-satisfy the 4-frame `step_request_hold`. The policy's
entire influence is whether `action[expected] > 0.22` and beats the other side
by 0.20. Any policy that saturates that gate produces this identical metronome;
`_step_power` saturates too.

So npow's ablation (neutral action -> 0.003 m, random action -> 0.107 m) is
correct but proves less than it appears: it shows the gate must be crossed, not
that anything was learned about how to walk. The learning-progress video's
"emerging gait" between 40k and 80k interactions is the policy learning to
reliably cross a threshold.

**Consequence for this stage:** fall recovery cannot be learned in this
environment, at any assist scale, under any curriculum. The policy has no
channel through which to express a recovery -- the legs follow a hardcoded
sinusoid, the timing is a state machine, and balance is the external
stabilizer's job. Widening the action space is not a follow-on nicety; it is
the precondition. Attempts 1 and 2 were both, in hindsight, tuning the
schedule of a variable that cannot matter.

## The fall is not a fall, and cannot be made into one

Measured on the shipped policy, `pelvis_normal_height` (nominal standing height
is 0.67, the value the normal-balance term targets) and `upright_score`:

| Disturbance | min pelvis height | min upright | Outcome |
|---|---:|---:|---|
| 700 N | 0.368 | 0.566 | deep crouch, recovers, +1.52 m |
| 1600 N | 0.636 | 0.905 | **never leaves upright**, slides off, -0.61 m |
| 2200 N | 0.636 | 0.905 | same, -0.63 m |
| 700 N, orientation PD zeroed | 0.417 | 0.790 | *less* disturbed, stalls at +0.03 m |

Increasing the shove does not increase the fall. Past a point the robot simply
translates downslope while standing bolt upright until it slides off the bottom
of the route. It never topples.

Five separate mechanisms hold it up, and only the last is the policy's:

1. **The legs are position-controlled to fixed stance targets** (hip -0.18,
   knee 0.38, ankle -0.52). A position-servoed leg is a rigid strut; it cannot
   buckle, which is what falling over actually is.
2. The uncapped orientation PD (gain 420) holds torso attitude.
3. The uncapped lateral spring (700 N/m) pins the robot to y=0.
4. The normal-balance term holds height (capped, 100 N).
5. The rope catches downslope motion.

And nothing could get up afterwards even if it did fall: the arms are IK-locked
to the ascender, the legs run a hardcoded stroke, and the policy is the 23-step
metronome described above. There is no actuator authority anywhere in the loop
that could produce a stand-up.

So **walk -> fall -> get up -> resume cannot be shown honestly in this
environment.** Producing that clip would require either scripting the get-up
(fabricating the capability) or letting the orientation PD haul the torso
upright and calling it recovery (filming the puppet string). What the current
demo shows -- a genuine 0.35 m slip with a crouch and a resumed climb -- is the
deepest real disturbance available here.

Getting the real thing needs, in order: torque or compliant leg control so the
robot can actually collapse; a whole-body action space; and a separate get-up
policy. That is a new environment, not a parameter change.

### Gate: can weaker actuators enable a fall? No -- and the premise was wrong.

Tested 2026-08-30, ~30 min, shipped policy.

**Correction first.** The gate was set up on the belief that the model had
unlimited actuator torque, read off `actuator_forcerange` being `[0, 0]`. That
was a misreading. MuJoCo enforces per-joint limits through `actuatorfrcrange`
on the JOINT, and this model sets them on 43 of 44 joints with the real Unitree
G1 values: knee +/-139, hip +/-88, ankle +/-50, shoulder +/-25 Nm. The torque
model was already correct; there was no fidelity win to claim.

The gate result stands regardless, and is stronger for it. With those real
limits already active, the orientation assist capped at 20 Nm and the lateral
spring removed entirely, a 1400 N shove degrades uprightness only to 0.656:

| Config | ascent | min pelvis | min upright | toppled |
|---|---:|---:|---:|---|
| baseline, no shove | 1.514 | 0.636 | 0.965 | no |
| 900 N shove, orient cap 60 Nm | 0.768 | 0.634 | 0.964 | no |
| + lateral spring 700 -> 0 | 0.759 | 0.621 | 0.956 | no |
| + orient cap 20 Nm, 1400 N shove | 0.046 | 0.395 | 0.656 | no |

So the robot cannot be toppled even with realistic actuator torque and every
external assist stripped. The cause is `_leg_targets`
(`fixed_line_slope_env.py:416`), which hardcodes
`hip, knee, ankle = -0.18, 0.38, -0.52` on EVERY control step. The legs are
actively driven back to a standing pose 50 times a second at kp=500. The robot
is not held up by the balance assist -- it is held up by its own leg
controller, continuously re-posing it. No parameter reachable from outside the
policy overrides that.

### Positive control: does the leg channel have authority?

The metronome finding rests on negative evidence -- different weights, identical
behaviour. This tests it from the other side by injecting open-loop leg
residuals and measuring trajectory divergence:

| Leg residual | Max trajectory divergence | min upright |
|---:|---:|---:|
| none (reference) | -- | 0.965 |
| **0.02 rad (1.1 deg)** | **0.105 m** | 0.962 |
| 0.05 rad | 0.169 m | 0.953 |
| 0.15 rad | 0.196 m | 0.901 |
| 0.30 rad | 0.269 m | 0.874 |
| **replacing the entire policy** | **0.000 m** | -- |

A one-degree open-loop leg perturbation moves the robot 10 cm. Swapping the
whole policy moves it nothing. The leg channel has ample authority; the policy
simply does not reach it. Uprightness also degrades monotonically with residual
amplitude, so that channel could plausibly carry balance -- which is what makes
widening the action space the indicated next step rather than a guess.

Cost: 2 minutes, versus the ~3 hours a retrain would have taken to answer the
same question with a 45% chance of an uninterpretable result.

## Rope appearance

`fixed_rope` was a single 11.4 m capsule -- constant diameter, specular
highlight, no lay -- which reads as steel cable. `scripts/generate_rope_visual.py`
replaces it with three helical strands of short capsules plus matte materials.

Safe by construction: the original geom is already `contype="0"
conaffinity="0"` and the ascender's position comes from `_rope_point()` in
Python, not from the geom. The original is kept, made invisible, so anything
reading the scene for the rope axis still finds it. Verified: nq 50, nv 49,
nu 43, neq 0 unchanged.

Two things worth knowing if you re-tune it:

- **The lay aliases if undersampled.** The first attempt used 96 segments
  against 99 turns -- just over one twist per capsule -- and rendered as three
  straight parallel rails, looking *more* like cable than the thing it
  replaced. `MIN_SEGMENTS_PER_TURN` now makes that a hard failure.
- **Pitch is a deliberate compromise.** A real 26 mm rope lays at ~0.10-0.15 m
  pitch; over 11.4 m that is ~100 turns needing ~1000 segments per strand.
  0.25 m is a long lay that costs realism up close and buys a twist that
  resolves at the distance this scene is filmed from. Cost: ngeom 140 -> 1820,
  all visual.

## Open

- The annealed fine-tune's numbers. The test that matters: does it hold up at
  assist x0.25, where the shipped checkpoint manages 0.071 m?
- Every recovery number must be reported with the assist scale it was measured
  at. At x1.0 the number is not about the policy.
- Widen the action space once a policy stands unaided.
- Nothing here is pushed. This is npow's private repo.

## Related work

[abhijitbetigeri/HimalayaExpedition](https://github.com/abhijitbetigeri/HimalayaExpedition)
— Robotic Expedition in Himalayas. Companion project covering the wider
Himalayan robotics track: ice/snow locomotion under domain randomization, wind
loading, and fixed-line ascent on MuJoCo Playground (MJX/GPU), alongside the
LiveKit voice interface used in this demo.

