# Unitree G1 Learned Mountaineering Self-Arrest

## Status

The diverse-condition learned-policy result is complete. One deterministic
Stable-Baselines3 PPO checkpoint arrests the G1 in all nine named oblique-fall
scenarios and all 60 unseen randomized falls. These tests disable every
non-axe body/slope friction contact, so the result cannot pass by sliding to a
stop on the torso or limbs.

The earlier fall-line-only result is preserved in
`models/ppo_self_arrest_fall_line_v1/`; it is no longer canonical.

Canonical artifacts:

- video: [`videos/g1_self_arrest_diverse_suite.mp4`](videos/g1_self_arrest_diverse_suite.mp4)
- video metrics: [`videos/g1_self_arrest_diverse_suite.json`](videos/g1_self_arrest_diverse_suite.json)
- PPO checkpoint: [`models/ppo_self_arrest/g1_self_arrest_final.zip`](models/ppo_self_arrest/g1_self_arrest_final.zip)
- full fixed/randomized/causal audit: [`models/ppo_self_arrest/diverse_evaluation_report.json`](models/ppo_self_arrest/diverse_evaluation_report.json)
- checkpoint selection: [`models/ppo_self_arrest/checkpoint_selection_report.json`](models/ppo_self_arrest/checkpoint_selection_report.json)
- training metadata: [`models/ppo_self_arrest/training_metadata.json`](models/ppo_self_arrest/training_metadata.json)
- natural snow texture: [`assets/textures/alpine_snow_v1.png`](assets/textures/alpine_snow_v1.png)
- inspected montage: [`videos/diverse_suite_inspection_montage.jpg`](videos/diverse_suite_inspection_montage.jpg)

Absolute video path:

```text
/home/npow/code/himalaya/videos/g1_self_arrest_diverse_suite.mp4
```

## What is learned

This is a MuJoCo RL rollout, not a keyframed, time-indexed, or hardcoded video.
At every 100 Hz policy step, the recording scripts call
`policy.predict(observation, deterministic=True)` and apply the resulting
action before advancing physics.

The PPO actor and critic are 256×256 `tanh` MLPs. The actor receives a
125-value observation containing body orientation and angular velocity,
downhill/cross-slope/normal velocity, joint errors and velocities, measured
pick/chest/hand loads, pick geometry and blade angle, stroke displacement, and
the previous action. It outputs 14 continuous residual joint-position
commands: all seven joints of each arm.

PPO therefore has to adapt the two-arm pick plant and loading stroke to the
observed heading, lateral motion, and roll. The video uses one saved checkpoint
for every scenario; there is no per-scenario controller or trajectory.

This remains a deliberately hierarchical first-stage skill:

- the robot starts in a prepared prone feet-downhill slide;
- the waist, legs, and prepared body pose use low-level position control;
- articulated fingers use fixed closure targets, while real opposing-digit
  contacts are measured and required throughout the arrest;
- the axe is rigidly retained at the right palm and cannot be dropped;
- both complete arms, including the support hand position, are PPO-controlled.

It learns the pick plant and sustained loading from varied sliding attitudes.
It does not learn grasp acquisition, tool retention, fall recovery, or a
complete deployable mountaineering behavior.

## Diverse named scenarios

Heading is relative to the fall line. Cross-slope velocity and roll can agree,
oppose, or have mixed signs. The nominal downhill component is 4.2–4.8 m/s.
The table reports the actual no-body-friction video rollouts.

| Scenario | Heading | Cross-slope | Roll | First pick angle | Plant travel | Final speed |
|---|---:|---:|---:|---:|---:|---:|
| Fall-line reference | 0° | 0.0 m/s | 0° | 29.89° | 17.51 cm | 0.040 m/s |
| Left oblique | +20° | +0.8 m/s | +3° | 30.74° | 17.24 cm | 0.019 m/s |
| Right oblique | −20° | −0.8 m/s | −3° | 30.04° | 17.57 cm | 0.031 m/s |
| Body/velocity disagree—left | +25° | −0.9 m/s | +5° | 31.83° | 17.80 cm | 0.025 m/s |
| Body/velocity disagree—right | −25° | +0.9 m/s | −5° | 30.62° | 18.24 cm | 0.016 m/s |
| Hard compound—left | +35° | +1.2 m/s | +8° | 33.54° | 18.66 cm | 0.014 m/s |
| Hard compound—right | −35° | −1.2 m/s | −8° | 30.97° | 18.75 cm | 0.038 m/s |
| Hard crossed/rolled—left | +35° | −0.8 m/s | −9° | 30.65° | 17.80 cm | 0.039 m/s |
| Hard crossed/rolled—right | −35° | +0.8 m/s | +9° | 20.65° | 16.90 cm | 0.017 m/s |

All nine succeed. The minimum left opposing-digit contact fraction is 74.7%,
the minimum right fraction is 99.2%, and all nine plants satisfy the learned
motion gate.

The last two scenarios were added after audit found two different policy
failures: one checkpoint lost the support-hand grasp; another stopped with a
15.7° shallow entry. They were not hidden or solved by weakening the gates.
PPO was fine-tuned in neighborhoods around both failures until a checkpoint
passed them and the original seven cases.

## Unseen randomized audit

The randomized envelope is continuous rather than a lookup over the named
cases:

```text
downhill speed: 4.0 to 5.0 m/s
heading:       -40 to +40 degrees
cross-slope:   -1.5 to +1.5 m/s
body roll:     -10 to +10 degrees
```

For 60 unseen resets with every non-axe body friction contact disabled:

- successes: **60/60**;
- valid learned plants: **60/60**;
- mean final speed: **0.030 m/s**;
- maximum final speed: **0.063 m/s**;
- minimum contact delay: **24 policy steps**;
- minimum plant travel: **14.51 cm**;
- minimum lowering: **5.95 cm**;
- minimum first-contact blade angle: **18.98°**;
- terminal rolling blade-angle range: **24.68–33.13°**;
- minimum left opposing-digit contact fraction: **72.0%**;
- minimum right opposing-digit contact fraction: **98.9%**;
- minimum axe/wrist front-of-torso margin: **5.66 cm**.

The same 60 seeds with normal friction produce numerically identical results.

## Causal evidence

| Condition | Fixed | Randomized | Mean final speed |
|---|---:|---:|---:|
| Normal simulation | 9/9 | 60/60 | 0.026 / 0.030 m/s |
| All non-axe body friction disabled | **9/9** | **60/60** | **0.026 / 0.030 m/s** |
| Axe-specific cohesive-snow resistance disabled | 0/9 | 0/30 | 0.397 / 0.336 m/s |
| Physical pick collision disabled | 0/9 | 0/30 | 13.526 / 13.526 m/s |

Normal and no-body-friction trajectories match. Removing the phenomenological
cohesive-snow force prevents success, although rigid Coulomb contact at the
physical pick still produces some drag. Disabling pick collision removes both
mechanisms: every rollout accelerates to the runaway guard near 13.5 m/s. This
is the strongest check that the arrest is caused by the axe pick rather than
passive body friction.

## Physical contact and success gates

The visible pick and its 8 mm collision proxy share one axe-local frame. On
each MuJoCo substep, `mj_step1` computes current contact, the code applies a
velocity-opposing cohesive-snow force only at a currently contacting and
properly angled pick, and `mj_step2` integrates it. The force is zero on the
same substep if rigid pick contact is absent.

Success requires all of the following for 25 consecutive policy steps:

```text
slope speed < 0.20 m/s and < 25% of initial speed
50-step rigid-pick contact window is complete
rolling rigid-pick contact fraction > 0.50
rolling mean axe snow load > 100 N
first rigid contact occurs at step >= 20
pick travel at first contact > 0.08 m
pick lowering at first contact > 0.05 m
first-contact blade angle > 18 degrees
rolling blade angle remains between 22 and 42 degrees
chest-down score > 0.60
14-finger grip score > 0.85
left opposing-digit rollout contact fraction > 0.70
right opposing-digit rollout contact fraction > 0.90
axe and both wrists stay > 3 cm in front of the torso
axe head remains in the upper-torso region
```

An invalid slow stop is penalized and eventually terminated. A shallow pick,
passive friction, brief tap, released support hand, or behind-the-back axe pose
cannot pass.

## Training and selection

Training used Stable-Baselines3 PPO with 18 parallel local MuJoCo workers. A
curriculum expanded speed, heading, cross-slope motion, and roll to the full
randomized envelope. The finishing stage mixed continuous random resets with
jittered named/adversarial anchors and strengthened the reward penalty for a
premature or sub-18° first contact.

The final stage resumed from the 1,989,270-step checkpoint, used a 2e-5
learning rate and 90% anchor-neighborhood sampling, and saved checkpoints every
20,000 steps. Checkpoint 2,169,252 was selected after 9/9 fixed and 60/60
randomized strict passes. A later 2,309,238-step checkpoint still passed 9/9
but regressed to 58/60 randomized, so it was rejected.

All training and evaluation ran locally on this machine. The CPU PPO/MuJoCo
path was faster for this small MLP and parallel simulation workload; no cloud
GPU or cloud CPU was used.

## Video and terrain design

The delivered H.264 video is 1920×1080, 50 FPS, and 53.86 seconds. Each segment
has:

- the same learned PPO checkpoint and non-axe body friction disabled;
- a fixed 12.5 m camera, so the robot crosses the image instead of being
  tracked at the center;
- a live 620×420 hand/pick view showing the shaft, pick, fingers, and support
  hand contacts;
- a live 420×285 slope-normal view showing heading and cross-slope travel;
- scenario heading, lateral velocity, roll, speed, pick contact, plant travel,
  blade angle, and snow load;
- the first 0.61 simulation-seconds at labeled 0.5× playback, followed by the
  same rollout at real time.

The old checkerboard, flat-white slope, and one-metre grid were removed. The
new scene uses a project-local natural wind-packed snow/ice texture, a sky and
terrain color gradient, low-angle directional light with cast shadows,
visual-only wind crust relief, and widely spaced route wands for scale. None of
these visual reference features collide with the robot or change physics.

The snow texture was generated with the built-in image-generation tool from a
prompt requesting a seamless orthographic alpine snow/ice diffuse texture with
blue-gray variation, sastrugi, granular ice, sparse mineral flecks, no baked
shadows, and no checker/grid/text/watermark. Runtime MuJoCo lighting supplies
the shadows.

## Reproduce

From `/home/npow/code/himalaya`:

```bash
source .venv/bin/activate

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python evaluate_diverse_policy.py \
  --model models/ppo_self_arrest/g1_self_arrest_final.zip \
  --randomized-episodes 60 \
  --seed 73000 \
  --output models/ppo_self_arrest/diverse_evaluation_report.json

MUJOCO_GL=egl OMP_NUM_THREADS=1 \
  python record_diverse_scenarios.py \
  --model models/ppo_self_arrest/g1_self_arrest_final.zip \
  --output videos/g1_self_arrest_diverse_suite.mp4 \
  --seed 81000
```

## Integrity

```text
b46970f01df9caa9a4786f9c99840095a896b2b4ca801433e7494003a77b6765  models/ppo_self_arrest/g1_self_arrest_final.zip
12360bb592b93339381a74c0073f53e0cba9746abafa44cb7fa53adf5ba45779  models/ppo_self_arrest/diverse_evaluation_report.json
7ccb0b6263e1cddf3f5a956b2eb43812ba55edbcb85ec378e3ca4340eb5f36df  models/ppo_self_arrest/checkpoint_selection_report.json
46dd31216865634f401769fa3d947621db12117d39e008010081be68be0f72d1  models/ppo_self_arrest/training_metadata.json
aeda3f802acb9dbd2b4ac10bb3b91329d910630da4c0552078ed4fce91d345c6  videos/g1_self_arrest_diverse_suite.mp4
6fe5db7988d43fbde18b3bced84f763dd77abf56ece7c59693e13adcb9a3ba19  videos/g1_self_arrest_diverse_suite.json
de829959d28e5019c77ea3b76e63b61b7ba384452c19c88786dd3cb20e979ee0  assets/textures/alpine_snow_v1.png
```

## Remaining scope

This is simulation-only. A later stage should free the axe, learn grasp
acquisition and retention, actuate the whole body, train head-first and
tumbling entries, randomize snow/contact and robot dynamics, and use a
conservative sim-to-real safety process before any hardware experiment.
