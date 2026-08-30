# Unitree G1 Inclined Fixed-Line Travel and Recovery

## Status

The rejected hanging-rope scene has been replaced. The current policy travels
up a 28° snow/ice slope with the fixed rope running parallel to the fall line.
It alternates real left/right boot contacts, uses boot-hugging crampon rails,
holds a handled ascender with the right hand, and keeps the left hand open for
balance. The current presentation composites only the synthetic far-field sky
with a Himalayan mountain plate; the moving snow texture, robot, shadows, rope,
and all contact geometry remain native to the MuJoCo scene.

- Current mountain video: `videos/g1_fixed_line_mountain_v2.mp4`
- Previous plain-scene video: `videos/g1_fixed_line_learned.mp4`
- Historical learning-over-time video: `videos/g1_fixed_line_learning_progress.mp4`
- Selected checkpoint: `models/ppo_fixed_line_slope/g1_fixed_line_final.zip`
- Ten-reset physical-rope evaluation: `models/ppo_fixed_line_slope/evaluation_report.json`
- Rollout telemetry: `videos/g1_fixed_line_mountain_v2.json`
- Environment: `fixed_line_slope_env.py`
- Scene: `assets/unitree_g1/scene_fixed_line_slope.xml`
- Integrated fall/get-up video: `videos/g1_fixed_line_fall_recovery.mp4`
- Integrated telemetry: `videos/g1_fixed_line_fall_recovery.json`
- Recovery checkpoint: `models/ppo_mountain_recovery/g1_mountain_recovery_final.zip`
- Recovery evaluation: `models/ppo_mountain_recovery/evaluation_report.json`
- Recovery controller/environment: `mountain_recovery.py`

The recorder calls the saved PPO policy on every frame. The learned videos do
not use the evaluator's scripted reference action.

## Verified result

The local acceptance suite uses ten independently perturbed resets for every
learned/control condition and five for the reference mechanics check.

| Condition | Success | Along-slope travel | Vertical gain | Steps | Grounded |
|---|---:|---:|---:|---:|---:|
| Learned PPO policy | **100%** | **1.535 m** | **0.698 m** | **21.3** | **99.2%** |
| Same policy, step actuation disabled | 0% | 0.734 m | 0.166 m | 0.0 | 100% |
| Same policy, arm pull disabled | 0% | 3.193 m* | 1.458 m* | 47.0 | 99.0% |
| Same policy, line and step actuation disabled | 0% | -0.550 m | -0.230 m | 0.0 | 100% |
| Neutral action | 0% | 0.734 m | 0.166 m | 0.0 | 100% |
| Uniform-random action | 0% | 0.581 m | 0.067 m | 3.8 | 99.7% |

\* The arm-disabled condition cannot satisfy the physical arm-load success
gate and therefore runs the full 1,100-step horizon. At the matched 300-step
horizon it reaches 0.811 m versus 0.954 m with the learned arm pull; its mean
time to 1.5 m is 552 steps versus 491 steps for the complete system.

All 31 executable gates pass. Learned episodes average 55.5% left-boot contact,
56.5% right-boot contact, and 12.9% double support. The longest interval with
neither boot registered is one 20 ms policy step. The recorded deterministic
mountain rollout reaches 1.510 m in 511 steps and peaks at 525 N measured ground
load. Its rope deforms by 18.0 cm, extends by at most 1.68 cm, reaches the 80 N
transverse cam-guide cap, and records zero core/leg, hand-hand, hand-slope, or
hand-through-rope penetration steps. It accumulates 90.45 N·s of right-arm pull
impulse with a 35.99 N peak learned Jumar load.

At the matched 300-step horizon, disabling alternating step actuation limits
progress to 0.245 m versus 0.954 m for the learned controller. Disabling only
the arm pull lowers progress to 0.811 m and increases time to target by 61
policy steps. Removing the line causes a 0.55 m downslope slide and failure.
Random actions never finish.

## Integrated fall, get-up, and resumed ascent

The same mountain environment now runs continuously through the failure case:
the saved climbing PPO advances for 112 policy steps; a finite 100 N downslope
pelvis force and 86 N·m torso torque act for 0.4 s; passive motors, gravity,
contacts, and the deformable fixed line produce and arrest the fall; a learned
four-action recovery policy gets the grounded robot upright; the right hand
physically re-grasps the lower Jumar; and the original climbing PPO resumes.
The floating base is never written after reset.

The initial chest catch limits fall-line loss to at most 3.8 mm in the two
declared lateral-disturbance cases. While grounded, the robot transfers to an
explicit 1.1 m energy-absorbing lanyard at the same locked rope coordinate.
That slack permits the hands-and-knees transition, then the tether catches at
the 700 N force cap with an equal-and-opposite reaction on interpolated rope
vertices. The nominal recovery moves 1.085 m downslope while grounded, then
the resumed climbing PPO completes two physically continuous segments totaling
3.070 m: net **+1.981 m after the fall**. Segment rebasing clears task counters
only; it does not change the robot, rope, or floating-base state.

The executable integrated gates report:

- 2/2 bounded lateral fall directions recover and settle through re-grasp.
- The unchanged pretrained WBC prior fails the same nominal post-climb case.
- Peak normalized motor torque is 1.0; maximum contact penetration is 27.8 mm.
- Rope/core collision frames and hand/rope penetration frames are both zero.
- The resumed policy completes 3.070 m uphill with zero rope/core or
  hand/rope penetration steps.

The evidence video uses a fixed free camera aimed at a point above the slope.
It never tracks the fallen pelvis, so the view cannot pan below the terrain;
the static world frame also makes the lower-left to upper-right ascent visible.

This is a bounded hackathon demonstration, not a universal-fall claim. The
tested envelope is the documented 112-step climb state, an 86 N·m/0.4 s pitch
disturbance, and +4 to +5 N lateral bias.

### Recovery learning provenance

The low-level whole-body prior and motion clip are the pinned Apache-2.0
artifacts from `wbc-mjlab/wbc-g1-deploy` commit
`6dabf86fddc2b7b429b09e74999732fcde3441f9`; this repository does not claim to
have trained that prior. The 145-observation, four-action PPO handoff policy
*was* trained here. Its actions state-dependently brake the left leg, right
leg, waist, and arm groups during the critical WBC-to-fixed-line transfer.

The selected cloud run used 24 CPU workers and reached 911,008 cumulative
interactions. Hugging Face Job:
`iteratehack/6a947232984507d9db4ed46a`. A further 50,176-step local fine-tune on
the finalized disturbance achieved held-out reward 137.34 in 171 steps, but it
was not promoted because the earlier checkpoint passed the wider physical
gate. Selection is based on the integrated evaluator, not the rendered video
or training reward alone.

## Terrain, rope, hands, and feet

The MuJoCo plane is rotated 28° above horizontal. The orange 14 mm handline is
a physical one-dimensional flex with 33 mass-bearing vertices, capsule
collision, pinned route anchors, stiff axial equality constraints, and internal
sheath/core damping. Gravity, contact, and the ascender reaction deform the
line; it is no longer a rigid display capsule. Four broad counter-wound yarn
bundles follow the live vertices across the camera view. Their alternating
colors, over/under gaps, matte finish, and lobed silhouette make the sheath
read as braided kernmantle rope instead of a smooth cable. The subdued flex
core remains the 14 mm collision and force-bearing body.

The rope follows the fall line at knee-to-waist height and is routed 32 cm to
the robot's right. Exact flex-contact telemetry—not a visual distance proxy—
rejects any rope contact with the pelvis, torso, head, or legs. The right hand
uses matched articulated render/contact envelopes around a 156 mm offset
ascender handle and physically collides with the sheath using a 6 mm predictive
contact margin. The forearm receives the equal-and-opposite reaction through a
compliant cam guide instead of rubbing directly on the line. The scene also
uses subtle snow/ice texture, a harness, leg loops, lanyard, compact cam
devices, and visual-only crampon rails and points attached to the physical
ankle bodies.

The right wrist is retained on a lower handled ascender at a compact 0.22 m
reach above the pelvis with seven-joint world-frame damped-least-squares IK.
The Jumar slides upward while unloaded and locks to a fixed rope coordinate
during each pull. Curled distal finger links remain around the handle while the
rope moves through the cam channel. The left arm is not
forced across the torso: it stays bent, open, and clear for balance. Across ten
learned rollouts, mean worst-case wrist-to-handle registration error is 11.4 cm,
minimum hand separation averages 54.5 cm, and there are no hand collisions or
hand/rope penetrations.
The final video includes a live oblique close-up of the cam, handle, deforming
rope sheath, and harness tether.

The ankle pitch is now four hundredths of a radian more downslope than the
original render, keeping the front points visibly buried instead of leaving an
ambiguous gap at the snow. The old oversized visual plates were replaced by
narrow rails, crossbars, and short points that follow the physical boot frame.

The three policy actions request left and right uphill steps and command the
right-arm Jumar pull. A step request must persist for four policy frames; the
low-level target then executes an 18-frame swing/stance stroke while PPO chooses
the arm load. The learned positive arm command is capped at 18% bodyweight and
shaped over the stroke. It applies uphill force at the right wrist and the exact
equal-and-opposite force to interpolated deformable-rope vertices at the locked
Jumar coordinate. This is a force-balanced rope interaction, not a body-frame
teleport or video effect.

Position-actuated leg movement still resolves through real boot/slope contact.
A separate one-way chest catch prevents lost height and advances only during a
deliberate stroke, avoiding rectification of idle controller noise. A capped
low-level balance assist controls lateral drift, torso rotation, and at most
100 N in the surface-normal direction; it cannot advance the robot along the
slope. The arm-disabled ablation quantifies the learned arm's contribution.

## Learning progression

The historical `videos/g1_fixed_line_learning_progress.mp4` synchronizes four
deterministic policies from the same reset. It predates the deformable-rope
retrofit and is retained only as training-history evidence; current mechanics
are verified by the physical-rope report linked above.

| Policy stage | Along-slope travel | Steps | Success |
|---|---:|---:|---:|
| Initialized network, 0 interactions | 0.004 m | 0 | no |
| Early checkpoint, 39,984 interactions | 0.004 m | 0 | no |
| Emerging gait, 79,968 interactions | step/arm exploration | — | no |
| Coordinated pull, 199,920 interactions | arm-assisted ascent | 20 | **yes** |
| Selected checkpoint, 219,912 interactions | 1.5 m target | 20–22 | **yes** |

The panels run at the same 50 Hz policy rate and display live slope distance,
step count, grip score, and a 1.5 m progress bar. Companion measurements are in
`videos/g1_fixed_line_learning_progress.json`.

## Hugging Face Jobs

The current three-action arm/step policy was trained locally on CPU because
MuJoCo rollout and iterative IK are CPU-bound; using the available A100 credit
would not accelerate this workload materially. Earlier two-action experiments
used CPU Performance Hugging Face Jobs:

- Initial inclined run: `iteratehack/6a932b8245686a1580c15c27`, 543 seconds.
  Its checkpoint was retained but its original permissive random-action and
  ablation report was rejected.
- Force-free grounded fine-tune: `iteratehack/6a932ea2984507d9db4ebe3a`,
  265 seconds, completed with every cloud gate passing.
- Current training completed 304,368 cumulative interactions; the best
  deterministic checkpoint was selected at 219,912 interactions.
- Private artifacts: `hf://buckets/npow/himalaya-fixed-line`.

## Reproduce

```bash
uv run --with-requirements requirements.txt python evaluate_fixed_line.py \
  --model models/ppo_fixed_line_slope/g1_fixed_line_final.zip \
  --episodes 10 \
  --output models/ppo_fixed_line_slope/evaluation_report.json

uv run --with-requirements requirements.txt python record_fixed_line.py \
  --model models/ppo_fixed_line_slope/g1_fixed_line_final.zip \
  --output videos/g1_fixed_line_mountain_v2.mp4

uv run --with-requirements requirements.txt python record_fixed_line_learning.py \
  --stage 'Early policy=models/ppo_fixed_line_slope/checkpoints/g1_fixed_line_39984_steps.zip' \
  --stage 'Emerging gait=models/ppo_fixed_line_slope/checkpoints/g1_fixed_line_79968_steps.zip' \
  --stage 'Selected policy=models/ppo_fixed_line_slope/g1_fixed_line_final.zip'

uv run --with-requirements requirements.txt python scripts/validate_fixed_line_grasp.py

uv run --with-requirements requirements.txt python evaluate_mountain_recovery.py

MUJOCO_GL=egl uv run --with-requirements requirements.txt \
  python record_fixed_line_recovery.py \
  --output videos/g1_fixed_line_fall_recovery.mp4
```

## Scope

This remains a prepared-route controller. PPO learns the alternating uphill
step requests, right-arm pull magnitude, and the four-group fall-to-climb
handoff; low-level position control executes each leg stroke, IK maintains the
prepared ascender grip, and a pinned pretrained WBC supplies the base get-up
motion vocabulary. It does not learn autonomous initial rope attachment,
unseen anchor passing, arbitrary-fall recovery, or hardware-safe control.
