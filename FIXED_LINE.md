# Unitree G1 Inclined Fixed-Line Travel — Stage 1 Handoff

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
- Learning-over-time video: `videos/g1_fixed_line_learning_progress.mp4`
- Selected checkpoint: `models/ppo_fixed_line_slope/g1_fixed_line_final.zip`
- Ten-reset mountain-scene evaluation: `models/ppo_fixed_line_slope/evaluation_report_mountain_v2.json`
- Rollout telemetry: `videos/g1_fixed_line_mountain_v2.json`
- Environment: `fixed_line_slope_env.py`
- Scene: `assets/unitree_g1/scene_fixed_line_slope.xml`

The recorder calls the saved PPO policy on every frame. The learned videos do
not use the evaluator's scripted reference action.

## Verified result

The local acceptance suite uses ten independently perturbed resets for every
learned/control condition and five for the reference mechanics check.

| Condition | Success | Along-slope travel | Vertical gain | Steps | Grounded |
|---|---:|---:|---:|---:|---:|
| Learned PPO policy | **100%** | **1.511 m** | **0.702 m** | **46.4** | **94.2%** |
| Same policy, step actuation disabled | 0% | 0.003 m | -0.095 m | 0.0 | 100% |
| Same policy, line and step actuation disabled | 0% | -0.551 m | -0.235 m | 0.0 | 100% |
| Neutral action | 0% | 0.003 m | -0.095 m | 0.0 | 100% |
| Uniform-random action | 0% | 0.107 m | 0.006 m | 7.7 | 99.1% |

All executable gates pass. Learned episodes average 53.9% left-boot contact,
51.5% right-boot contact, and 11.2% double support. The longest interval with
neither boot registered is one 20 ms policy step. The recorded deterministic
mountain rollout reaches 1.519 m in 1,080 steps, peaks at 587 N measured ground
load, and has no hand-hand or hand-slope collisions.

Disabling the alternating step mechanism while retaining the safety line
reduces progress from 1.510 m to 0.004 m. Removing the line causes a 0.55 m
downslope slide and failure. Random actions cannot repeatedly satisfy the
four-frame, correctly alternating step request and never finish.

## Terrain, rope, hands, and feet

The MuJoCo plane is rotated 28° above horizontal. The orange handline follows
the same direction and remains roughly knee-to-waist high normal to the local
surface; it is not vertical. The scene uses subtle snow/ice texture, a harness,
leg loops, lanyard, one-way capture device, handled ascender, and visual-only
crampon plates and points attached to the actual ankle bodies.

The right wrist is retained on the handled ascender with seven-joint
world-frame damped-least-squares IK. Curled distal finger links remain close to
and periodically span both sides of the handle centerline. The left arm is not
forced across the torso: it stays bent, open, and clear for balance. Across the
ten learned rollouts, the right digit-center/device error stays below 12.8 cm,
hand separation stays above 37.7 cm, and there are no hand collisions.

The ankle pitch is now four hundredths of a radian more downslope than the
original render, keeping the front points visibly buried instead of leaving an
ambiguous gap at the snow. The old oversized visual plates were replaced by
narrow rails, crossbars, and short points that follow the physical boot frame.

The two policy actions request left and right uphill steps. A request must be
deliberate and persist for four policy frames; then the low-level leg target
executes an 18-frame swing/stance stroke. The policy must read the expected
side from observation and alternate requests. Each boot's stock MuJoCo contact
geometries and surface friction generate the travel.

There is **no auxiliary uphill force**. Position-actuated leg movement resolves
through real boot/slope contact. A one-way tangential rope catch prevents lost
height but does not apply commanded uphill drive. A capped low-level balance
assist controls lateral drift, torso rotation, and at most 100 N in the surface
normal direction; it cannot advance the robot along the slope.

## Learning progression

`videos/g1_fixed_line_learning_progress.mp4` synchronizes four deterministic
policies from the same reset under the final force-free mechanics:

| Policy stage | Along-slope travel | Steps | Success |
|---|---:|---:|---:|
| Initialized network, 0 interactions | 0.004 m | 0 | no |
| Early checkpoint, 39,984 interactions | 0.004 m | 0 | no |
| Emerging gait, 79,968 interactions | 1.333 m | 43 | no |
| Selected checkpoint, 259,896 interactions | 1.510 m | 44 | **yes** |

The panels run at the same 50 Hz policy rate and display live slope distance,
step count, grip score, and a 1.5 m progress bar. Companion measurements are in
`videos/g1_fixed_line_learning_progress.json`.

## Hugging Face Jobs

Training used CPU Performance Hugging Face Jobs because MuJoCo rollout and IK
are CPU-bound:

- Initial inclined run: `iteratehack/6a932b8245686a1580c15c27`, 543 seconds.
  Its checkpoint was retained but its original permissive random-action and
  ablation report was rejected.
- Force-free grounded fine-tune: `iteratehack/6a932ea2984507d9db4ebe3a`,
  265 seconds, completed with every cloud gate passing.
- Selected checkpoint: 259,896 total interactions.
- Approximate corrected-scene compute cost: $0.43 at $1.90/hour.
- Private artifacts: `hf://buckets/npow/himalaya-fixed-line`.

## Reproduce

```bash
source .venv/bin/activate

python evaluate_fixed_line.py \
  --model models/ppo_fixed_line_slope/g1_fixed_line_final.zip \
  --episodes 10 \
  --output models/ppo_fixed_line_slope/evaluation_report.json

MUJOCO_GL=egl python record_fixed_line.py \
  --model models/ppo_fixed_line_slope/g1_fixed_line_final.zip \
  --output videos/g1_fixed_line_mountain_v2.mp4

MUJOCO_GL=egl python record_fixed_line_learning.py \
  --stage 'Early policy=models/ppo_fixed_line_slope/checkpoints/g1_fixed_line_39984_steps.zip' \
  --stage 'Emerging gait=models/ppo_fixed_line_slope/checkpoints/g1_fixed_line_79968_steps.zip' \
  --stage 'Selected policy=models/ppo_fixed_line_slope/g1_fixed_line_final.zip'

MUJOCO_GL=egl python scripts/validate_fixed_line_grasp.py
```

## Scope

This is still a prepared-position Stage 1 controller. PPO learns the alternating
uphill step requests; low-level position control executes each leg stroke and
IK maintains the prepared ascender grip. It does not yet learn autonomous rope
attachment, hand acquisition/regrasping, obstacle negotiation, anchor passing,
or recovery from a large fall. Those should remain separate later stages.
