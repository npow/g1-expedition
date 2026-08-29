# Icefall G1

A learned MuJoCo PPO policy that teaches a Unitree G1 to plant and load an
ice axe during oblique, cross-slope mountaineering falls.

This is reinforcement learning, not a keyframed demonstration. One saved PPO
checkpoint controls all 14 arm joints at 100 Hz across every recorded
scenario. The task audits physical pick contact, blade angle, visible plant
motion, opposing-digit grasp contacts, and axe placement.

## Result

- **9/9** named oblique and mixed-sign scenarios arrest with all non-axe body
  friction disabled.
- **60/60** unseen randomized falls arrest over heading ±40°, cross-slope
  velocity ±1.5 m/s, body roll ±10°, and downhill speed 4–5 m/s.
- With physical pick collision disabled, **0/39** tests arrest and the robot
  accelerates to approximately 13.53 m/s.

The full evidence video includes a fixed wide camera, live hand/pick view,
slope-normal view, scenario parameters, and contact telemetry:

[Watch the diverse self-arrest suite](videos/g1_self_arrest_diverse_suite.mp4)

For exact scenario tables, acceptance gates, causal ablations, training
history, hashes, and limitations, see [HANDOFF.md](HANDOFF.md).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python evaluate_diverse_policy.py \
  --model models/ppo_self_arrest/g1_self_arrest_final.zip \
  --randomized-episodes 60 \
  --seed 73000 \
  --output models/ppo_self_arrest/diverse_evaluation_report.json

MUJOCO_GL=egl python record_diverse_scenarios.py \
  --model models/ppo_self_arrest/g1_self_arrest_final.zip \
  --output videos/g1_self_arrest_diverse_suite.mp4 \
  --seed 81000
```

## Repository map

- `himalaya_env.py` — MuJoCo task, observations, rewards, success gates, and
  causal ablations.
- `train.py` — PPO curriculum and adversarial reset-anchor training.
- `diverse_scenarios.py` — named scenarios and continuous reset envelope.
- `evaluate_diverse_policy.py` — fixed, randomized, friction, snow, and
  pick-disabled audits.
- `record_diverse_scenarios.py` — multi-view evidence renderer.
- `assets/` — G1, axe, scene, and natural snow assets.
- `models/ppo_self_arrest/` — canonical policy and audit reports.

## Scope and safety

This is a simulation-only hierarchical skill. The initial prone pose, fingers,
legs, and waist use low-level control; PPO learns the two-arm plant and load.
The axe remains rigidly retained at the right palm. This is not evidence of
learned grasp acquisition, tumbling recovery, hardware readiness, or real
mountaineering safety. Do not deploy it on a robot without a separate,
conservative sim-to-real validation and safety program.

The Unitree G1 model under `assets/unitree_g1/` retains its BSD-3-Clause
license and attribution.
