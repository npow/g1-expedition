# G1 Expedition

Four simulation-first learned and hierarchical skills for the Unitree G1:
mountaineering self-arrest, fixed-line ascent, and cooperative disaster-recovery
transport, plus physical fall recovery.

## Scenarios

| Scenario | Simulator / learning | Result and evidence |
|---|---|---|
| **Ice-axe self-arrest** | MuJoCo + PPO; all 14 arm joints | 9/9 named oblique falls and 60/60 unseen randomized falls arrest with non-axe body friction disabled. [Video](videos/g1_self_arrest_diverse_suite.mp4) · [technical handoff](HANDOFF.md) |
| **Fixed-line ascender travel + fall recovery** | MuJoCo deformable rope + climbing PPO + slope-recovery PPO over a pinned WBC prior | 10/10 ascent resets travel 1.54 m. The integrated rollout climbs, takes a finite physical fall, is arrested, gets up, re-grasps the lower Jumar, then completes two continuous 1.5 m ascent segments: 3.070 m uphill and net +1.981 m after the grounded transition. A fixed above-slope camera makes the world translation visible. The braided 33-node line records zero core/leg or hand-through-rope penetration. [Integrated video](videos/g1_fixed_line_fall_recovery.mp4) · [technical handoff](FIXED_LINE.md) |
| **Physical fall recovery** | MuJoCo contacts + pretrained RL whole-body controller | 4/4 finite-push falls settle on the ground and recover with zero floating-base teleports. Reference-only and motors-off ablations both fail. [Video](videos/g1_physical_getup.mp4) · [technical handoff](GETUP.md) |
| **Cooperative rescue transport** | Isaac Lab + parameter-shared MAPPO over frozen AGILE locomotion | Variable teams of 2/3/5 G1s, physical tension-only slings, crate/timber/girder profiles, tests, and compact crate/timber pilot checkpoints. [Project documentation](cooperative_beam_isaaclab/README.md) |

All videos call their saved policies during physics rollout; they are not
keyframed demonstrations. Each scenario documents its hierarchical boundary
instead of presenting low-level prepared controllers as learned behavior.

## Repository map

```text
HANDOFF.md                         self-arrest results, ablations, and limits
FIXED_LINE.md                      ascender results, ablations, and limits
himalaya_env.py                    MuJoCo self-arrest environment
fixed_line_slope_env.py            MuJoCo inclined fixed-line environment
mountain_recovery.py               fall, tether, slope get-up, and re-grasp
getup_controller.py                physical fall + pretrained RL WBC adapter
train_mountain_recovery.py         PPO slope-recovery training entry point
train.py / train_fixed_line.py     self-arrest and ascent PPO entry points
evaluate_*.py                      causal and policy evaluations
record_*.py                        evidence-video renderers
assets/                            G1, axe, rope/ascender, and terrain assets
models/ppo_self_arrest/            canonical self-arrest PPO checkpoint
models/ppo_fixed_line_slope/       canonical ascender PPO run/checkpoints
models/ppo_mountain_recovery/      selected integrated recovery PPO + report
cooperative_beam_isaaclab/         isolated Isaac Lab cooperative MAPPO project
videos/                            final policy evidence and telemetry
```

Large videos and learned checkpoints use Git LFS.

## MuJoCo setup

The self-arrest and ascender scenarios use the root environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Evaluate and render self-arrest:

```bash
python evaluate_diverse_policy.py \
  --model models/ppo_self_arrest/g1_self_arrest_final.zip \
  --randomized-episodes 60 --seed 73000 \
  --output models/ppo_self_arrest/diverse_evaluation_report.json

MUJOCO_GL=egl python record_diverse_scenarios.py \
  --model models/ppo_self_arrest/g1_self_arrest_final.zip \
  --output videos/g1_self_arrest_diverse_suite.mp4 --seed 81000
```

Evaluate and render fixed-line ascent:

```bash
python evaluate_fixed_line.py \
  --model models/ppo_fixed_line_slope/g1_fixed_line_final.zip \
  --episodes 10 \
  --output models/ppo_fixed_line_slope/evaluation_report.json

MUJOCO_GL=egl python record_fixed_line.py \
  --model models/ppo_fixed_line_slope/g1_fixed_line_final.zip \
  --output videos/g1_fixed_line_mountain_v2.mp4

python evaluate_mountain_recovery.py

MUJOCO_GL=egl python record_fixed_line_recovery.py \
  --output videos/g1_fixed_line_fall_recovery.mp4
```

Evaluate and render physical fall recovery:

```bash
python scripts/fetch_getup_assets.py
python evaluate_getup.py --output models/wbc_getup/evaluation_report.json

MUJOCO_GL=egl python record_getup.py \
  --output videos/g1_physical_getup.mp4 \
  --report videos/g1_physical_getup.json
```

The whole-body get-up prior is a pinned Apache-2.0 pretrained artifact from
`wbc-mjlab/wbc-g1-deploy`. The four-action slope transfer/handoff PPO is
trained in this repository (911,008 cloud interactions plus a 50,176-step
selection audit). See [`FIXED_LINE.md`](FIXED_LINE.md) and [`GETUP.md`](GETUP.md)
for the exact learned/control boundary.

## Isaac Lab setup

Cooperative transport is intentionally isolated because it requires Isaac Sim,
Isaac Lab, skrl, and a CUDA-capable local system. See
[`cooperative_beam_isaaclab/README.md`](cooperative_beam_isaaclab/README.md)
for installation, smoke tests, MAPPO training/evaluation, and checkpoint usage.

The project includes simulator-independent tests:

```bash
cd cooperative_beam_isaaclab
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
```

## Scope and safety

These are simulation research tasks, not deployment-ready robot skills or
mountaineering/rescue safety systems. Prepared poses, low-level controllers,
tool retention, frozen locomotion, or slings are explicitly documented for
each scenario. Hardware use requires a separate conservative sim-to-real,
fall-protection, load-rating, and emergency-stop program.

The Unitree G1 model under `assets/unitree_g1/` retains its BSD-3-Clause
license and attribution.
