# G1 Expedition

**Optimus Prime's Unitree G1 stack for alpine failure recovery:** ice-axe
self-arrest, fixed-line ascent and fall recovery, controlled rappel evidence,
and cooperative rescue-load transport. Single-robot skills use MuJoCo; learned
multi-robot transport uses Isaac Sim/Isaac Lab and shares compact teammate state
through LiveKit during deployed inference.

## Hackathon submission

- [Two-minute demo video](submission/optimus_prime_g1_expedition_2min.mp4)
  with [captions](submission/optimus_prime_g1_expedition_2min.srt).
- [90-second pitch deck](submission/optimus_prime_pitch_90s.pptx),
  [PDF](submission/optimus_prime_pitch_90s.pdf), and
  [timed speaker script](submission/pitch_script_90s.md).
- [Reproducible submission package](submission/README.md) containing source
  clips, theme assets, narration, and the complete FFmpeg/ElevenLabs/PptxGenJS
  build pipeline.

## Scenarios

| Scenario | Simulator / learning | Result and evidence |
|---|---|---|
| **Ice-axe self-arrest** | MuJoCo + Stable-Baselines3 PPO; 125 observations → 14 arm-joint targets; 256×256 actor/critic MLPs | 9/9 named oblique falls and 60/60 unseen randomized falls arrest with non-axe body friction disabled. [Video](videos/g1_self_arrest_diverse_suite.mp4) · [technical handoff](HANDOFF.md) |
| **Fixed-line ascender travel + fall recovery** | MuJoCo deformable rope; 114→3 ascent PPO plus 145→4 recovery/handoff PPO over a pinned 29-DoF WBC prior | 10/10 ascent resets travel 1.54 m. The integrated rollout climbs, takes a finite physical fall, is arrested, gets up, re-grasps the lower Jumar, then completes two continuous 1.5 m ascent segments: 3.070 m uphill and net +1.981 m after the grounded transition. The braided 33-node line records zero core/leg or hand-through-rope penetration. [Integrated video](videos/g1_fixed_line_fall_recovery.mp4) · [technical handoff](FIXED_LINE.md) |
| **Physical fall recovery** | MuJoCo contacts + pretrained RL whole-body controller | 4/4 finite-push falls settle on the ground and recover with zero floating-base teleports. Reference-only and motors-off ablations both fail. [Video](videos/g1_physical_getup.mp4) · [technical handoff](GETUP.md) |
| **Controlled rappel** | Final evidence media is packaged under `submission/source_videos/`; its trainer and checkpoint were not present in this checkout | The submission clip shows a 2.00 m controlled descent and reports 7/10 randomized starts. The repository deliberately does not invent simulator/model provenance that is absent from the source package. |
| **Cooperative rescue transport** | Isaac Sim 6 + Isaac Lab 3; parameter-shared MAPPO over frozen AGILE locomotion; LiveKit DataTracks at inference | Variable teams of 2/3/5 G1s, physical tension-only slings, crate/timber/girder profiles, compact pilot checkpoints, and one-state-uplink-per-robot LiveKit fan-out. [Project documentation](cooperative_beam_isaaclab/README.md) |

All repository-generated evidence videos call their saved policies during
physics rollout; they are not keyframed demonstrations. Each scenario
documents its hierarchical boundary instead of presenting low-level prepared
controllers as learned behavior.

## Technical architecture

| Capability | Physics and cadence | Learned model | Policy contract and RL technique |
|---|---|---|---|
| Self-arrest | MuJoCo contact-rich 35° ice slope; axe pick is the only high-friction arrest path | Stable-Baselines3 PPO, separate 256×256 Tanh actor/value MLPs; selected at 2,169,252 interactions | 125 observations → 14 joint-position residuals; GAE(0.95), γ=0.995, clip 0.2, 8 epochs |
| Fixed-line ascent | MuJoCo at 2 ms physics / 50 Hz policy with a collision-bearing 33-node flex rope, Jumar ratchet, boot contacts, and equal/opposite rope forces | PPO, 256×256 actor/value MLPs; selected at 219,912 interactions | 114 observations → 3 actions: step phase/stride and loaded-arm pull; 12 parallel workers |
| Fall recovery | Same MuJoCo slope and rope; 50 Hz pretrained WBC with ten 2 ms substeps | Four-action PPO handoff over the pinned `wbc-mjlab/wbc-g1-deploy` 29-DoF ONNX policy; selected at 599,920 interactions | 145 observations → group-wise WBC braking/handoff actions; reference-only and motors-off causal ablations |
| Standalone get-up | MuJoCo contact dynamics, per-motor torque caps, no floating-base pose reset | Pinned pretrained 132-observation → 29-action residual WBC plus motion reference | RL policy inference at 50 Hz; adapted and evaluated here, not retrained here |
| Cooperative lift | Isaac Sim 6.0.1 / Isaac Lab 3.0, 200 Hz physics / 50 Hz policy, dynamic payload and unilateral spring-damper slings | skrl parameter-shared MAPPO: 98-value local encoder, 7-value teammate tokens, 128-D four-head attention; centralized 768→512→256 critic | 10 actions/G1: `[vx, vy, yaw_rate, hip_height]` + two 3-D wrist offsets. Frozen AGILE maps the locomotion command and 83-value body state to 12 leg targets; batched DLS IK drives the arms |

The demo reel's overload, lift, and abort clips come from the MuJoCo
`mountain-robotics` control baseline. The Isaac Lab project is the learned,
variable-team MAPPO system and the deployment path that uses LiveKit.

PPO/MAPPO checkpoints are selected and evaluated as frozen artifacts. The
reported pass rates come from denominator-bearing evaluation reports under
`models/`, not from training return alone.

## LiveKit inference state bus

Training uses in-process tensors because Isaac Lab runs hundreds of vectorized
worlds on one GPU. Deployment uses a different transport boundary: every G1
publishes one 63-byte coordination frame per inference tick to a LiveKit room.
The room's SFU fans that state out to all policy participants, replacing an
`N × N` robot-connection graph with one uplink per robot.

Each frame contains:

```text
robot id · sequence · timestamp · world pose · linear velocity · load ratio
```

The receiving process converts fresh state into the exact relative-pose,
relative-velocity, and load-share teammate tokens expected by the frozen MAPPO
actor. Out-of-order packets are rejected. State older than the configured
deadline is zero-filled while local proprioception and frozen AGILE control
continue, so a thin satellite link does not block the local servo loop.

The transport is implemented in
[`livekit_state_bus.py`](cooperative_beam_isaaclab/src/cooperative_beam_isaaclab/tasks/livekit_state_bus.py)
and exposed as a deployment bridge in
[`livekit_state_bridge.py`](cooperative_beam_isaaclab/scripts/livekit_state_bridge.py).
It uses LiveKit's named DataTracks, matching the
[hackathon's LiveKit guidance](https://www.livekit.info/himalaya-robotics-hack)
for robot telemetry and dependable remote communications.

```bash
cd cooperative_beam_isaaclab
uv sync --extra livekit

# Mint one room token per identity, then run one bridge beside each policy.
LIVEKIT_URL=wss://your-project.livekit.cloud \
LIVEKIT_TOKEN="$G1_0_TOKEN" \
python scripts/livekit_state_bridge.py \
  --robot-id g1_0 --team g1_0,g1_1,g1_2
```

The bridge accepts JSON Lines from the local estimator and returns the complete
checkpoint-compatible actor observation. Credentials remain environment-only
and are never stored in the repository.

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
  scripts/livekit_state_bridge.py  deployed LiveKit state transport process
  .../tasks/livekit_state_bus.py   binary codec, freshness gate, actor adapter
videos/                            final policy evidence and telemetry
submission/                        final demo, pitch deck, source clips, rebuild pipeline
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
Isaac Lab, skrl, and a CUDA-capable local system. LiveKit is an optional
deployment extra and is not required for vectorized training. See
[`cooperative_beam_isaaclab/README.md`](cooperative_beam_isaaclab/README.md)
for installation, smoke tests, MAPPO training/evaluation, and checkpoint usage.

The project includes simulator-independent tests:

```bash
cd cooperative_beam_isaaclab
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
```

The Unitree G1 model under `assets/unitree_g1/` retains its BSD-3-Clause
license and attribution.

## Related work

[abhijitbetigeri/HimalayaExpedition](https://github.com/abhijitbetigeri/HimalayaExpedition)
— Robotic Expedition in Himalayas. Companion project covering the wider
Himalayan robotics track: ice/snow locomotion under domain randomization, wind
loading, and fixed-line ascent on MuJoCo Playground (MJX/GPU), alongside the
LiveKit voice interface used in this demo.

