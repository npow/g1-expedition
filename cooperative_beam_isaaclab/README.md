# Cooperative G1 disaster-recovery transport in Isaac Lab

This is an isolated, simulation-only Isaac Lab project. Variable-size Unitree G1 teams cooperatively lift a
recognizable rescue payload, carry it 0.85 m while turning 20 degrees, and place it in a marked drop zone.

The registered examples deliberately vary object scale, mass, and team size:

| Task | Scenario | Robots | Mass curriculum |
|---|---|---:|---:|
| `Isaac-Cooperative-G1-Rescue-Crate-Direct-v0` | dense rescue-equipment crate | 2 | 4–10 kg |
| `Isaac-Cooperative-G1-Timber-Direct-v0` | 2.2 m fallen timber | 3 | 8–18 kg |
| `Isaac-Cooperative-G1-Footbridge-Girder-Direct-v0` | 3.25 m collapsed steel girder | 5 | 15–30 kg |

## What is implemented

- Two, three, or five independently simulated Isaac Lab `G1_29DOF_CFG` articulations.
- NVIDIA's frozen AGILE G1 loco-manipulation policy, fetched through Isaac Lab at runtime. One stacked
  inference maps every robot's high-level locomotion commands to its 12 leg targets; the policy file is
  not copied into this repository.
- Ten learned actions per robot: `[vx, vy, yaw_rate, hip_height]` plus a 3D position offset for each wrist.
  Isaac Lab's batched damped-least-squares IK converts wrist commands to the two seven-joint arm targets.
- A dynamic payload rigid body with randomized mass, friction, inertia, gravity, and contact response.
- Two tension-only rescue slings per robot, attached to the actual G1 palm links. The sling forces are
  applied to each palm and as equal-and-opposite force/torque on the payload. This is a physical coupling,
  not a kinematic animation or payload teleport.
- Long-axis stations alternate between opposite payload sides. Each G1 faces inward and its two slings attach to
  the near edge without crossing, avoiding the same-side crowding that destabilizes larger teams.
- A smooth lift → carry/turn → lower trajectory and a marked drop zone.
- A vectorized `DirectMARLEnv`; actor and central-state widths are derived from team size.
- Team rewards for payload tracking, levelness, heading, upright robots, and balanced load sharing.
- Failure detection for falls, drops, and excessive sling extension, plus success and load diagnostics.
- A parameter-shared MAPPO actor and centralized critic. The actor cross-attends from local state to a variable
  number of exchangeable teammate tokens, following TeamHOI's coordination structure. IPPO remains an ablation.
- A LiveKit deployment transport that publishes one compact state stream per robot and reconstructs the same
  teammate-token observation expected by the frozen actor, without an all-to-all robot connection graph.
- A staged curriculum sized to the actual 10,000-iteration run: learn lift/level first, gradually add
  transport/turning over 24k–150k vector control steps, and increase mass to 18 kg over 180k steps.

## How the robots coordinate

Each actor sees its own proprioception, payload state and target, both local sling vectors, its load share, and
compact messages containing every other robot's relative pose, relative velocity, and load share. All actors receive
the same team reward. A single shared actor processes every station; teammate order is removed by its attention
operation. During MAPPO training, the shared critic additionally sees the concatenated team state and payload mass;
at inference, each actor needs only its local/message observation.

For odd team sizes, equal per-robot tension would put more total force on the side with more stations and roll the
payload. The load target therefore assigns half of the total support to each side and divides that half among the
robots on that side (for three robots, `25% / 50% / 25%`). No-tension slings receive no load-balancing reward.

Coordination therefore comes from four channels:

1. Mechanical coupling through the common payload and slings.
2. One shared policy trained on experience from every station and a shared objective that rewards balanced tension.
3. Explicit low-bandwidth teammate tokens, while privileged global state is limited to the training critic.
4. At deployed inference, a LiveKit room fans each robot's single state uplink to every teammate process.

The learned layer does not rediscover walking. The frozen AGILE controller observes the full 29-DoF body, including
the loaded arms, and stabilizes the legs from the four high-level commands. This is the main sample-efficiency gain.

This is a prepared-sling transport task. It intentionally avoids making dexterous grasp discovery the first
bottleneck. A later experiment can replace the sling model with contact-only grasping without changing the MARL
interface.

## LiveKit deployment topology

Vectorized training keeps teammate tensors in-process. Deployment preserves the actor contract but replaces that
memory boundary with a LiveKit named DataTrack:

```text
g1_0 -- 63-byte state --\
g1_1 -- 63-byte state ----> LiveKit room fan-out ----> teammate-token adapter ----> shared MAPPO actor
g1_N -- 63-byte state --/
```

Every packet carries robot identity, sequence, timestamp, world pose, linear velocity, and measured load ratio.
Each receiver converts the other robots into recipient-frame relative position, relative velocity, and load-share
tokens. Out-of-order packets are rejected; missing or stale teammate state is zero-filled, so local proprioception
and the frozen AGILE servo path do not wait on the network. Each robot maintains one state uplink to the room rather
than direct links to every other robot.

The binary codec, freshness gate, and checkpoint-compatible observation adapter live in
[`livekit_state_bus.py`](src/cooperative_beam_isaaclab/tasks/livekit_state_bus.py). Run
[`livekit_state_bridge.py`](scripts/livekit_state_bridge.py) beside each deployed policy process:

```bash
uv sync --extra livekit

LIVEKIT_URL=wss://your-project.livekit.cloud \
LIVEKIT_TOKEN="$G1_0_TOKEN" \
python scripts/livekit_state_bridge.py \
  --robot-id g1_0 --team g1_0,g1_1,g1_2
```

The bridge reads local estimator state as JSON Lines on standard input. If the line also includes the actor's
98-value local observation, it returns the full frozen-checkpoint observation; otherwise it returns just the fresh
teammate fields and freshness mask. One room token should be minted per canonical identity (`g1_0`, `g1_1`, ...).
This follows the hackathon's official
[LiveKit robotics guidance](https://www.livekit.info/himalaya-robotics-hack): WebRTC transport for robot telemetry,
remote inference, operator video, and voice, without direct VPN or port-forwarding dependencies.

## Local setup and validation

The tested installation on Odin is:

- Isaac Sim 6.0.1 / current Isaac Lab 3.0 development checkout
- Python 3.12
- RTX PRO 6000 Blackwell (96 GB)
- skrl 2.1.0

Set paths if your checkout differs:

```bash
export ISAACLAB_ROOT=/home/npow/isaac-validation/IsaacLab
export ISAACLAB_VENV=/home/npow/isaac-validation/.isaac-venv
source "$ISAACLAB_VENV/bin/activate"
uv pip install -e .
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
```

Run the fast simulator-independent tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
```

Run a finite GPU physics smoke test:

```bash
python scripts/smoke.py --num_envs 4 --steps 380 --random_actions --viz none
```

Exercise the whole hierarchical lift path—high-level wrist command, IK, sling tension, payload motion, and frozen
locomotion—without a learned checkpoint:

```bash
python scripts/smoke.py --num_envs 4 --steps 150 --scripted_lift --viz none
```

Exercise lift, carry, turn, and lowering with a non-learned formation controller. This is a physics-feasibility
baseline, not a policy benchmark:

```bash
python scripts/smoke.py --num_envs 4 --steps 600 --scripted_transport --viz none
```

`--pretension 0.05` is a useful diagnostic: it shortens the slings after calibration so nonzero cable forces
can be checked without a trained policy.

The same geometry can be screened at a different team size and fixed mass without defining another task:

```bash
python scripts/smoke.py \
  --task Isaac-Cooperative-G1-Timber-Direct-v0 \
  --team_size 5 --payload_mass 30 --steps 180 --scripted_lift --viz none
```

Run one complete MAPPO update:

```bash
python scripts/train.py \
  --task Isaac-Cooperative-G1-Timber-Direct-v0 \
  --algorithm MAPPO --num_envs 8 --max_iterations 1 --viz none
```

## Included pilot checkpoints

The repository includes compact extractions of the best profile-specific
crate/2-G1 and timber/3-G1 pilot checkpoints under `artifacts/`, together with
the exact environment and skrl agent parameter snapshots used by each run:

```text
artifacts/crate_pilot/best_agent.pt
artifacts/timber_pilot/best_agent.pt
```

These are pilot baselines, not a claim of cross-geometry or cross-team-size
generalization. The footbridge-girder task is implemented but has no selected
checkpoint in this workspace. Checksums and provenance are recorded in
[`artifacts/README.md`](artifacts/README.md).

Start a normal local run:

```bash
NUM_ENVS=256 scripts/run_local.sh --max_iterations 10000
```

Training outputs are written to `logs/skrl/cooperative_g1_payload/`.

MAPPO parameter sharing is on by default. Set `COOP_PARAMETER_SHARING=0` only for the independent-policy ablation.

Evaluate a frozen same-team checkpoint for two complete episodes per environment:

```bash
python scripts/evaluate.py \
  --task Isaac-Cooperative-G1-Timber-Direct-v0 \
  --checkpoint /path/to/agent_48000.pt \
  --num_envs 64 --steps 1200 --payload_mass 8 --transport_scale 1.0 --viz none
```

Unlike a generic play script, `evaluate.py` pins the transport curriculum scale. A fresh environment otherwise starts
at scale zero and tests only lifting in place. The evaluator prints one machine-readable `[EVAL_RESULT]` JSON record
containing success, termination reasons, target error, cooperative time, jerk, load variation, and peak tension.
Use `--actor_only --team_size N` for zero-shot transfer of the team-size-independent actor; the centralized critic is
intentionally not loaded across team sizes.

## Hugging Face Jobs

The launch script uses the public NVIDIA Isaac Lab container and mounts only this project's `src/` and `scripts/`
directories through Hugging Face Jobs, excluding local logs and caches. It defaults to the `iteratehack` namespace
(override with `HF_NAMESPACE`), a single RTX PRO 6000, 512 environments, and 10,000 MAPPO iterations. The local
Hugging Face token is passed as a Job secret so the final checkpoint tarball can be persisted to the private
`jobs-artifacts` bucket:

```bash
HF_TASK=Isaac-Cooperative-G1-Timber-Direct-v0 \
HF_NUM_ENVS=512 HF_ITERATIONS=10000 scripts/hf_job.sh
```

`HF_ITERATIONS` is the training stopping criterion. Each iteration collects the configured 24 vector control steps,
so the default run reaches 240,000 curriculum steps. Hugging Face requires a platform timeout, so the script uses
a deliberately generous 30-day backstop by default; `HF_TIMEOUT` remains available for infrastructure safety but
is not chosen from a dollar budget. Inspect and cancel with:

```bash
uv tool run --from 'huggingface_hub>=0.35' hf jobs ps
uv tool run --from 'huggingface_hub>=0.35' hf jobs logs -f JOB_ID
uv tool run --from 'huggingface_hub>=0.35' hf jobs cancel JOB_ID
```

Download and extract a completed run by its job name:

```bash
scripts/hf_download_artifacts.sh cooperative-g1-timber-pilot-2k
```

Run the same finite checkpoint evaluation remotely:

```bash
HF_JOB_NAME=cooperative-g1-timber-eval \
HF_TASK=Isaac-Cooperative-G1-Timber-Direct-v0 \
HF_CHECKPOINT=/absolute/path/to/agent_48000.pt \
HF_TEAM_SIZE=3 HF_PAYLOAD_MASS=8 HF_TRANSPORT_SCALE=1.0 \
scripts/hf_eval_job.sh
```

Do not use the H200 by default: this environment benefits from parallel simulator capacity, and the RTX PRO 6000
is the more budget-efficient first experiment.

## Scaling and generalization protocol

Following TeamHOI, the decisive result should come from one shared actor evaluated across conditions, not from
comparing unrelated object-specific policies. The three initial pilots are deliberately profile-specific baselines:
crate/2-G1, timber/3-G1, and girder/5-G1 each train a separate actor. They measure whether each task can learn at all,
but cannot demonstrate transfer because object geometry, mass range, and team size all change together.

`scripts/benchmark_matrix.py` produces an explicit geometry × team-size × mass manifest with these non-overlapping
splits:

- Training support: rescue crate and fallen timber; 2, 3, or 5 robots; 2, 4, or 6 kg per robot.
- Geometry holdout: the footbridge girder is never used to update the unified actor.
- Team-size holdouts: 4 robots test interpolation between trained counts, while 6 robots test extrapolation and
  crowding; neither count is used to update the unified actor.
- Mass holdout: 9 kg per robot is never used to update the unified actor and is an overload stress test, not a claim
  about real G1 capacity.
- Compound holdouts combine two or three unseen factors, such as a 4-G1, 36 kg girder.

Holding mass per robot fixed while varying team size separates coordination overhead from simply making each robot's
share lighter. Holding total mass fixed while varying team size should be reported as an additional mechanical-scaling
slice during evaluation. The manifest's command column is only a scripted physics smoke command; learned-policy
evaluation must use the frozen checkpoint and must never update it on held-out rows.

Report the four TeamHOI-style task metrics already logged by the environment:

1. Episode success rate and final payload-to-target distance.
2. Cooperative time ratio: fraction of transport steps when every robot carries non-trivial sling tension.
3. Mean payload jerk, measuring transport smoothness.
4. Mean transport load coefficient of variation and episode peak sling tension, measuring whether extra robots
   actually share load and whether any arm experiences a dangerous transient.
5. Payload kilograms per robot and per arm, reported as normalized simulation loads rather than hardware ratings.

Use the three nominal profiles for environment checks, train the shared actor across the training profiles/team sizes,
then reserve at least one geometry and the 9 kg/robot row for zero-shot evaluation. Also report payload kilograms per
robot and per arm. Unitree's arm-load specification is useful context but is not a whole-body cooperative-lift rating,
so linear addition is only a simulation hypothesis to test.

The attention actor's learned parameter shapes do not depend on teammate count. To initialize a different-size task
from an existing MAPPO checkpoint while deliberately reinitializing its size-dependent critic, set
`COOP_ACTOR_CHECKPOINT=/path/to/agent.pt` before calling `scripts/train.py`.

## Research and code reused

- [Isaac Lab's G1 loco-manipulation task](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/locomanipulation_g1_env_cfg.py)
  supplies the exact G1 asset, [pretrained AGILE policy](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/IsaacLab/Policies/Agile/agile_locomotion.pt),
  83-value observation contract, output scaling, and controller cadence used here. The surrounding Isaac Lab code is
  BSD-3-Clause. Because separate policy redistribution terms are not stated, the artifact is fetched at runtime.
- [TeamHOI](https://github.com/sail-sg/TeamHOI) supplies the shared decentralized-policy, teammate-token, staged
  lift/transport, formation, and load-balancing design. Its legacy Isaac Gym/generic-humanoid checkpoint is not
  compatible with G1, so the MIT-licensed concepts are adapted to current Isaac Lab rather than pretending the
  checkpoint can be fine-tuned.
- [CooHOI](https://github.com/Winston-Gu/CooHOI) motivates single-skill-to-team transfer. Frozen AGILE already provides
  the reusable individual locomotion skill; a separate single-G1 payload-pretraining phase remains a fallback if team
  lift learning stalls.
- [COLA](https://github.com/Yushi-Du/COLA_Code) is the closest released G1 collaborative-carrying pipeline. Its
  teacher/student and compliance ideas remain useful, but its Sim 5.1 controller models one G1 plus an externally
  controlled light training bar rather than a variable-size all-G1 team, so its student JIT is not used as an
  interchangeable checkpoint.

The highest-value next robustness experiment is to add message delay/dropout and uneven payload center of mass,
testing whether the learned coordination is robust rather than scripted around symmetric conditions.

## Layout

```text
src/cooperative_beam_isaaclab/tasks/
  cooperative_beam_env_cfg.py   scene, payload, curriculum, spaces, reward scales
  cooperative_beam_env.py       physics, slings, observations, rewards, resets
  hierarchical_controller.py    frozen AGILE batch plus GPU wrist IK
  control_contract.py           exact high-level/AGILE tensor contracts
  formation.py                  alternating-side station and load-target geometry
  livekit_state_bus.py          binary state track, freshness gate, actor adapter
  teamhoi_model.py              shared teammate-token attention actor
  parameter_sharing.py          shared skrl MAPPO actor/critic/optimizer adapter
  trajectory.py                 simulator-independent trajectory/reward helpers
  agents/                       MAPPO and IPPO configurations
scripts/
  smoke.py                      finite headless physics validation
  benchmark_matrix.py           mass × team-size screening matrix
  train.py                      wrapper around Isaac Lab's maintained skrl trainer
  evaluate.py                   finite frozen-policy evaluation and JSON metrics
  livekit_state_bridge.py       deployed LiveKit transport beside one G1 actor
  run_local.sh                  local MAPPO launch
  hf_job.sh                     iteration-controlled Hugging Face Jobs launch
  hf_eval_job.sh                remote frozen-policy evaluation launch
  hf_smoke_matrix.sh             parallel remote validation of all payload profiles
  hf_heldout_smoke_matrix.sh     one-factor and compound held-out physics probes
tests/
  test_trajectory.py            trajectory and reward invariants
  test_livekit_state_bus.py     packet, ordering, frame transform, stale-state tests
```
