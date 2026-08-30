# Physical fall and learned G1 get-up

This document covers the isolated flat-ground provenance baseline. The same
pinned prior is integrated into the 28° fixed-line mountain scene through a
four-action PPO policy trained in this repository; see
[`FIXED_LINE.md`](FIXED_LINE.md#integrated-fall-get-up-and-resumed-ascent) and
`videos/g1_fixed_line_fall_recovery.mp4`.

This scenario now renders a continuous physical fall and recovery:

1. The G1 settles in a motor-supported stand.
2. A `100 N` world-frame shove acts on the torso for `0.20 s`.
3. Motor stiffness is removed; gravity, damping, collision geometry, and the
   MuJoCo contact solver produce the fall and let it settle for `3.0 s`.
4. A grounded, joint-only alignment moves into the policy's floor-ready
   posture. The floating base remains free.
5. A pretrained reinforcement-learned whole-body controller runs at `50 Hz`
   with ten `2 ms` physics substeps per action and per-motor torque limits.

There is no floating-base write after the fall starts and no upward assist
force during recovery.

## Result

The checked report is
[`models/wbc_getup/evaluation_report.json`](models/wbc_getup/evaluation_report.json).

- `4/4` backward, softer, and laterally biased physical falls recovered.
- Minimum final pelvis height: `0.771 m`.
- Minimum final torso upright alignment: `0.997`.
- Maximum normalized motor torque: `1.000` (the configured catalog cap).
- Maximum reported contact penetration: `29.2 mm`.
- Floating-base teleports after fall start: `0`.
- Reference trajectory without learned residuals: **fails**.
- Motors off during recovery: **fails**.

The causal ablations matter: this is not a kinematic replay hidden inside a
video. The motion reference gives the pretrained controller an intent, while
the network's state-dependent residuals are necessary to stand in this model.

## Policy provenance and honest learning boundary

The policy was **not trained in this repository**. It is the Apache-2.0
whole-body G1 policy and get-up clip from
[`wbc-mjlab/wbc-g1-deploy`](https://github.com/wbc-mjlab/wbc-g1-deploy),
pinned to commit `6dabf86fddc2b7b429b09e74999732fcde3441f9`. The download script
verifies SHA-256 digests, and the upstream license is kept beside the assets in
`third_party/wbc_g1_getup/`.

This adapter reproduces the upstream `132`-element observation contract,
reference-residual action mapping, PD gains, and motor torque catalog. It does
not relabel upstream training as hackathon training.

## Why this structure

The implementation follows the central findings of recent G1 standing-up
work: get-up is a contact-rich whole-body task, useful policies must handle
grounded configurations rather than a suspended base, and smoothness/actuator
constraints must be restored before deployment. The design references:

- [HoST: Learning Humanoid Standing-up Control across Diverse Postures](https://arxiv.org/abs/2502.08378)
- [Learning Getting-Up Policies for Real-World Humanoid Robots](https://arxiv.org/abs/2502.12152)
- [MuJoCo contact and constraint computation](https://mujoco.readthedocs.io/en/latest/computation/)
- [Unitree's official `Lie2StandUp` SDK example](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/g1/high_level/g1_loco_client_example.py)

The Unitree example explicitly limits lie-to-stand operation to a hard, flat,
rough surface. This scenario deliberately uses that envelope; it is not a
claim of safe hardware recovery on snow, ice, rocks, or around people.

## Reproduce

```bash
uv run --with-requirements requirements.txt python scripts/fetch_getup_assets.py

uv run --with-requirements requirements.txt python evaluate_getup.py \
  --output models/wbc_getup/evaluation_report.json

MUJOCO_GL=cgl uv run --with-requirements requirements.txt python record_getup.py \
  --output videos/g1_physical_getup.mp4 \
  --report videos/g1_physical_getup.json
```

On Linux, use `MUJOCO_GL=egl` for headless rendering.

## Files

- `assets/unitree_g1/scene_getup.xml`: solver, compliant rough floor, lighting.
- `getup_controller.py`: fall mechanics and pretrained WBC adapter.
- `evaluate_getup.py`: four perturbations plus causal ablations.
- `record_getup.py`: uncut policy-in-the-loop evidence render.
- `scripts/fetch_getup_assets.py`: pinned, digest-verified upstream assets.
- `third_party/wbc_g1_getup/`: policy, clip, license, and attribution.

## Limits

This is simulation evidence, not a hardware safety certificate. The roughly `6 kN`
peak solver contact force is a rigid-mesh impact diagnostic, not a calibrated
load-cell prediction. The collision meshes, actuator dynamics, ground model,
latency, temperature, and structural compliance still require system
identification and conservative real-robot validation with fall protection and
an emergency stop.
