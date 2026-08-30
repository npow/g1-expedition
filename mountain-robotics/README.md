# Alpine Coordinated Lift

**Two Unitree G1 humanoids clearing a fallen tree from a Himalayan approach trail.**

Himalaya Robotics Hackathon · Track 1 (Movement) + Track 3 (Thinking)

<img src="out/frame_HOLD.png" width="100%">

---

## The problem

A conifer comes down across the fixed-line route above base camp. It is 1.7 m
long and weighs about 11 kg — well inside what a G1 can hold. But **no single
humanoid can move it**, because a robot gripping a 1.7 m log anywhere except
its centre of mass cannot keep it level, and a robot gripping it *at* the
centre has no leverage over its yaw. The load is not heavy. It is *long*.

That is the actual reason cooperative manipulation exists, and it is the
reason this is a two-robot problem rather than one robot doing it twice.

Two G1s stand at either end, clip both hands into rope slings choked around
the trunk, weigh it in place, decide whether the team can take it, lift it
clear of the rocks it is resting on, hold it level through whatever the
mountain does next, move it off the trail line, and set it down.

## Why this is the right task for extreme conditions

- **The load is unknown.** Nobody hands a mountain robot a spec sheet. The
  team weighs the log through its own slings before committing, and infers
  where its centre of mass sits from how the load splits between the ends.
- **The answer is sometimes no.** A go/no-go check on measured load versus arm
  capacity is the decision a rope team actually makes. `--mass 30` shows the
  pair weighing a log, declining it, and setting it back down intact.
- **Conditions degrade mid-task.** Gusts, verglas underfoot, and an off-centre
  load all arrive while the thing is already in the air.
- **Failure has to be graceful.** Every abort is a controlled set-down from
  wherever the load currently is, never a drop.

---

## Submission

| | |
|---|---|
| **Track** | Track 1 (Movement) + Track 3 (Thinking) |
| **Demo video** | `out/submission.mp4` — 65 s, built by `scripts/make_reel.py` |
| **Live demo** | `python scripts/voice_demo.py --open` — voice console + live physics |
| **Clips** | `out/01_nominal` · `02_nogo` · `03_gust` · `04_hands` · `05_front` |
| **Reproduce every claim** | `python scripts/scenarios.py` — 30 seconds |

---

## Running it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python mujoco==3.12.0 numpy imageio imageio-ffmpeg pillow

# interactive — this is the one to run live.
# note: mjpython, not python. MuJoCo's passive viewer must own the main
# thread on macOS; everything else in this repo runs under normal python.
PYTHONPATH=. .venv/bin/mjpython scripts/live.py

# the three set pieces
PYTHONPATH=. .venv/bin/mjpython scripts/live.py --mass 30    # team declines the lift
PYTHONPATH=. .venv/bin/mjpython scripts/live.py --wind 45    # gust during the carry
PYTHONPATH=. .venv/bin/mjpython scripts/live.py --ice 0.3    # verglas underfoot

# headless trace, and video with the telemetry overlay
PYTHONPATH=. .venv/bin/python scripts/run_mission.py
PYTHONPATH=. .venv/bin/python scripts/record.py --out out/lift.mp4

# every claim in this README, re-measured in 30 seconds
PYTHONPATH=. .venv/bin/python scripts/scenarios.py

# rebuild the submission reel from the recorded clips
PYTHONPATH=. .venv/bin/python scripts/make_reel.py     # -> out/submission.mp4
```

Everything above runs on a MacBook Air M1 at ~7× real time. No GPU required.

### Voice-controlled live demo

The stage-safe demo is a local browser console backed by the actual MuJoCo
lift, with a recorded-visual fallback if the laptop cannot create an offscreen
graphics context. The control service and button fallback need no network;
browser speech recognition and the optional LiveKit path may. Robot motion does
not start until the command boundary recognizes an explicit lift intent.

```bash
# opens http://127.0.0.1:8765
PYTHONPATH=. .venv/bin/python scripts/voice_demo.py --open
```

Click **Start listening**, allow microphone access, and say **“lift the log.”**
The **Run demo** button exercises the identical command path when venue audio
is unreliable. **Operator stop** requests the controller's controlled set-down
instead of killing the simulation.

The primary voice path is a LiveKit Agent with safety-bounded function tools:
`lift_log`, `test_heavy_load`, `simulate_verglas_ice`, `add_wind_gust`, `telemetry_status`, `reset_system`, and `operator_stop`. The local service strictly validates every intent before changing the simulation:

```bash
uv pip install --python .venv/bin/python '.[voice]'
# put LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET in .env.local
.venv/bin/python scripts/run_livekit_demo.py
```

The control service defaults to `http://127.0.0.1:8765`. Set
`ALPINE_CONTROL_URL` for another host and use the same `ALPINE_CONTROL_TOKEN`
on both processes when the endpoint is not laptop-local.

Supported voice commands:
- **“Lift the log”**: Initiates autonomous coordinated lift (19 cm) and clearance.
- **“Test heavy load”**: Autonomous weigh-in (Track 3 Thinking) and safety decline of a 30 kg log.
- **“Simulate verglas”**: Himalayan ice friction drop ($\mu \to 0.45$) underfoot.
- **“Add a wind gust”**: 45 N crosswind lateral drag during the loaded phase.
- **“Status report”**: Spoken readout of phase, measured load, tilt angle, load sharing, and peak force.
- **“Reset system”**: Returns robots to the armed ready stance.
- **“Operator stop”**: Emergency high-priority controlled set-down.

See [`DEMO.md`](DEMO.md) for the two-minute stage runbook and architecture details.

---

## How it works

The scene is generated, not authored. `alpine_lift/scene.py` emits the whole
MJCF as text — two prefixed copies of menagerie's `g1_mjx.xml`, the payload,
the rocks it rests on, 229 explicit contact pairs and four sling constraints —
so every physical parameter that matters is one keyword argument and domain
randomisation is just regenerating the string.

### Coordination: one virtual object, not two arm plans

The two robots are never given independent trajectories. The controller
maintains a single **commanded payload pose**, and all four palm targets are
that pose composed with each hand's fixed sling-loop offset. Because every
hand derives from one moving frame, the team is kinematically consistent by
construction — there is no leader, no follower, and no way for the two robots
to disagree about where the log is going.

### Grip: `equality/connect` as a carabiner

Each hand is clipped to the payload by a 3-DoF `connect` constraint between
the palm site and a sling loop. That is the right abstraction for a carabiner
in a sling — load is carried, the hand is free to rotate about the clip-in —
and it makes the solver's Lagrange multipliers a direct readout of the force
each hand is transmitting. The whole load-sharing and safety layer reads true
interaction forces with no added sensors.

Slings engage soft and stiffen over the TENSION phase. Switching a `connect`
on across even a 45 mm gap yanks the sites together and spikes the force past
the sling rating; ramping the constraint time constant from 0.40 s to 0.05 s
is the simulated equivalent of taking up slack before you load a sling.

### Whole-body IK with a balance task

`kinematics.py` solves all 35 DoF per robot — 6 floating-base plus 29 joints —
against stacked weighted tasks: feet pinned, CoM over the support polygon,
palms on the loops, pelvis height and attitude, posture in the nullspace. The
CoM row is not optional: without it the solver happily returns a configuration
that reaches the target and falls over.

### Balance

`balance.py` is a capture-point regulator. It feeds corrections in as a shift
of the **IK's CoM target**, not as joint offsets bolted onto the IK's answer,
plus a small fast ankle term. It also knows what the robot is holding: half an
11 kg log hanging off the hands acts well outside the footprint, and a
controller reasoning only about `subtree_com` will walk the machine past its
own toes while insisting it is balanced.

### The coordinator

`coordinator.py` weighs the load, decides go/no-go once, levels the payload,
paces the team to its slower member, and releases any sling held above rating.

---

## What the engineering actually cost

The interesting parts of this build were the things that did not work. All of
these are measurements from this repo, not received wisdom.

**Menagerie's G1 falls over in three seconds.** Stock `scene_mjx.xml`, stock
keyframes, no controller: the pelvis travels 0.9 m and ends at 0.1 m height.
Position servos alone do not stabilise an underactuated biped — which is
exactly why the Playground models ship paired with learned policies. Standing
had to be built, not assumed.

**The IK collapsed the robot in a quarter of a second.** Pinning the feet to a
world-frame stance captured at mission start creates a positive feedback loop:
the pelvis sags a few millimetres under load, the solver sees feet that have
dropped with it, and "corrects" them by retracting the legs — which lowers the
pelvis further. Refreshing the foot targets from measurement every tick makes
the foot rows carry zero residual and act as a pure constraint, which is what
forces bent knees to lower the pelvis rather than lift the feet.

**Stiffer arms made it worse.** Raising arm gains to match the legs (kp 270)
put an 8 Hz resonance between the arm servos and the compliant sling straight
through the carry. With the controller *frozen*, sling force still rang ±60 N
about a 108 N load. At kp 145 the same measurement gives a standard deviation
of 3.9 N. Holding a compliant load is a case where the softer joint is the
more stable one.

**Feed-forward helps on arms and destroys legs.** Compensating
`qfrc_constraint` removes ~7 cm of palm sag on an arm, where that term is the
sling load. On a leg it is dominated by ground contact — the force already
holding the robot up — and feeding it back closes a loop through the contact
solver that drives palm error to 40 cm within a second.

**The exploration was the disturbance.** Two training runs reported the policy
getting steadily worse — 73% success down to 57%, mean tilt climbing from 14°
to 30°. It was not the policy. `log_std` was free to drift upward under an
entropy bonus, and this task is unusually sensitive to action noise: holding
the residual at zero the baseline succeeds 70% of the time, at exploration
std 0.135 that falls to 60%, and at 0.27 to 50%. The training curve was
plotting the noise eating the baseline. Clamping std at 0.11 and dropping the
entropy bonus fixes the measurement. Worth knowing before reading any RL
curve on a task where the *initial* policy is already good.

**The obvious fix was the wrong one.** Two humanoids holding one rigid body
form a closed kinematic chain, so it looks obvious that the carry needs PD
feedback on the measured payload pose. It does not. Every nonzero gain made
the hold shorter, monotonically: 6.8 s open-loop, 4.8 s at kp 0.2, 2.2 s at
kp 0.2 with kd 0.25, 1.9 s at kp 0.35. Too much phase lag between a palm
target and the payload turns added loop gain into added divergence. The code
is still in the tree at zero gain, because the negative result is worth more
than the code.

**The reward paid the agent to give up.** The first reward had a small
survival bonus against dense tracking penalties, so a whole episode summed to
about -400. That makes quitting the best move on the board: tip a robot over
at step 300 and you skip 460 steps of accumulated penalty, and the flat -12
abort cost does not come close to covering the difference. Training went
backwards exactly as you would predict -- 71% success down to 58% over half a
million steps, mean tilt climbing from 14 to 32 degrees. The fix is two
lines: make a well-tracked step score positive so surviving accrues reward
rather than debt, and charge an abort for the *rest of the episode it threw
away*. Successful and failed episodes now sit at +620 and -724.

**Four integrators pulling on one log is three too many.** Post-clip palm
integrators wind internal force into the closed chain that nothing regulates.
A hold that survives 12.8 s without them collapses in 6.8 s with them. They
are now active during the reach only, where each hand's error really is that
hand's to fix.

**Half the load was invisible.** The weigh-in read 5.6 kg for an 11 kg log
and would have had the team decline loads it could handle. The sling
constraints are not the only path into the payload: a hand cupped under a log
carries part of it by simply touching it, and that share appears in no
equality multiplier. Adding contact forces to the load estimate fixed the
weigh-in to within 2% — and, because the balance controller uses the same
number to place its effective centre of mass, it also raised the randomised
success rate from 43% to 58% and turned verglas at mu=0.45 from a fall into a
completed mission. One missing term, three symptoms.

**The simulator refused a dishonest payload.** The first design was a boulder.
At the mass two G1s could actually lift, a 0.5 m granite block would have had
to weigh 11 kg — so the payload became a log, where the number is real. The
sim also rejected the first mass: at 24 kg the robots simply could not lift
it. That constraint became the go/no-go feature rather than something to tune
away.

---

## Results

Full mission, nominal conditions, MacBook Air M1:

| | |
|---|---|
| clip-in accuracy | 23 mm palm-to-loop at engagement |
| payload lift | 19 cm, commanded 24 cm |
| tilt held through carry | **3.9°** peak while gripped |
| load sharing | 50 / 50 |
| peak hand force | 201 N against a 340 N sling rating |
| weigh-in accuracy | **within 2%** across 8–20 kg |
| abort rate | 0 |

The weigh-in is the number worth looking at. Across 8, 11, 14, 17 and 20 kg
logs the team estimates 8.0, 10.9, 13.7, 16.7 and 19.8 kg respectively, and
lifts all of them. At 24 kg and above it declines — 111 N per hand against a
60 N rating — and sets the log back down without it ever leaving the rocks.
The boundary is a measurement, not a threshold anyone tuned.

| payload | estimate | outcome |
|---|---|---|
| 8 kg | 8.0 kg | lifted 18 cm |
| 11 kg | 10.9 kg | lifted 19 cm |
| 14 kg | 13.7 kg | lifted 20 cm |
| 17 kg | 16.7 kg | lifted 20 cm |
| 20 kg | 19.8 kg | lifted 18 cm |
| 24 kg | — | **declined**, 112 N/hand |
| 30 kg | — | **declined**, 111 N/hand |
| 40 kg | — | **declined**, 106 N/hand |

Under randomised conditions — mass 9–13 kg, CoM offset ±16 cm, ground friction
0.7–1.0, gusts to 14 N, pushes to 35 N, verglas — the scripted controller
alone completes **71%** of missions (measured, 24 episodes). Single-factor
sensitivity, 10 episodes each against a 10/10 nominal:

| disturbance | success |
|---|---|
| CoM offset ±16 cm | 10/10 |
| mass 8–15 kg | 10/10 |
| ground friction 0.35–1.0 | 7/10 |
| push 0–90 N | 7/10 |
| verglas | 7/10 |
| **gusts 0–60 N** | **5/10** |

At the level of individual set pieces: verglas at mu=0.45 completes, mu=0.30
does not; a 40 N push completes, 70 N does not; a 20 N gust already takes it
down.

Gusts are where the model-based stack runs out, which is what the learned
residual is aimed at.

---

## The learned part

`env.py` wraps the mission as an RL environment where the policy outputs a
**residual** on the controller that already works — per robot: palm height
left and right, CoM bias in x and y, squat depth. Ten numbers, each worth
±3.5 cm, expressed in the coordinator's own units rather than reaching around
it to the joints.

This shape was chosen for two reasons. There is always a working baseline to
fall back on, so the demo cannot end with nothing to show. And the comparison
that matters — scripted alone versus scripted plus residual, *same seed, same
gust, same patch of ice* — is a controlled experiment rather than two
anecdotes.

Training is PPO (`train/train_residual.py`) on Hugging Face Jobs. The
environment is the bottleneck at ~340 steps/s per core, so the job runs on
`l4x4` for its 48 vCPUs; the policy update genuinely uses the GPU while those
cores run the env workers. The export is `policy.npz` — plain arrays, loadable
with numpy alone — so the laptop running the live demo needs no PyTorch and
has no framework that can fail to import five minutes before a pitch.

```bash
# evaluate baseline vs residual on identical episodes
PYTHONPATH=. .venv/bin/python scripts/evaluate.py --policy out/policy.npz
PYTHONPATH=. .venv/bin/mjpython scripts/live.py --policy out/policy.npz --wind 45
```

### It did not work, and that is the finding

5.0M environment steps, 1395 PPO updates, ~85 minutes on an A10G. Evaluated
deterministically on 40 episodes, **same seeds, same masses, same gusts, same
patches of verglas** for both controllers:

| controller | success | lift | tilt | peak hand force |
|---|---|---|---|---|
| scripted only | **60%** | 17.4 cm | 34.4° | 456 N |
| scripted + residual | 55% | 17.2 cm | 36.0° | 502 N |

The policy rescued 2 episodes the baseline lost and lost 4 the baseline won.
On 40 episodes that is a two-episode difference — inside the noise, and
pointing the wrong way. The trained policy is in the repo and on the Hub, and
the honest summary is that **the model-based controller is better than what
five million steps of residual PPO learned on top of it.**

Why, as best I can tell: episode return is dominated by which disturbance was
drawn, not by what the policy did. A ±3.5 cm palm correction moves the return
by far less than the difference between a 2 N gust and a 14 N one, so the
advantage estimate is mostly measuring the weather. The training curve is
consistent with this — it tracks the baseline's success rate for the whole run
and never separates from it.

What I would change, in order:

1. **Train on the failure mode only.** Single-factor sensitivity says gusts are
   where the stack runs out (5/10, against 10/10 on mass and CoM offset).
   Spreading episodes across disturbances the controller already handles
   perfectly spends most of the budget where there is nothing to learn.
2. **Cut the episode-level variance.** Fix the disturbance draw across a
   batch, or subtract a learned per-condition baseline, so the advantage
   reflects the policy rather than the draw.
3. **Give the residual authority where the failures are.** They are balance
   failures, and the policy's CoM channel is ±1.6 cm against a support polygon
   of ±9 cm. That ceiling was set to stop exploration noise wrecking the
   baseline; with the noise now clamped it can be raised.

A known approximation in the implementation, for completeness: actions are
sampled pre-tanh and the log-probability is taken on the pre-squash Gaussian,
without the tanh Jacobian correction. Common shortcut, not exactly correct.

---

## Limitations

Stated plainly, because they are real.

- **The feet never move.** Everything here is a standing manipulation task.
  The achievable lateral work envelope is about ±15 cm; moving the log further
  needs stepping, which is out of scope.
- **The carry has a ~10 s stability window.** Internal force wound into the
  closed chain by four independently-servoed hands eventually pulls a robot
  off balance. The phase budget is sized to fit inside it. The principled fix
  is explicit internal-force regulation — decomposing grasp forces into
  resultant and squeeze components and regulating the squeeze — which is the
  first thing to build next.
- **Grip is a constraint, not contact.** Hands are clipped to sling loops
  rather than closing fingers on a rope. Force-limited and breakable, but not
  a dexterous grasp.
- **Joint gains are not menagerie's.** `gains.py` documents every change and
  keeps commanded torque inside each joint's declared `actuatorfrcrange`, so
  nothing asks for torque the hardware does not have.
- **The arm capacity number is the soft one.** `arm_capacity = 60 N` per hand
  is the usable force for a braced two-handed hold with the load close to the
  chest. A G1 arm is rated 2–3 kg outstretched, which is a different posture,
  and the measured per-hand force here includes closed-chain internal force
  that a free-body split does not show. It is the first assumption to check
  against hardware, and the go/no-go threshold moves with it.
- **Sim only.** No hardware transfer was attempted. The G1 model is the one
  MuJoCo Playground transferred to hardware, which is the reason it was chosen.

---

## Layout

```
alpine_lift/
  scene.py         procedural MJCF: robots, payload, terrain, contacts, slings
  indexing.py      name -> id resolution for the composed two-robot model
  kinematics.py    whole-body damped-least-squares IK with a CoM task
  balance.py       load-aware capture-point regulator
  gains.py         actuator gain schedule and why it differs from menagerie
  forces.py        sling loads read from the constraint solver
  coordinator.py   weigh-in, go/no-go, levelling, pace, sling safety
  controller.py    phase machine and the shared virtual-payload reference
  mission.py       scene + physics + controller + disturbances
  env.py           residual-RL environment
  policy.py        numpy-only inference for the trained policy
  render.py        offscreen rendering with the telemetry overlay
scripts/           live.py       interactive viewer (run with mjpython on macOS)
                   run_mission.py headless trace of a single mission
                   scenarios.py   re-measures every claim in this README
                   record.py      mp4 with the telemetry overlay
                   compare.py     side-by-side baseline vs residual
                   make_reel.py   stitches the clips into out/submission.mp4
                   evaluate.py    baseline vs residual over N seeds
train/             train_residual.py (PPO, runs as a Hugging Face Job)
```

## Credits

- Robot model: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
  `unitree_g1` (BSD-3-Clause), the MJX variant transferred to hardware in
  [MuJoCo Playground](https://playground.mujoco.org/).
- Physics: [MuJoCo](https://mujoco.org/) 3.12.
- GPU training: Hugging Face Jobs.

Built during the Himalaya Robotics Hackathon, 29–30 August 2026.
