# Optimus Prime — detailed 90-second pitch

## 0:00–0:15 — The problem

Mountains punish small failures. A slip becomes a fall; storm debris blocks the route. The G1 we bring to the Himalayas needs skills that use alpine tools, recover, and know when to stop.

## 0:15–0:30 — The extreme-condition use case

Optimus Prime built four skills: ice-axe self-arrest, fixed-line recovery, controlled rappel, and coordinated lifting that shares a long load.

## 0:30–0:47 — Single-robot PPO

The single-robot skills use MuJoCo PPO. Self-arrest maps 125 measurements through two 256-unit layers to 14 arm residuals at 100 hertz. Rewards require real pick contact, load, blade angle, grip, and body pose. The curriculum expands from gentle slides to five meters per second with cross-slope and adversarial falls.

## 0:47–1:04 — Cooperative MAPPO

In Isaac Lab, one shared MAPPO actor embeds 98 local features and seven-value teammate tokens with four-head attention. It outputs ten commands per G1 while frozen AGILE controls the legs. A central critic trains a team reward for tracking, levelness, load balance, and staying upright; the curriculum adds transport, turning, and heavier mass.

## 1:04–1:17 — LiveKit state and voice

At inference, LiveKit carries state and voice. One uplink per robot feeds every actor. The agent answers telemetry questions and turns “move the log” into a confirmed, gated start intent. Local policies still control motion.

## 1:17–1:30 — Proof and handoff

Frozen tests arrested nine of nine named and sixty of sixty randomized falls. Rappel descended two meters; the lift split load evenly. Now, the demo.
