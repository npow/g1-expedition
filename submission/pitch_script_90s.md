# Optimus Prime — 90-second pitch

## 0:00–0:18 — The problem

Mountains punish small failures. A slip becomes a fall. A cliff turns poor foot placement into a blocked descent. Storm debris demands team handling. The Unitree G1 we bring to the Himalayas needs more than walking—it needs skills that recognize failure, use alpine tools, and recover without forcing unsafe actions.

## 0:18–0:39 — The extreme-condition use case

Optimus Prime built G1 Expedition: four skills spanning the movement, action, and thinking tracks. Ice-axe self-arrest stops uncontrolled slides. Fixed-line recovery gets the robot back on its feet. Rappel couples brake control with foot placement. And coordinated lifting lets two robots share a long load while refusing overloads.

## 0:39–0:57 — The technical architecture

Self-arrest and fixed-line recovery use MuJoCo PPO plus a pretrained whole-body get-up prior. Cooperative lifting uses Isaac Lab: shared MAPPO produces ten commands per G1 while frozen AGILE stabilizes the legs.

## 0:57–1:13 — LiveKit state and voice

At inference, LiveKit carries state and voice. One state uplink per robot feeds every actor. The LiveKit agent answers telemetry questions and turns “move the log” into a confirmed start intent; hold and abort remain available. Voice supervises; local policies control motion.

## 1:13–1:30 — Proof and handoff

Frozen tests arrested nine of nine named and sixty of sixty randomized falls. Rappel descended two meters; the lift split load evenly and refused overload or imbalance. Together, the skills recover, descend, coordinate, and keep the operator in the loop. Now, the demo.
