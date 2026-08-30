# Optimus Prime — simplified 90-second pitch

## 0:00–0:15 — The problem

The mountain amplifies small errors. Our goal is simple: when G1 slips, reaches a cliff, or finds a blocked route, it can recover or keep moving without sending a person into danger.

## 0:15–0:30 — Four expedition skills

We built four skills: stop a slide, recover on a fixed line, descend under control, and move a load that one robot cannot.

## 0:30–0:45 — Learn the response

For single-robot skills, 125 observation features enter a compact PPO policy and become arm commands. Training starts with easy falls and expands to fast, cross-slope cases. Success only counts when the axe physically bites.

## 0:45–1:00 — Learn to coordinate

For lifting, every G1 runs the same MAPPO actor. It combines local state with teammate tokens and outputs ten high-level commands. We teach lift first, then carry, turn, and heavier loads.

## 1:00–1:15 — LiveKit state and voice

LiveKit is the field bus. Each robot publishes one state stream. Voice can ask for load or issue “move the log”; confirmation gates the mission, while local policies control joints.

## 1:15–1:30 — Evidence and handoff

Frozen tests: sixty-nine of sixty-nine arrests, recovery to standing on an icy fixed line, a two-meter rappel, and a fifty-fifty target load split. Now, the demo.
