# Optimus Prime — Expedition Skills for Humanoid Mountaineering (Simplified 90s)

## 0:00–0:15 — The problem

The mountain amplifies small errors. Our goal is simple: when G1 slips, reaches a cliff, or finds a blocked route, it can recover or keep moving without sending a person into danger.

## 0:15–0:30 — Four expedition skills

We built four skills: stop a slide, recover on a fixed line, descend under control, and move a load that one robot cannot.

## 0:30–0:45 — Learn the response

In MuJoCo, 125 observation features enter a compact PPO policy for single-robot skills. Physical reward shaping enforces true pick contact and blade angle, while curriculum learning progresses from easy slides to fast cross-slope falls. Success only counts when the axe physically bites.

## 0:45–1:00 — Learn to coordinate

In Isaac Sim, every G1 runs the same MAPPO actor trained with centralized team reward shaping for load balance and levelness. A multi-agent curriculum teaches lift first, then carry, turn, and heavier loads.

## 1:00–1:15 — LiveKit state and voice

LiveKit is the field bus. Each robot publishes one state stream. Voice can ask for load or issue “move the log”; confirmation gates the mission, while local policies control joints.

## 1:15–1:30 — Open challenges: The C.L.I.M.B. roadmap

To scale from simulation to the real mountain, we frame open challenges with C.L.I.M.B.: Contact dynamics on snow and ice, Lighting and sensor breakdown in whiteouts, Inter-robot mesh comms without cloud, Mechanical rope elasticity, and Balance without simulation stabilizer assists. Now, the demo.
