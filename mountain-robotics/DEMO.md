# Alpine Lift — Voice-Controlled Himalayan Coordinated Humanoids

Live multi-modal demonstration of two Unitree G1 humanoids executing coordinated manipulation in extreme alpine terrain under real-time voice guidance powered by **LiveKit Agents**.

---

## LiveKit Voice Architecture

```
[ Operator Voice / Mic ]
           │
           ▼
[ LiveKit Agents (LiveKit Cloud / Local Session) ]
  • STT: Nova-3 streaming transcription
  • Reasoning: Function tool selection with strict motion guardrails
  • Voice: Sonic-3 low-latency spoken synthesis
           │
           ▼ Function Tool Calls:
           ├── lift_log()             ──> POST /api/command {"phrase": "lift the log"}
           ├── test_heavy_load()      ──> POST /api/command {"phrase": "test heavy load"}
           ├── add_wind_gust()        ──> POST /api/command {"phrase": "add a wind gust"}
           ├── simulate_verglas_ice() ──> POST /api/command {"phrase": "simulate verglas"}
           ├── telemetry_status()     ──> POST /api/command {"phrase": "status report"}
           ├── reset_system()         ──> POST /api/command {"phrase": "reset system"}
           └── operator_stop()        ──> POST /api/command {"phrase": "operator stop"}
           │
           ▼
[ Alpine Local Control Server (http://127.0.0.1:8765) ]
  • MuJoCo 3.12 50 Hz coupled multi-body simulation
  • Four-point compliant sling constraints (soft contacts, no welds)
  • Autonomous load weighing & safety gate (Track 3 Thinking)
  • Real-time offscreen OpenGL rendering HUD
```

---

## Complete Voice Command Set

| Command | Example Voice Utterance | Hackathon Track | System Response & Physics Behavior |
| :--- | :--- | :--- | :--- |
| **Lift Log** | *“Lift the log”* / *“Raise the timber”* | Main Mission | Initiates reach, clip, tension weighing, coordinated lift (19 cm), shift, and set-down. |
| **Test Heavy Load** | *“Test heavy load”* / *“Evaluate 30kg log”* | Track 3 Thinking | G1s weigh the 30 kg log, detect arm limit exceeded (>60 N/hand), decline lift, and safely abort. |
| **Add Wind Gust** | *“Add a wind gust”* / *“Simulate blizzard”* | Extreme Conditions | Injects a 45 N crosswind lateral drag during the loaded lift phase. |
| **Simulate Verglas** | *“Simulate verglas”* / *“Verglas underfoot”* | Track 1 Movement | Reduces ground contact friction ($\mu \to 0.45$) to test foothold and posture regulation on ice. |
| **Telemetry Status** | *“Status report”* / *“Telemetry readout”* | Operator HUD | Spoken readout of phase, measured mass, log tilt, load sharing A/B, and peak hand force. |
| **Operator Stop** | *“Operator stop”* / *“Emergency abort”* | Safety Interlock | High-priority immediate controlled set-down and gentle sling release. |
| **Reset System** | *“Reset system”* / *“Ready stance”* | Stage Reset | Returns both robots to the armed pre-lift stance. |

---

## Five-Minute Stage Preflight

```bash
cd /Users/pasithea/g1-expedition/mountain-robotics

# 1. Start the visual HUD & physics server (port 8765):
PYTHONPATH=. .venv/bin/python scripts/voice_demo.py --open
```

To run with LiveKit Agents:
1. Ensure your LiveKit project keys are in `.env.local`:
   ```bash
   LIVEKIT_URL=wss://your-project.livekit.cloud
   LIVEKIT_API_KEY=...
   LIVEKIT_API_SECRET=...
   ```
2. Launch the integrated runner:
   ```bash
   .venv/bin/python scripts/run_livekit_demo.py
   ```

*Note*: If venue Wi-Fi is spotty, the browser console (`http://127.0.0.1:8765`) includes built-in browser speech recognition and one-click stage controls with zero cloud dependencies.

---

## Two-Minute Stage Presentation Script

1. **0:00 — The Extreme Challenge:**
   > *“Fallen timber on high-altitude Himalayan trails blocks critical supply routes. A 12 kg log is light enough for two Unitree G1s, but rigid welds don't exist in the field—slings slip, terrain is iced, and high winds blow.”*

2. **0:20 — LiveKit Natural Voice Command:**
   > Say into the microphone: **“Alpine, lift the log.”**
   > *LiveKit Agent confirms and calls `lift_log()`. Both G1s transition from `READY` to `REACH` and `CLIP`.*

3. **0:40 — Track 3 Thinking (Weigh-in & Load Sharing):**
   > Say: **“Status report.”**
   > *LiveKit speaks the live telemetry: Phase, mass estimate (10.9 kg), tilt (3.9°), load share (50/50).*
   > *“Notice the G1s weigh the obstacle before lifting. If we test a 30 kg log (`test heavy load`), the system autonomously rejects the lift to protect the actuators.”*

4. **1:05 — Extreme Condition Disturbances (Track 1 & 2):**
   > Say: **“Simulate verglas.”**
   > *Ground friction drops to $\mu=0.45$. Point to the foot contacts and posture adjustments.*
   > Say: **“Add a wind gust.”**
   > *45 N crosswind gust hits the robots. Telemetry shows the tilt transient while the coordinated controller holds stability.*

5. **1:35 — Safe Completion & Close:**
   > *“The log is lifted 19 cm, shifted clear of the trail, lowered in a controlled descent, and released. That is end-to-end voice-commanded cooperative robotics under Himalayan conditions.”*
