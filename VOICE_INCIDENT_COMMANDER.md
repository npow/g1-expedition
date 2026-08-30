# G1 Expedition Voice Incident Commander

The incident commander is a hands-free radio interface for the cooperative G1
team. ElevenLabs Scribe v2 Realtime transcribes the operator, a tool-calling
voice agent reads the actual coordination state, and ElevenLabs Flash v2.5
speaks concise answers. The existing LiveKit room carries microphone audio and
the G1 state DataTracks.

Voice is a supervisory interface, not a motor controller. The language model
cannot emit joint targets. It may request only `start`, `hold`, `resume`, or
`abort`; the deterministic commander checks phase and the 150 ms telemetry
deadline plus reported load distribution before it admits activation. `start`
and `resume` require a separate explicit confirmation turn. Abort never
requires confirmation. Hardware emergency stop remains local and physical.

Confirmation is not entrusted to a tool argument chosen by the language model.
The deterministic boundary reads the latest user transcript and accepts only
the exact normalized phrase `confirm start` or `confirm resume`, while the
matching request is pending and less than 15 seconds old.

## What is implemented

- ElevenLabs realtime speech-to-text and streaming text-to-speech through the
  maintained LiveKit Agents plugin.
- Tool-grounded answers for mission phase, per-robot load and velocity, link
  freshness, load balance, and the last state transition.
- A clearly labelled credential-free telemetry fixture with demo-only link-loss
  injection.
- Live subscription to the same `g1-coordination-state-v1` tracks that feed the
  frozen MAPPO policy.
- Authorized LiveKit RPC delivery to each robot bridge. RPC responses report
  that a command was queued; they do not claim that motion occurred.
- A continuous active-mission watchdog that broadcasts abort after the
  telemetry deadline, without waiting for another voice request.
- All-robot acknowledgement for non-abort commands; partial delivery changes
  the commander to aborted and triggers a best-effort abort broadcast.
- A per-robot, deduplicated command queue exposed to the local deterministic
  supervisor as `supervisory_commands` in the bridge's JSON Lines output.

## Credential-free demonstration

This exercises the exact deterministic safety core without voice services,
LiveKit Cloud, Isaac Lab, or a GPU:

```bash
cd cooperative_beam_isaaclab
uv run --extra voice python scripts/incident_commander_console.py
```

Suggested sequence:

```text
report status
assess load
begin lift
confirm start
simulate loss of g1-1
why did you stop
```

The banner and every fault-injection response say that this is demo telemetry.
It must not be presented as a live simulator or robot run.

## ElevenLabs voice demonstration

Install the isolated voice dependencies:

```bash
cd cooperative_beam_isaaclab
uv sync --extra voice
```

Set the LiveKit project credentials used by LiveKit Agents and the ElevenLabs
API key used directly by both voice plugins:

```bash
export LIVEKIT_URL=wss://your-project.livekit.cloud
export LIVEKIT_API_KEY=...
export LIVEKIT_API_SECRET=...
export ELEVEN_API_KEY=...
```

The ElevenLabs key must be active and permitted to use both Scribe realtime
speech-to-text and text-to-speech. An `auth_error` during startup means the key
or its product permissions must be replaced; the agent deliberately does not
silently substitute fabricated transcripts.

Run a microphone/speaker console against the labelled fixture:

```bash
INCIDENT_DEMO_MODE=1 \
  uv run --extra voice python scripts/voice_incident_commander.py console
```

Or run text-only while checking tool behavior:

```bash
INCIDENT_DEMO_MODE=1 \
  uv run --extra voice python scripts/voice_incident_commander.py console --text
```

The default voice is ElevenLabs' public George voice, matching the submitted
video. Override it with `ELEVENLABS_VOICE_ID`. Override the LiveKit Inference
LLM with `INCIDENT_LLM_MODEL`; the LLM never supplies telemetry directly.

## Live telemetry and command boundary

Run each policy-side bridge with a token whose LiveKit identity is the matching
`g1_N` and explicitly authorize the voice participant:

```bash
LIVEKIT_TOKEN="$G1_0_TOKEN" \
python scripts/livekit_state_bridge.py \
  --robot-id g1_0 --team g1_0,g1_1,g1_2 \
  --command-caller incident-commander
```

Repeat for the other robot identities. Start the voice agent in the same room
with the authorized identity and without demo mode:

```bash
uv run --extra voice python scripts/voice_incident_commander.py connect \
  --room expedition --participant-identity incident-commander
```

Each robot bridge accepts only current, versioned, bounded RPC commands whose
source matches the authenticated identity. It rejects malformed or expired
commands, handles duplicate IDs idempotently, and returns an acknowledgement.
Its next JSON Lines output includes, for example:

```json
{
  "policy_observation": [0.0],
  "teammate_fresh": [true, true],
  "supervisory_commands": [
    {
      "version": 1,
      "command_id": "8fc4...",
      "intent": "hold",
      "issued_at_ns": 1788123456789000000,
      "source": "incident-commander"
    }
  ]
}
```

The robot-local supervisor remains responsible for mapping that high-level
intent to a safe state transition. It must never treat this queue as a direct
actuator command, and it must retain an independent emergency-stop path.

## Judge demo

The highest-signal sequence is short:

1. "Expedition Control, report team status."
2. "Which robot carries the highest load?"
3. "Begin lift." The agent requires confirmation.
4. "Confirm start."
5. "Simulate loss of G1-1." In demo mode, this visibly triggers the bounded
   automatic-abort path.
6. "Why did you stop?" The answer comes from the recorded deterministic state
   transition, not the language model's guess.

Relevant code:

- `tasks/incident_commander.py`: safety state machine and telemetry summaries.
- `scripts/voice_incident_commander.py`: ElevenLabs/LiveKit voice tools.
- `scripts/incident_commander_console.py`: credential-free demo.
- `tasks/livekit_state_bus.py`: state packets and authorized supervisory RPC.
- `tests/test_incident_commander.py`: phase, freshness, confirmation, and
  fault-injection tests.
- `tests/test_voice_incident_commander.py`: RPC-envelope and continuous-watchdog
  integration tests.
