"""Voice control for the G1 fixed-line demo, over LiveKit.

Why voice is not decoration here
--------------------------------
A rescuer on a fixed line at altitude is gloved, roped in, in wind, often
unable to see the robot. There is no keyboard and no touchscreen. Voice is not
a nicer interface for this deployment; it is the only usable one.

Uses LiveKit Inference for STT/LLM/TTS -- models are given as STRINGS, which
routes them through LiveKit's gateway on your existing LIVEKIT_* credentials.
No separate provider key.

The agent never touches the physics. It writes intents to sim_bridge, which the
simulation drains once per control step. MuJoCo runs at 50 Hz; a speech round
trip is 300 ms-2 s, so anything else would stall the sim mid-stride.

    # terminal 1
    .venv/bin/mjpython demo_live.py
    # terminal 2
    .venv/bin/python voice_agent.py console      # local mic, no room needed
    .venv/bin/python voice_agent.py dev          # joins a LiveKit room
"""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib

from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.plugins import silero

import sim_bridge

# Credentials live in the himalaya-hack repo; never copied, never printed.
for env in (pathlib.Path(".env"),
            pathlib.Path.home() / "projects/himalaya-hack/.env"):
    if env.exists():
        load_dotenv(env)
        break

logger = logging.getLogger("g1-voice")

INSTRUCTIONS = """You are the voice interface for a Unitree G1 humanoid climbing
a fixed rope on a 28 degree snow slope, in simulation.

You are talking to an operator who cannot use their hands. Be terse. One short
sentence. Say what you did or what the robot is doing, nothing else. No
pleasantries, no restating the question.

You may ONLY use the tools provided. If asked for anything else -- change a
gain, alter the gait, make it climb faster -- say plainly that you cannot do
that. Never invent a capability. Never claim the robot did something you did
not command.

Distances are metres, force is newtons, mass is kilograms."""


class G1Operator(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS)

    @function_tool
    async def shove(self, newtons: int = 700) -> str:
        """Apply a lateral impulse to knock the robot off its stride.

        Args:
            newtons: Impulse strength. 400 is a nudge it absorbs, 700 causes a
                0.35 m slip it recovers from, 1000 or more it cannot recover.
        """
        sim_bridge.send("shove", newtons=newtons)
        return f"applied a {newtons} newton shove"

    @function_tool
    async def set_balance_assist(self, scale: float) -> str:
        """Scale the external balance stabiliser. 1.0 is full, 0.0 is off.

        Args:
            scale: 0.0 to 1.0. Below about 0.5 the robot cannot walk.
        """
        scale = max(0.0, min(1.0, scale))
        sim_bridge.send("assist", scale=scale)
        return f"balance assist set to {scale:.2f}"

    @function_tool
    async def reset(self) -> str:
        """Restart the climb from the bottom of the rope."""
        sim_bridge.send("reset")
        return "reset"

    @function_tool
    async def status(self) -> str:
        """Report what the robot is doing right now: phase, height, slip, load."""
        s = sim_bridge.status()
        return (f"phase {s.get('phase','unknown')}, "
                f"climbed {s.get('ascent_m',0):.2f} metres, "
                f"slip {s.get('slip_depth_m',0):.2f} metres, "
                f"assist {s.get('assist_scale',1.0):.2f}")


# What the robot says as the slip unfolds. Short, because it has to land inside
# the event -- a full sentence finishes after the robot has already recovered.
NARRATION = {
    "SLIP": "Slipping.",
    "RECOVER": "Caught it. Regaining stance.",
    "MOVING": "Recovered. Climbing again.",
}


async def narrate_phase(session) -> None:
    """Speak on phase CHANGE, unprompted.

    This is the point of the integration. An agent that only answers questions
    is a voice menu; one that calls the slip while the audience is watching it
    happen is the demo. The phase already exists -- demo_live.py publishes it
    every control step -- so this only has to notice the transition.

    Polls at 100 ms. The sim writes status atomically (write-then-rename), so a
    partial read is impossible, and a missed tick just means the next one wins.
    Never raises: narration must not be able to kill the session.
    """
    last = sim_bridge.status().get("phase")
    while True:
        try:
            phase = sim_bridge.status().get("phase")
            if phase != last:
                line = NARRATION.get(phase)
                if line:
                    session.say(line, allow_interruptions=True)
                last = phase
        except Exception:
            logger.debug("narration tick failed", exc_info=True)
        await asyncio.sleep(0.1)


async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        # Strings -> LiveKit Inference, billed to the LIVEKIT_* credentials.
        stt="assemblyai/universal-streaming:en",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2",
        vad=silero.VAD.load(),
    )
    await session.start(agent=G1Operator(), room=ctx.room)
    # Background narrator. Runs for the life of the session.
    asyncio.create_task(narrate_phase(session))
    await session.generate_reply(
        instructions="Say exactly: G1 on the line, ready."
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
