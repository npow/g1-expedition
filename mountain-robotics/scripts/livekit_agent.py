"""Optional LiveKit voice agent for the Alpine Lift control endpoint.

Start ``scripts/voice_demo.py`` first, authenticate the LiveKit CLI, then run:

    uv run scripts/livekit_agent.py console

The agent converts natural speech into one of two narrow function calls.  The
actual motion boundary remains in the local control service, so an LLM cannot
invent arbitrary robot actions.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    function_tool,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env.local"))

CONTROL_URL = os.environ.get("ALPINE_CONTROL_URL", "http://127.0.0.1:8765")
CONTROL_TOKEN = os.environ.get("ALPINE_CONTROL_TOKEN", "")


def _control(phrase: str) -> dict:
    body = json.dumps({"phrase": phrase}).encode()
    headers = {"Content-Type": "application/json", "X-Alpine-Voice": "livekit"}
    if CONTROL_TOKEN:
        headers["Authorization"] = f"Bearer {CONTROL_TOKEN}"
    request = urllib.request.Request(
        f"{CONTROL_URL.rstrip('/')}/api/command",
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=4) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            detail = json.load(exc)
        except Exception:
            detail = {"message": str(exc)}
        return {"ok": False, **detail}
    except urllib.error.URLError as exc:
        return {"ok": False, "message": f"Robot control service unavailable: {exc.reason}"}


def _mark_livekit_ready() -> None:
    headers = {"X-Alpine-Voice": "livekit"}
    if CONTROL_TOKEN:
        headers["Authorization"] = f"Bearer {CONTROL_TOKEN}"
    request = urllib.request.Request(
        f"{CONTROL_URL.rstrip('/')}/api/livekit/ready",
        data=b"{}",
        headers=headers,
        method="POST",
    )
    try:
        urllib.request.urlopen(request, timeout=4).close()
    except urllib.error.URLError:
        pass


class AlpineOperator(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are the concise voice interface for a two-robot alpine lift operating in extreme Himalayan conditions. "
                "You have access to safe robot control tools: "
                "1. lift_log: when operator clearly asks to lift, raise, hoist, or pick up the log. "
                "2. operator_stop: immediately when operator asks to stop, abort, halt, or stand down. "
                "3. add_wind_gust: when operator asks to add, simulate, or inject a 45 N wind gust. "
                "4. simulate_verglas_ice: when operator asks to simulate verglas, ice, black ice, or slippery terrain. "
                "5. test_heavy_load: when operator asks to test or evaluate a heavy, overweight, or 30 kg obstacle to test the autonomous weigh-in safety decline. "
                "6. telemetry_status: when operator asks for status, telemetry, diagnostics, or readout. "
                "7. reset_system: when operator asks to reset, re-arm, or return to ready stance. "
                "Never claim motion succeeded until a tool confirms acceptance. Be concise and crisp in your voice replies."
            )
        )

    @function_tool()
    async def lift_log(self, context: RunContext) -> dict:
        """Start the nominal coordinated lift after an explicit operator request."""
        context.disallow_interruptions()
        return await asyncio.to_thread(_control, "lift the log")

    @function_tool()
    async def test_heavy_load(self, context: RunContext) -> dict:
        """Evaluate a 30 kg heavy obstacle to demonstrate autonomous weighing and safety decline."""
        context.disallow_interruptions()
        return await asyncio.to_thread(_control, "test heavy load")

    @function_tool()
    async def operator_stop(self, context: RunContext) -> dict:
        """Request a controlled set-down immediately when the operator says stop or abort."""
        context.disallow_interruptions()
        return await asyncio.to_thread(_control, "operator stop")

    @function_tool()
    async def add_wind_gust(self, context: RunContext) -> dict:
        """Inject a 45 N wind gust during the loaded phase of an active lift."""
        context.disallow_interruptions()
        return await asyncio.to_thread(_control, "add a wind gust")

    @function_tool()
    async def simulate_verglas_ice(self, context: RunContext) -> dict:
        """Simulate Himalayan verglas surface ice (friction drop to mu=0.45) underfoot."""
        context.disallow_interruptions()
        return await asyncio.to_thread(_control, "simulate verglas")

    @function_tool()
    async def telemetry_status(self, context: RunContext) -> dict:
        """Report live telemetry including phase, estimated mass, tilt angle, load sharing, and peak force."""
        context.disallow_interruptions()
        return await asyncio.to_thread(_control, "status report")

    @function_tool()
    async def reset_system(self, context: RunContext) -> dict:
        """Reset the simulation and robots back to the armed ready stance."""
        context.disallow_interruptions()
        return await asyncio.to_thread(_control, "reset system")


server = AgentServer()


@server.rtc_session(agent_name="alpine-lift-operator")
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        stt="deepgram/nova-3:en",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
    )
    await session.start(room=ctx.room, agent=AlpineOperator())
    await ctx.connect()
    await asyncio.to_thread(_mark_livekit_ready)
    await session.generate_reply(
        instructions=(
            "Briefly say that Alpine Lift is armed with LiveKit voice. "
            "Mention you can lift the log, test a heavy load, simulate verglas ice, inject wind gusts, or check status."
        )
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
