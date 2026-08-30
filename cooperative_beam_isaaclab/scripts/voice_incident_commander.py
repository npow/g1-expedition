#!/usr/bin/env python3
"""LiveKit voice agent backed by ElevenLabs STT/TTS and real G1 telemetry."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cooperative_beam_isaaclab.tasks.incident_commander import (
    IncidentCommander,
    SupervisoryIntent,
    is_explicit_confirmation,
)
from cooperative_beam_isaaclab.tasks.livekit_state_bus import (
    SUPERVISORY_RPC_METHOD,
    TRACK_NAME,
    RobotCoordinationState,
)

try:  # Keep --help and the credential-free console usable without this extra.
    from livekit import agents
    from livekit.agents import Agent, AgentServer, AgentSession, RunContext, function_tool, inference
    from livekit.plugins import elevenlabs
except ImportError as exc:  # pragma: no cover - exercised only without the optional runtime
    raise SystemExit(
        "Install the voice extra first: uv sync --extra voice\n"
        "For a credential-free demo, run scripts/incident_commander_console.py"
    ) from exc


TEAM = tuple(
    dict.fromkeys(
        item.strip() for item in os.environ.get("INCIDENT_TEAM_IDS", "g1_0,g1_1,g1_2").split(",") if item.strip()
    )
)
DEMO_MODE = os.environ.get("INCIDENT_DEMO_MODE", "0").lower() in {"1", "true", "yes"}
MAX_AGE_MS = float(os.environ.get("INCIDENT_MAX_AGE_MS", "150"))
VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
LLM_MODEL = os.environ.get("INCIDENT_LLM_MODEL", "google/gemma-4-31b-it")


def require_voice_environment() -> None:
    required = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "ELEVEN_API_KEY")
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    placeholders = [name for name in required if os.environ.get(name, "").strip() in {"...", "replace-me"}]
    if missing or placeholders:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if placeholders:
            details.append("placeholder values in " + ", ".join(placeholders))
        raise RuntimeError("Voice session credentials are not configured: " + "; ".join(details))


class DemoTelemetry:
    def __init__(self, commander: IncidentCommander) -> None:
        self.commander = commander
        self.sequence = 0

    def refresh(self) -> None:
        now = time.monotonic_ns()
        count = len(self.commander.expected_robot_ids)
        loads = [0.25, 0.50, 0.25] if count == 3 else [1.0 / count] * count
        for index, (robot_id, load) in enumerate(zip(self.commander.expected_robot_ids, loads, strict=True)):
            self.commander.ingest_state(
                RobotCoordinationState(
                    robot_id=robot_id,
                    sequence=self.sequence,
                    sender_time_ns=time.time_ns(),
                    position_w=(0.35 * index, (-1.0) ** index * 0.45, 0.82),
                    linear_velocity_w=(0.08, 0.0, 0.0),
                    orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
                    load_ratio=load,
                ),
                received_monotonic_ns=now,
            )
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF


class TelemetrySubscriber:
    def __init__(self, room, commander: IncidentCommander) -> None:
        self.room = room
        self.commander = commander
        self.tasks: set[asyncio.Task] = set()

        @room.on("data_track_published")
        def on_data_track_published(track) -> None:
            if track.info.name != TRACK_NAME:
                return
            task = asyncio.create_task(self._consume(track))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

    async def _consume(self, track) -> None:
        try:
            async for frame in track.subscribe():
                try:
                    state = RobotCoordinationState.decode(bytes(frame.payload))
                except ValueError:
                    continue
                if state.robot_id == track.publisher_identity:
                    self.commander.ingest_state(state)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a robot may leave without ending the voice session
            return

    async def close(self) -> None:
        for task in tuple(self.tasks):
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)


@dataclass(frozen=True)
class CommandDelivery:
    acknowledged_robot_ids: tuple[str, ...]
    failures: tuple[str, ...]
    total_robot_count: int
    demo_mode: bool = False

    @property
    def complete(self) -> bool:
        return self.demo_mode or len(self.acknowledged_robot_ids) == self.total_robot_count

    def summary(self) -> str:
        if self.demo_mode:
            return "Demo telemetry mode: no robot RPC was sent."
        message = (
            f"Supervisory command acknowledged by {len(self.acknowledged_robot_ids)} "
            f"of {self.total_robot_count} robot bridges."
        )
        if self.failures:
            message += " No acknowledgement from " + ", ".join(self.failures) + "."
        return message


class RpcCommandBroadcaster:
    def __init__(self, room, robot_ids: tuple[str, ...], demo_mode: bool) -> None:
        self.room = room
        self.robot_ids = robot_ids
        self.demo_mode = demo_mode

    async def send(self, intent: SupervisoryIntent) -> CommandDelivery:
        if self.demo_mode:
            return CommandDelivery((), (), len(self.robot_ids), demo_mode=True)
        command_id = str(uuid.uuid4())
        payload = json.dumps(
            {
                "version": 1,
                "command_id": command_id,
                "intent": intent.value,
                "issued_at_ns": time.time_ns(),
                "source": self.room.local_participant.identity,
            },
            separators=(",", ":"),
        )

        async def send_one(robot_id: str) -> tuple[str, bool, str]:
            try:
                raw = await self.room.local_participant.perform_rpc(
                    destination_identity=robot_id,
                    method=SUPERVISORY_RPC_METHOD,
                    payload=payload,
                    response_timeout=8.0,
                    max_round_trip_latency=2.0,
                )
                response = json.loads(raw)
                accepted = bool(response.get("accepted"))
                return robot_id, accepted, str(response.get("reason", "queued"))
            except Exception as exc:  # noqa: BLE001 - summarize per-robot transport failure
                return robot_id, False, type(exc).__name__

        results = await asyncio.gather(*(send_one(robot_id) for robot_id in self.robot_ids))
        acknowledgements = tuple(robot_id for robot_id, accepted, _ in results if accepted)
        failures = tuple(f"{robot_id} ({reason})" for robot_id, accepted, reason in results if not accepted)
        return CommandDelivery(acknowledgements, failures, len(results))


def _latest_user_utterance(context: RunContext) -> str:
    for item in reversed(context.session.history.items):
        if getattr(item, "role", None) == "user":
            return getattr(item, "text_content", None) or ""
    return ""


class ExpeditionVoiceAgent(Agent):
    def __init__(
        self,
        commander: IncidentCommander,
        broadcaster: RpcCommandBroadcaster,
        demo: DemoTelemetry | None,
    ) -> None:
        mode = "DEMO TELEMETRY FIXTURE" if demo is not None else "LIVE TELEMETRY"
        super().__init__(
            instructions=f"""
You are Expedition Control, a concise radio operator for a simulated Unitree G1 rescue team.
Session mode: {mode}. Say the mode whenever the user asks whether data is live.
Never invent telemetry, success, safety, hardware capability, or command delivery. Use a tool for every
robot-state claim. Speak in short radio sentences without markdown.

The language model never controls joints. It may request only start, hold, resume, or abort through
request_team_command. Start and resume require two turns: first request the intent, then ask the operator
to say exactly confirm start or confirm resume. The deterministic tool reads the user transcript and decides
whether confirmation is valid; you cannot set or infer it. Abort needs no confirmation. Never describe an
acknowledged or queued RPC as applied motion.
Emergency stop is local and physical; this voice interface is not an emergency stop.
Fault injection is available only when the tool says demo mode is enabled.
""".strip()
        )
        self.commander = commander
        self.broadcaster = broadcaster
        self.demo = demo

    def _refresh_demo(self) -> None:
        if self.demo is not None:
            self.demo.refresh()

    @function_tool()
    async def get_team_status(self, context: RunContext) -> str:
        """Read mission phase, coordination-link freshness, and all current load shares."""
        self._refresh_demo()
        return self.commander.status()

    @function_tool()
    async def assess_team_load(self, context: RunContext) -> str:
        """Assess current load balance using the deterministic simulation gate."""
        self._refresh_demo()
        return self.commander.assess_load()

    @function_tool()
    async def get_robot_load(self, context: RunContext, robot_id: str) -> str:
        """Read one robot's current load share and speed.

        Args:
            robot_id: Canonical team identity such as g1_1.
        """
        self._refresh_demo()
        return self.commander.robot_load(robot_id)

    @function_tool()
    async def explain_last_transition(self, context: RunContext) -> str:
        """Explain the last accepted, rejected, or automatic mission-state transition."""
        return self.commander.explain_last_transition()

    @function_tool()
    async def request_team_command(self, context: RunContext, intent: str) -> str:
        """Request a bounded supervisory intent after deterministic safety checks.

        Args:
            intent: Exactly one of start, hold, resume, or abort.
        """
        self._refresh_demo()
        try:
            parsed = SupervisoryIntent(intent.lower())
        except ValueError:
            return "Rejected. Intent must be start, hold, resume, or abort."
        confirmed = is_explicit_confirmation(_latest_user_utterance(context), parsed)
        decision = self.commander.request_command(parsed, confirmed=confirmed)
        if not decision.accepted:
            return decision.message
        delivery = await self.broadcaster.send(parsed)
        if not delivery.complete and parsed != SupervisoryIntent.ABORT:
            abort = self.commander.request_command(SupervisoryIntent.ABORT)
            abort_delivery = await self.broadcaster.send(SupervisoryIntent.ABORT)
            return (
                f"{decision.message} Delivery was incomplete, so the supervisor failed closed. "
                f"{delivery.summary()} {abort.message} {abort_delivery.summary()}"
            )
        return f"{decision.message} {delivery.summary()}"

    @function_tool()
    async def inject_demo_link_loss(self, context: RunContext, robot_id: str) -> str:
        """In demo mode only, mask one robot's telemetry to demonstrate stale-link handling.

        Args:
            robot_id: Canonical team identity such as g1_1.
        """
        response = self.commander.inject_link_loss(robot_id)
        if self.commander.phase.value == "aborted":
            delivery = await self.broadcaster.send(SupervisoryIntent.ABORT)
            response += " " + delivery.summary()
        return response

    @function_tool()
    async def restore_demo_link(self, context: RunContext, robot_id: str) -> str:
        """In demo mode only, restore telemetry previously masked by fault injection.

        Args:
            robot_id: Canonical team identity such as g1_1.
        """
        self._refresh_demo()
        return self.commander.restore_link(robot_id)


async def freshness_watchdog(
    commander: IncidentCommander,
    broadcaster: RpcCommandBroadcaster,
    session: AgentSession,
    demo: DemoTelemetry | None,
) -> None:
    """Continuously fail an active mission closed when coordination state expires."""
    interval_s = max(0.025, min(commander.max_age_ms / 2_000.0, 0.25))
    while True:
        await asyncio.sleep(interval_s)
        if demo is not None:
            demo.refresh()
        decision = commander.enforce_freshness()
        if decision is None or not decision.accepted:
            continue
        delivery = await broadcaster.send(SupervisoryIntent.ABORT)
        session.say(
            f"{decision.message} {delivery.summary()}",
            allow_interruptions=False,
        )


server = AgentServer()


@server.rtc_session(agent_name="g1-incident-commander")
async def incident_commander_session(ctx: agents.JobContext) -> None:
    require_voice_environment()
    commander = IncidentCommander(TEAM, max_age_ms=MAX_AGE_MS, demo_mode=DEMO_MODE)
    demo = DemoTelemetry(commander) if DEMO_MODE else None
    if demo is not None:
        demo.refresh()
    subscriber = None if DEMO_MODE else TelemetrySubscriber(ctx.room, commander)
    broadcaster = RpcCommandBroadcaster(ctx.room, TEAM, DEMO_MODE)
    session = AgentSession(
        stt=elevenlabs.STT(model="scribe_v2_realtime"),
        llm=inference.LLM(model=LLM_MODEL),
        tts=elevenlabs.TTS(voice_id=VOICE_ID, model="eleven_flash_v2_5"),
    )
    await session.start(
        room=ctx.room,
        agent=ExpeditionVoiceAgent(commander, broadcaster, demo),
    )
    watchdog_task = asyncio.create_task(
        freshness_watchdog(commander, broadcaster, session, demo),
        name="incident-commander-freshness-watchdog",
    )

    async def close_incident_commander() -> None:
        watchdog_task.cancel()
        await asyncio.gather(watchdog_task, return_exceptions=True)
        if subscriber is not None:
            await subscriber.close()

    ctx.add_shutdown_callback(close_incident_commander)
    mode = "demo telemetry fixture" if DEMO_MODE else "live telemetry"
    await session.generate_reply(
        instructions=f"Say: Expedition Control online in {mode} mode. Ask for team status or load assessment."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
