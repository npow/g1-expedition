from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("livekit.agents")

from cooperative_beam_isaaclab.tasks.incident_commander import (
    IncidentCommander,
    MissionPhase,
    SupervisoryIntent,
)
from cooperative_beam_isaaclab.tasks.livekit_state_bus import RobotCoordinationState

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "voice_incident_commander.py"
SPEC = importlib.util.spec_from_file_location("voice_incident_commander", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
VOICE_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VOICE_MODULE
SPEC.loader.exec_module(VOICE_MODULE)
CommandDelivery = VOICE_MODULE.CommandDelivery
RpcCommandBroadcaster = VOICE_MODULE.RpcCommandBroadcaster
freshness_watchdog = VOICE_MODULE.freshness_watchdog
require_voice_environment = VOICE_MODULE.require_voice_environment


class FakeBroadcaster:
    def __init__(self) -> None:
        self.intents: list[SupervisoryIntent] = []

    async def send(self, intent: SupervisoryIntent) -> CommandDelivery:
        self.intents.append(intent)
        return CommandDelivery(("g1_0", "g1_1"), (), 2)


class FakeSession:
    def __init__(self) -> None:
        self.announcements: list[str] = []

    def say(self, text: str, **_kwargs) -> None:
        self.announcements.append(text)


def test_delivery_requires_every_robot_acknowledgement() -> None:
    assert CommandDelivery(("g1_0", "g1_1"), (), 2).complete
    partial = CommandDelivery(("g1_0",), ("g1_1 (timeout)",), 2)
    assert not partial.complete
    assert "1 of 2" in partial.summary()
    assert CommandDelivery((), (), 2, demo_mode=True).complete


def test_voice_environment_rejects_missing_or_placeholder_credentials(monkeypatch) -> None:
    names = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "ELEVEN_API_KEY")
    for name in names:
        monkeypatch.setenv(name, "configured")
    require_voice_environment()

    monkeypatch.delenv("ELEVEN_API_KEY")
    with pytest.raises(RuntimeError, match="missing ELEVEN_API_KEY"):
        require_voice_environment()

    monkeypatch.setenv("ELEVEN_API_KEY", "...")
    with pytest.raises(RuntimeError, match="placeholder values in ELEVEN_API_KEY"):
        require_voice_environment()


def test_rpc_envelope_uses_authenticated_local_identity() -> None:
    class LocalParticipant:
        identity = "authorized-commander"

        def __init__(self) -> None:
            self.payloads: list[dict[str, object]] = []

        async def perform_rpc(self, **kwargs) -> str:
            self.payloads.append(json.loads(kwargs["payload"]))
            return '{"accepted":true,"queued":true}'

    class Room:
        local_participant = LocalParticipant()

    async def scenario() -> None:
        room = Room()
        broadcaster = RpcCommandBroadcaster(room, ("g1_0", "g1_1"), demo_mode=False)
        delivery = await broadcaster.send(SupervisoryIntent.HOLD)
        assert delivery.complete
        assert len(room.local_participant.payloads) == 2
        assert {payload["source"] for payload in room.local_participant.payloads} == {"authorized-commander"}
        assert {payload["intent"] for payload in room.local_participant.payloads} == {"hold"}
        assert len({payload["command_id"] for payload in room.local_participant.payloads}) == 1

    asyncio.run(scenario())


def test_watchdog_broadcasts_and_announces_abort_after_state_expires() -> None:
    async def scenario() -> None:
        commander = IncidentCommander(("g1_0", "g1_1"), max_age_ms=5.0)
        received = time.monotonic_ns()
        for index, robot_id in enumerate(commander.expected_robot_ids):
            commander.ingest_state(
                RobotCoordinationState(
                    robot_id=robot_id,
                    sequence=1,
                    sender_time_ns=time.time_ns(),
                    position_w=(float(index), 0.0, 0.8),
                    linear_velocity_w=(0.0, 0.0, 0.0),
                    orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
                    load_ratio=0.5,
                ),
                received_monotonic_ns=received,
            )
        commander.request_command(SupervisoryIntent.START, now_monotonic_ns=received)
        started = commander.request_command(
            SupervisoryIntent.START,
            confirmed=True,
            now_monotonic_ns=received,
        )
        assert started.accepted

        broadcaster = FakeBroadcaster()
        session = FakeSession()
        task = asyncio.create_task(freshness_watchdog(commander, broadcaster, session, None))
        try:
            await asyncio.sleep(0.08)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        assert commander.phase == MissionPhase.ABORTED
        assert broadcaster.intents == [SupervisoryIntent.ABORT]
        assert len(session.announcements) == 1
        assert "Automatic abort" in session.announcements[0]
        assert "2 of 2 robot bridges" in session.announcements[0]

    asyncio.run(scenario())
