from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from cooperative_beam_isaaclab.tasks.livekit_state_bus import (
    LOCAL_OBSERVATION_DIM,
    LatestStateTable,
    LiveKitRobotStateClient,
    RobotCoordinationState,
    assemble_policy_observation,
    teammate_observation_parts,
)


def state(
    robot_id: str,
    sequence: int,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    load: float = 0.5,
) -> RobotCoordinationState:
    return RobotCoordinationState(robot_id, sequence, 123, position, velocity, orientation, load)


def test_binary_packet_round_trip() -> None:
    original = state("g1_2", 7, (1.25, -0.5, 2.0), (0.1, 0.2, 0.3), load=0.25)
    decoded = RobotCoordinationState.decode(original.encode())
    assert decoded.robot_id == original.robot_id
    assert decoded.sequence == original.sequence
    assert decoded.position_w == pytest.approx(original.position_w)
    assert decoded.linear_velocity_w == pytest.approx(original.linear_velocity_w)
    assert decoded.orientation_wxyz == pytest.approx(original.orientation_wxyz)
    assert decoded.load_ratio == pytest.approx(original.load_ratio)


def test_older_packets_do_not_replace_latest_state() -> None:
    table = LatestStateTable()
    assert table.update(state("g1_1", 4, (4.0, 0.0, 0.0)), received_monotonic_ns=100)
    assert not table.update(state("g1_1", 3, (3.0, 0.0, 0.0)), received_monotonic_ns=200)
    latest = table.fresh("g1_1", max_age_ms=1.0, now_monotonic_ns=300)
    assert latest is not None
    assert latest.sequence == 4


def test_teammate_state_is_rotated_into_recipient_frame() -> None:
    # Recipient yaw is +90 degrees, so world +Y is body +X.
    half = math.sqrt(0.5)
    recipient = state("g1_0", 1, (0.0, 0.0, 0.0), orientation=(half, 0.0, 0.0, half))
    teammate = state("g1_1", 1, (0.0, 2.0, 0.0), (0.0, 1.0, 0.0), load=0.4)
    table = LatestStateTable()
    table.update(teammate, received_monotonic_ns=1_000)
    kinematics, loads, fresh = teammate_observation_parts(
        recipient, ["g1_1"], table, max_age_ms=10.0, now_monotonic_ns=2_000
    )
    assert kinematics == pytest.approx([2.0, 0.0, 0.0, 1.0, 0.0, 0.0], abs=1.0e-6)
    assert loads == pytest.approx([0.4])
    assert fresh == [True]


def test_stale_state_is_zero_filled_without_changing_checkpoint_width() -> None:
    recipient = state("g1_0", 1, (0.0, 0.0, 0.0))
    table = LatestStateTable()
    table.update(state("g1_1", 1, (1.0, 0.0, 0.0)), received_monotonic_ns=0)
    observation, fresh = assemble_policy_observation(
        [0.25] * LOCAL_OBSERVATION_DIM,
        recipient,
        ["g1_1", "g1_2"],
        table,
        max_age_ms=0.1,
        now_monotonic_ns=1_000_000,
    )
    assert len(observation) == LOCAL_OBSERVATION_DIM + 2 * 7
    assert observation[:LOCAL_OBSERVATION_DIM] == [0.25] * LOCAL_OBSERVATION_DIM
    assert observation[LOCAL_OBSERVATION_DIM:] == [0.0] * 14
    assert fresh == [False, False]


def test_supervisory_rpc_requires_authorized_caller_and_deduplicates() -> None:
    client = LiveKitRobotStateClient(
        "g1_0", "wss://example.invalid", "token", command_callers=("incident-commander",)
    )
    payload = json.dumps({"command_id": "command-1", "intent": "hold", "issued_at_ns": 123})
    unauthorized = json.loads(
        client._receive_command(SimpleNamespace(caller_identity="stranger", payload=payload))
    )
    assert not unauthorized["accepted"]
    assert client.drain_commands() == []

    invocation = SimpleNamespace(caller_identity="incident-commander", payload=payload)
    accepted = json.loads(client._receive_command(invocation))
    duplicate = json.loads(client._receive_command(invocation))
    assert accepted == {"accepted": True, "queued": True, "robot_id": "g1_0"}
    assert duplicate == {"accepted": True, "duplicate": True, "robot_id": "g1_0"}
    assert client.drain_commands() == [json.loads(payload)]
    assert client.drain_commands() == []
