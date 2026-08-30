from __future__ import annotations

import math

from cooperative_beam_isaaclab.tasks.incident_commander import (
    IncidentCommander,
    MissionPhase,
    SupervisoryIntent,
    is_explicit_confirmation,
)
from cooperative_beam_isaaclab.tasks.livekit_state_bus import RobotCoordinationState


def state(robot_id: str, sequence: int, load: float) -> RobotCoordinationState:
    index = int(robot_id.rsplit("_", 1)[1])
    return RobotCoordinationState(
        robot_id=robot_id,
        sequence=sequence,
        sender_time_ns=123,
        position_w=(float(index), 0.0, 0.8),
        linear_velocity_w=(0.1, 0.0, 0.0),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        load_ratio=load,
    )


def ready_commander(*, demo_mode: bool = False, sink=None) -> IncidentCommander:
    commander = IncidentCommander(("g1_0", "g1_1", "g1_2"), demo_mode=demo_mode, command_sink=sink)
    commander.ingest_state(state("g1_0", 1, 0.25), received_monotonic_ns=1_000)
    commander.ingest_state(state("g1_1", 1, 0.50), received_monotonic_ns=1_000)
    commander.ingest_state(state("g1_2", 1, 0.25), received_monotonic_ns=1_000)
    return commander


def test_status_and_load_assessment_are_grounded_in_fresh_telemetry() -> None:
    commander = ready_commander()
    status = commander.status(now_monotonic_ns=2_000)
    assert "3 of 3 robot links are fresh" in status
    assert "G1 0 25 percent" in status
    assert "G1 1 50 percent" in status
    assessment = commander.assess_load(now_monotonic_ns=2_000)
    assert "within the configured simulation gate" in assessment
    assert "G1 1 at 50 percent" in assessment


def test_start_requires_a_pending_request_then_explicit_confirmation() -> None:
    commands: list[SupervisoryIntent] = []
    commander = ready_commander(sink=commands.append)
    direct_confirmation = commander.request_command(SupervisoryIntent.START, confirmed=True, now_monotonic_ns=2_000)
    assert not direct_confirmation.accepted
    assert direct_confirmation.confirmation_required
    assert "No pending start request" in commander.explain_last_transition()

    request = commander.request_command(SupervisoryIntent.START, now_monotonic_ns=2_000)
    assert not request.accepted
    assert request.confirmation_required
    assert commander.phase == MissionPhase.READY

    confirmed = commander.request_command(SupervisoryIntent.START, confirmed=True, now_monotonic_ns=2_000)
    assert confirmed.accepted
    assert commander.phase == MissionPhase.ACTIVE
    assert commands == [SupervisoryIntent.START]


def test_stale_teammate_blocks_start() -> None:
    commander = ready_commander()
    commander.request_command(SupervisoryIntent.START, now_monotonic_ns=2_000)
    decision = commander.request_command(
        SupervisoryIntent.START,
        confirmed=True,
        now_monotonic_ns=200_000_000,
    )
    assert not decision.accepted
    assert "stale or missing" in decision.message
    assert commander.phase == MissionPhase.READY


def test_hold_is_accepted_when_telemetry_is_stale() -> None:
    commander = ready_commander()
    commander.request_command(SupervisoryIntent.START, now_monotonic_ns=2_000)
    commander.request_command(SupervisoryIntent.START, confirmed=True, now_monotonic_ns=2_000)
    decision = commander.request_command(SupervisoryIntent.HOLD, now_monotonic_ns=200_000_000)
    assert decision.accepted
    assert commander.phase == MissionPhase.HOLD


def test_hold_from_ready_cannot_create_an_alternate_start_path() -> None:
    commander = ready_commander()
    hold = commander.request_command(SupervisoryIntent.HOLD, now_monotonic_ns=2_000)
    assert not hold.accepted
    assert commander.phase == MissionPhase.READY

    resume = commander.request_command(SupervisoryIntent.RESUME, now_monotonic_ns=2_000)
    assert not resume.accepted
    assert commander.phase == MissionPhase.READY


def test_activation_rejects_overloaded_or_incomplete_telemetry() -> None:
    overloaded = IncidentCommander(("g1_0", "g1_1"))
    overloaded.ingest_state(state("g1_0", 1, 0.7), received_monotonic_ns=1_000)
    overloaded.ingest_state(state("g1_1", 1, 0.3), received_monotonic_ns=1_000)
    decision = overloaded.request_command(SupervisoryIntent.START, now_monotonic_ns=2_000)
    assert not decision.accepted
    assert "above the configured 65 percent limit" in decision.message

    incomplete = IncidentCommander(("g1_0", "g1_1"))
    incomplete.ingest_state(state("g1_0", 1, 0.2), received_monotonic_ns=1_000)
    incomplete.ingest_state(state("g1_1", 1, 0.2), received_monotonic_ns=1_000)
    decision = incomplete.request_command(SupervisoryIntent.START, now_monotonic_ns=2_000)
    assert not decision.accepted
    assert "sum to 40 percent" in decision.message


def test_confirmation_phrase_and_window_are_deterministic() -> None:
    assert is_explicit_confirmation("Confirm start.", SupervisoryIntent.START)
    assert is_explicit_confirmation("confirm resume", SupervisoryIntent.RESUME)
    assert not is_explicit_confirmation("yes, confirm start", SupervisoryIntent.START)
    assert not is_explicit_confirmation("confirm resume", SupervisoryIntent.START)
    assert not is_explicit_confirmation("confirm abort", SupervisoryIntent.ABORT)

    commander = IncidentCommander(("g1_0", "g1_1", "g1_2"), confirmation_window_s=0.001)
    commander.ingest_state(state("g1_0", 1, 0.25), received_monotonic_ns=1_000)
    commander.ingest_state(state("g1_1", 1, 0.50), received_monotonic_ns=1_000)
    commander.ingest_state(state("g1_2", 1, 0.25), received_monotonic_ns=1_000)
    commander.request_command(SupervisoryIntent.START, now_monotonic_ns=2_000)
    expired = commander.request_command(
        SupervisoryIntent.START,
        confirmed=True,
        now_monotonic_ns=2_000_000,
    )
    assert not expired.accepted
    assert "confirmation window expired" in expired.message


def test_invalid_telemetry_is_rejected() -> None:
    commander = ready_commander()
    assert not commander.ingest_state(state("g1_1", 2, math.nan))
    assert not commander.ingest_state(state("g1_1", 2, 1.1))


def test_abort_is_accepted_even_when_every_link_is_stale() -> None:
    commands: list[SupervisoryIntent] = []
    commander = ready_commander(sink=commands.append)
    decision = commander.request_command(SupervisoryIntent.ABORT, now_monotonic_ns=200_000_000)
    assert decision.accepted
    assert commander.phase == MissionPhase.ABORTED
    assert commands == [SupervisoryIntent.ABORT]


def test_active_mission_watchdog_aborts_on_stale_telemetry() -> None:
    commands: list[SupervisoryIntent] = []
    commander = ready_commander(sink=commands.append)
    commander.request_command(SupervisoryIntent.START, now_monotonic_ns=2_000)
    commander.request_command(SupervisoryIntent.START, confirmed=True, now_monotonic_ns=2_000)
    decision = commander.enforce_freshness(now_monotonic_ns=200_000_000)
    assert decision is not None and decision.accepted
    assert "150 millisecond deadline" in decision.message
    assert commander.phase == MissionPhase.ABORTED
    assert commands == [SupervisoryIntent.START, SupervisoryIntent.ABORT]


def test_demo_link_loss_aborts_active_mission_and_explains_why() -> None:
    commands: list[SupervisoryIntent] = []
    commander = ready_commander(demo_mode=True, sink=commands.append)
    commander.request_command(SupervisoryIntent.START, now_monotonic_ns=2_000)
    commander.request_command(SupervisoryIntent.START, confirmed=True, now_monotonic_ns=2_000)

    response = commander.inject_link_loss("g1-1")
    assert "demo-only" in response.lower()
    assert commander.phase == MissionPhase.ABORTED
    assert "G1 1" in commander.explain_last_transition()
    assert commands == [SupervisoryIntent.START, SupervisoryIntent.ABORT]


def test_fault_injection_is_disabled_in_live_mode() -> None:
    commander = ready_commander(demo_mode=False)
    assert commander.inject_link_loss("g1_1") == "Fault injection is disabled outside demo mode."


def test_out_of_order_state_does_not_replace_latest_reading() -> None:
    commander = ready_commander()
    assert commander.ingest_state(state("g1_1", 2, 0.4), received_monotonic_ns=2_000)
    assert not commander.ingest_state(state("g1_1", 1, 0.9), received_monotonic_ns=3_000)
    assert "40 percent" in commander.robot_load("g1 1", now_monotonic_ns=4_000)
