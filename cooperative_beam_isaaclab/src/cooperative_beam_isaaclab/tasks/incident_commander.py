"""Safety-gated, telemetry-grounded interface for the expedition voice agent.

The language model never emits joint targets. It can only ask this module for
status or request one of a small set of supervisory intents. The deterministic
state machine below owns admission, confirmation, freshness, and demo-only
fault injection.
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum

from .livekit_state_bus import RobotCoordinationState


class MissionPhase(StrEnum):
    READY = "ready"
    ACTIVE = "active"
    HOLD = "hold"
    ABORTED = "aborted"


class SupervisoryIntent(StrEnum):
    START = "start"
    HOLD = "hold"
    RESUME = "resume"
    ABORT = "abort"


@dataclass(frozen=True)
class CommandDecision:
    intent: SupervisoryIntent
    accepted: bool
    confirmation_required: bool
    phase: MissionPhase
    message: str


@dataclass(frozen=True)
class TeamSnapshot:
    phase: MissionPhase
    fresh: tuple[RobotCoordinationState, ...]
    stale_robot_ids: tuple[str, ...]
    masked_robot_ids: tuple[str, ...]


CommandSink = Callable[[SupervisoryIntent], None]


def _sequence_is_newer(candidate: int, previous: int) -> bool:
    delta = (candidate - previous) & 0xFFFFFFFF
    return 0 < delta < 0x80000000


def _spoken_robot_id(robot_id: str) -> str:
    return robot_id.replace("_", " ").upper()


def is_explicit_confirmation(utterance: str, intent: SupervisoryIntent | str) -> bool:
    """Accept only the exact, independently transcribed activation phrase."""
    try:
        parsed = SupervisoryIntent(str(intent).lower())
    except ValueError:
        return False
    if parsed not in (SupervisoryIntent.START, SupervisoryIntent.RESUME):
        return False
    normalized = " ".join(re.findall(r"[a-z0-9]+", utterance.lower()))
    return normalized == f"confirm {parsed.value}"


class IncidentCommander:
    """Own live team state and admit only bounded supervisory commands."""

    def __init__(
        self,
        expected_robot_ids: Iterable[str],
        *,
        max_age_ms: float = 150.0,
        max_load_share: float = 0.65,
        confirmation_window_s: float = 15.0,
        demo_mode: bool = False,
        command_sink: CommandSink | None = None,
    ) -> None:
        expected = tuple(dict.fromkeys(expected_robot_ids))
        if len(expected) < 2:
            raise ValueError("Cooperative incident command requires at least two robot identities")
        if max_age_ms <= 0:
            raise ValueError("max_age_ms must be positive")
        if not 0.5 < max_load_share <= 1.0:
            raise ValueError("max_load_share must be in (0.5, 1.0]")
        if confirmation_window_s <= 0:
            raise ValueError("confirmation_window_s must be positive")
        self.expected_robot_ids = expected
        self.max_age_ms = float(max_age_ms)
        self.max_load_share = float(max_load_share)
        self.confirmation_window_s = float(confirmation_window_s)
        self.demo_mode = bool(demo_mode)
        self._command_sink = command_sink
        self._lock = threading.Lock()
        self._states: dict[str, tuple[RobotCoordinationState, int]] = {}
        self._masked_robot_ids: set[str] = set()
        self._phase = MissionPhase.READY
        self._pending_confirmation: SupervisoryIntent | None = None
        self._pending_confirmation_deadline_ns: int | None = None
        self._last_transition = "Mission initialized in ready state."

    @property
    def phase(self) -> MissionPhase:
        with self._lock:
            return self._phase

    def ingest_state(
        self,
        state: RobotCoordinationState,
        *,
        received_monotonic_ns: int | None = None,
    ) -> bool:
        """Accept a current team frame and reject unknown/out-of-order data."""
        if state.robot_id not in self.expected_robot_ids:
            return False
        values = (
            *state.position_w,
            *state.linear_velocity_w,
            *state.orientation_wxyz,
            state.load_ratio,
        )
        if (
            not all(math.isfinite(value) for value in values)
            or not 0.0 <= state.load_ratio <= 1.0
            or math.sqrt(sum(value * value for value in state.orientation_wxyz)) < 1.0e-6
        ):
            return False
        received = time.monotonic_ns() if received_monotonic_ns is None else int(received_monotonic_ns)
        with self._lock:
            previous = self._states.get(state.robot_id)
            if previous is not None and not _sequence_is_newer(state.sequence, previous[0].sequence):
                return False
            self._states[state.robot_id] = (state, received)
        return True

    def snapshot(self, *, now_monotonic_ns: int | None = None) -> TeamSnapshot:
        now = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
        maximum_age_ns = int(self.max_age_ms * 1_000_000)
        with self._lock:
            phase = self._phase
            entries = dict(self._states)
            masked = set(self._masked_robot_ids)
        fresh: list[RobotCoordinationState] = []
        stale: list[str] = []
        for robot_id in self.expected_robot_ids:
            entry = entries.get(robot_id)
            if robot_id in masked or entry is None or now - entry[1] > maximum_age_ns:
                stale.append(robot_id)
            else:
                fresh.append(entry[0])
        return TeamSnapshot(phase, tuple(fresh), tuple(stale), tuple(sorted(masked)))

    def status(self, *, now_monotonic_ns: int | None = None) -> str:
        snapshot = self.snapshot(now_monotonic_ns=now_monotonic_ns)
        fresh_by_id = {state.robot_id: state for state in snapshot.fresh}
        freshness = f"{len(snapshot.fresh)} of {len(self.expected_robot_ids)} robot links are fresh"
        if snapshot.stale_robot_ids:
            stale = ", ".join(_spoken_robot_id(robot_id) for robot_id in snapshot.stale_robot_ids)
            freshness += f"; stale or missing: {stale}"
        load_parts = [
            f"{_spoken_robot_id(robot_id)} {100.0 * fresh_by_id[robot_id].load_ratio:.0f} percent"
            for robot_id in self.expected_robot_ids
            if robot_id in fresh_by_id
        ]
        loads = "No current load readings." if not load_parts else "Load shares: " + ", ".join(load_parts) + "."
        return f"Mission is {snapshot.phase.value}. {freshness}. {loads}"

    def assess_load(self, *, now_monotonic_ns: int | None = None) -> str:
        snapshot = self.snapshot(now_monotonic_ns=now_monotonic_ns)
        if snapshot.stale_robot_ids:
            stale = ", ".join(_spoken_robot_id(robot_id) for robot_id in snapshot.stale_robot_ids)
            return f"Load assessment unavailable because telemetry is stale or missing for {stale}."
        states = snapshot.fresh
        if not states:
            return "Load assessment unavailable because no telemetry is fresh."
        highest = max(states, key=lambda state: state.load_ratio)
        lowest = min(states, key=lambda state: state.load_ratio)
        total = sum(state.load_ratio for state in states)
        spread = highest.load_ratio - lowest.load_ratio
        if highest.load_ratio > self.max_load_share:
            judgment = (
                f"Unsafe imbalance: {_spoken_robot_id(highest.robot_id)} exceeds the configured "
                f"{100.0 * self.max_load_share:.0f} percent share limit"
            )
        elif not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=0.15):
            judgment = f"Load readings are incomplete; reported shares sum to {100.0 * total:.0f} percent"
        else:
            judgment = "Load distribution is within the configured simulation gate"
        return (
            f"{judgment}. Highest is {_spoken_robot_id(highest.robot_id)} at "
            f"{100.0 * highest.load_ratio:.0f} percent; team spread is {100.0 * spread:.0f} percentage points."
        )

    def robot_load(self, robot_id: str, *, now_monotonic_ns: int | None = None) -> str:
        canonical = robot_id.strip().lower().replace("-", "_").replace(" ", "_")
        if canonical.startswith("g1") and not canonical.startswith("g1_"):
            canonical = canonical.replace("g1", "g1_", 1)
        snapshot = self.snapshot(now_monotonic_ns=now_monotonic_ns)
        for state in snapshot.fresh:
            if state.robot_id == canonical:
                speed = math.sqrt(sum(value * value for value in state.linear_velocity_w))
                return (
                    f"{_spoken_robot_id(canonical)} carries {100.0 * state.load_ratio:.0f} percent of the "
                    f"reported team load and is moving at {speed:.2f} meters per second."
                )
        if canonical not in self.expected_robot_ids:
            return f"{robot_id} is not a member of this team."
        return f"{_spoken_robot_id(canonical)} has no fresh telemetry."

    def request_command(
        self,
        intent: SupervisoryIntent | str,
        *,
        confirmed: bool = False,
        now_monotonic_ns: int | None = None,
    ) -> CommandDecision:
        parsed = SupervisoryIntent(str(intent).lower())
        now = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
        snapshot = self.snapshot(now_monotonic_ns=now)

        if parsed == SupervisoryIntent.ABORT:
            return self._accept(
                parsed, MissionPhase.ABORTED, "Abort accepted. The team must enter its local safe state."
            )

        if parsed == SupervisoryIntent.HOLD:
            if snapshot.phase != MissionPhase.ACTIVE:
                return self._reject(parsed, f"Hold is not valid from {snapshot.phase.value}.")
            return self._accept_if_phase(
                parsed,
                (MissionPhase.ACTIVE,),
                MissionPhase.HOLD,
                "Hold accepted. The team must maintain its local safe state.",
            )

        required_phase = MissionPhase.READY if parsed == SupervisoryIntent.START else MissionPhase.HOLD
        if snapshot.phase != required_phase:
            return self._reject(
                parsed,
                f"{parsed.value.capitalize()} is only valid from {required_phase.value}, not {snapshot.phase.value}.",
            )

        gate_reason = self._activation_gate_reason(snapshot)
        if gate_reason is not None:
            return self._reject(parsed, gate_reason)

        if not confirmed:
            message = f"Confirmation required before {parsed.value}. Ask the operator to say confirm {parsed.value}."
            with self._lock:
                if self._phase != required_phase:
                    phase = self._phase
                else:
                    self._pending_confirmation = parsed
                    self._pending_confirmation_deadline_ns = now + int(self.confirmation_window_s * 1_000_000_000)
                    self._last_transition = message
                    return CommandDecision(parsed, False, True, self._phase, message)
            return self._reject(parsed, f"Mission changed to {phase.value}; request {parsed.value} again.")

        with self._lock:
            pending = self._pending_confirmation
            deadline = self._pending_confirmation_deadline_ns
        if pending != parsed:
            return self._reject(
                parsed,
                f"No pending {parsed.value} request. Ask for the command first, then explicit confirmation.",
                confirmation_required=True,
            )
        if deadline is None or now > deadline:
            return self._reject(
                parsed,
                f"The {parsed.value} confirmation window expired. Ask for the command again.",
            )

        message = (
            "Start accepted. The local policy may begin the lift."
            if parsed == SupervisoryIntent.START
            else "Resume accepted. The local policy may continue."
        )
        return self._accept_if_phase(parsed, (required_phase,), MissionPhase.ACTIVE, message)

    def _activation_gate_reason(self, snapshot: TeamSnapshot) -> str | None:
        if snapshot.stale_robot_ids:
            stale = ", ".join(_spoken_robot_id(robot_id) for robot_id in snapshot.stale_robot_ids)
            return f"Command rejected because telemetry is stale or missing for {stale}."
        if not snapshot.fresh:
            return "Command rejected because no telemetry is fresh."
        highest = max(snapshot.fresh, key=lambda state: state.load_ratio)
        if highest.load_ratio > self.max_load_share:
            return (
                f"Command rejected because {_spoken_robot_id(highest.robot_id)} carries "
                f"{100.0 * highest.load_ratio:.0f} percent, above the configured "
                f"{100.0 * self.max_load_share:.0f} percent limit."
            )
        total = sum(state.load_ratio for state in snapshot.fresh)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=0.15):
            return f"Command rejected because reported load shares sum to {100.0 * total:.0f} percent."
        return None

    def inject_link_loss(self, robot_id: str) -> str:
        """Mask a robot only in an explicitly labelled demo session."""
        if not self.demo_mode:
            return "Fault injection is disabled outside demo mode."
        canonical = robot_id.strip().lower().replace("-", "_").replace(" ", "_")
        if canonical.startswith("g1") and not canonical.startswith("g1_"):
            canonical = canonical.replace("g1", "g1_", 1)
        if canonical not in self.expected_robot_ids:
            return f"{robot_id} is not a member of this team."
        with self._lock:
            self._masked_robot_ids.add(canonical)
            previous = self._phase
            if previous == MissionPhase.ACTIVE:
                self._phase = MissionPhase.ABORTED
                self._pending_confirmation = None
                self._pending_confirmation_deadline_ns = None
                self._last_transition = (
                    f"Demo link loss for {_spoken_robot_id(canonical)} forced an automatic abort from active."
                )
                sink = self._command_sink
            else:
                self._last_transition = f"Demo link loss injected for {_spoken_robot_id(canonical)}."
                sink = None
        if sink is not None:
            sink(SupervisoryIntent.ABORT)
        suffix = " The active mission was aborted." if previous == MissionPhase.ACTIVE else ""
        return f"Demo-only link loss injected for {_spoken_robot_id(canonical)}.{suffix}"

    def restore_link(self, robot_id: str) -> str:
        if not self.demo_mode:
            return "Fault injection is disabled outside demo mode."
        canonical = robot_id.strip().lower().replace("-", "_").replace(" ", "_")
        if canonical.startswith("g1") and not canonical.startswith("g1_"):
            canonical = canonical.replace("g1", "g1_", 1)
        with self._lock:
            existed = canonical in self._masked_robot_ids
            self._masked_robot_ids.discard(canonical)
            if existed:
                self._last_transition = f"Demo link for {_spoken_robot_id(canonical)} restored."
        if not existed:
            return f"No injected link loss was active for {_spoken_robot_id(canonical)}."
        return f"Demo link restored for {_spoken_robot_id(canonical)}. The mission remains {self.phase.value}."

    def explain_last_transition(self) -> str:
        with self._lock:
            return self._last_transition

    def enforce_freshness(self, *, now_monotonic_ns: int | None = None) -> CommandDecision | None:
        """Abort an active mission once any expected coordination link is stale."""
        snapshot = self.snapshot(now_monotonic_ns=now_monotonic_ns)
        if snapshot.phase != MissionPhase.ACTIVE or not snapshot.stale_robot_ids:
            return None
        stale = ", ".join(_spoken_robot_id(robot_id) for robot_id in snapshot.stale_robot_ids)
        return self._accept(
            SupervisoryIntent.ABORT,
            MissionPhase.ABORTED,
            f"Automatic abort: telemetry exceeded the {self.max_age_ms:.0f} millisecond deadline for {stale}.",
            expected_phases=(MissionPhase.ACTIVE,),
        )

    def _reject(
        self,
        intent: SupervisoryIntent,
        message: str,
        *,
        confirmation_required: bool = False,
    ) -> CommandDecision:
        with self._lock:
            self._pending_confirmation = None
            self._pending_confirmation_deadline_ns = None
            phase = self._phase
            self._last_transition = message
        return CommandDecision(intent, False, confirmation_required, phase, message)

    def _accept_if_phase(
        self,
        intent: SupervisoryIntent,
        expected_phases: tuple[MissionPhase, ...],
        phase: MissionPhase,
        message: str,
    ) -> CommandDecision:
        return self._accept(intent, phase, message, expected_phases=expected_phases)

    def _accept(
        self,
        intent: SupervisoryIntent,
        phase: MissionPhase,
        message: str,
        *,
        expected_phases: tuple[MissionPhase, ...] | None = None,
    ) -> CommandDecision:
        with self._lock:
            if expected_phases is not None and self._phase not in expected_phases:
                changed = f"Command rejected because mission changed to {self._phase.value}."
                self._pending_confirmation = None
                self._pending_confirmation_deadline_ns = None
                self._last_transition = changed
                return CommandDecision(intent, False, False, self._phase, changed)
            self._phase = phase
            self._pending_confirmation = None
            self._pending_confirmation_deadline_ns = None
            self._last_transition = message
            sink = self._command_sink
        if sink is not None:
            sink(intent)
        return CommandDecision(intent, True, False, phase, message)
