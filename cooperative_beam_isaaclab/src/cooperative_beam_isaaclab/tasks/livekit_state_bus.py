"""LiveKit fan-out transport for decentralized cooperative-policy state.

Training keeps teammate state in-process for vectorized throughput. At
deployment each robot publishes one compact state track to a LiveKit room and
subscribes to the other tracks. The actor observation contract is unchanged:
98 local values followed by six kinematic values per teammate and then one
load-ratio value per teammate.
"""

from __future__ import annotations

import asyncio
import json
import math
import struct
import threading
import time
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

LOCAL_OBSERVATION_DIM = 98
TEAMMATE_KINEMATIC_DIM = 6
TEAMMATE_TOKEN_DIM = 7
TRACK_NAME = "g1-coordination-state-v1"
SUPERVISORY_RPC_METHOD = "g1.supervisory-command.v1"
SUPERVISORY_COMMANDS = frozenset({"start", "hold", "resume", "abort"})

_MAGIC = b"G1ST"
_VERSION = 1
_PACKET = struct.Struct("!4sBHIQ11f")


def robot_index(robot_id: str) -> int:
    """Parse the canonical ``g1_N`` identity used by policy checkpoints."""
    prefix, separator, suffix = robot_id.rpartition("_")
    if separator != "_" or prefix != "g1" or not suffix.isdigit():
        raise ValueError(f"Robot identity must use the form g1_N, got {robot_id!r}")
    index = int(suffix)
    if not 0 <= index <= 0xFFFF:
        raise ValueError(f"Robot index is outside the packet range: {index}")
    return index


@dataclass(frozen=True)
class RobotCoordinationState:
    """State one G1 broadcasts once per inference tick."""

    robot_id: str
    sequence: int
    sender_time_ns: int
    position_w: tuple[float, float, float]
    linear_velocity_w: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    load_ratio: float

    def encode(self) -> bytes:
        values = (*self.position_w, *self.linear_velocity_w, *self.orientation_wxyz, self.load_ratio)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Coordination state contains a non-finite value")
        quaternion_norm = math.sqrt(sum(value * value for value in self.orientation_wxyz))
        if quaternion_norm < 1.0e-6:
            raise ValueError("Coordination orientation has zero norm")
        return _PACKET.pack(
            _MAGIC,
            _VERSION,
            robot_index(self.robot_id),
            self.sequence & 0xFFFFFFFF,
            self.sender_time_ns,
            *values,
        )

    @classmethod
    def decode(cls, payload: bytes) -> RobotCoordinationState:
        if len(payload) != _PACKET.size:
            raise ValueError(f"Expected {_PACKET.size} state bytes, got {len(payload)}")
        magic, version, index, sequence, sender_time_ns, *values = _PACKET.unpack(payload)
        if magic != _MAGIC:
            raise ValueError("Coordination packet has the wrong magic")
        if version != _VERSION:
            raise ValueError(f"Unsupported coordination packet version: {version}")
        return cls(
            robot_id=f"g1_{index}",
            sequence=sequence,
            sender_time_ns=sender_time_ns,
            position_w=tuple(values[0:3]),
            linear_velocity_w=tuple(values[3:6]),
            orientation_wxyz=tuple(values[6:10]),
            load_ratio=values[10],
        )


def _sequence_is_newer(candidate: int, previous: int) -> bool:
    delta = (candidate - previous) & 0xFFFFFFFF
    return 0 < delta < 0x80000000


class LatestStateTable:
    """Thread-safe last-value cache with local receive-time freshness gates."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, tuple[RobotCoordinationState, int]] = {}

    def update(self, state: RobotCoordinationState, received_monotonic_ns: int | None = None) -> bool:
        received = time.monotonic_ns() if received_monotonic_ns is None else received_monotonic_ns
        with self._lock:
            previous = self._states.get(state.robot_id)
            if previous is not None and not _sequence_is_newer(state.sequence, previous[0].sequence):
                return False
            self._states[state.robot_id] = (state, received)
        return True

    def fresh(
        self,
        robot_id: str,
        max_age_ms: float,
        now_monotonic_ns: int | None = None,
    ) -> RobotCoordinationState | None:
        now = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        with self._lock:
            entry = self._states.get(robot_id)
        if entry is None:
            return None
        state, received = entry
        if now - received > int(max_age_ms * 1_000_000):
            return None
        return state


def _world_to_body(vector_w: Sequence[float], orientation_wxyz: Sequence[float]) -> tuple[float, float, float]:
    w, x, y, z = orientation_wxyz
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1.0e-6:
        raise ValueError("Recipient orientation has zero norm")
    w, x, y, z = (component / norm for component in (w, x, y, z))
    # R maps body vectors to world vectors. Multiplying by R^T gives the
    # recipient-frame representation expected by the frozen MAPPO actor.
    rotation = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )
    return tuple(sum(rotation[row][column] * vector_w[row] for row in range(3)) for column in range(3))


def teammate_observation_parts(
    recipient: RobotCoordinationState,
    teammate_ids: Iterable[str],
    table: LatestStateTable,
    max_age_ms: float = 150.0,
    now_monotonic_ns: int | None = None,
) -> tuple[list[float], list[float], list[bool]]:
    """Build the checkpoint-compatible teammate fields from LiveKit state.

    Missing or stale states are zero-filled. The returned freshness vector is
    kept outside the frozen actor contract for observability and failover.
    """
    kinematics: list[float] = []
    loads: list[float] = []
    freshness: list[bool] = []
    for teammate_id in teammate_ids:
        teammate = table.fresh(teammate_id, max_age_ms, now_monotonic_ns)
        if teammate is None:
            kinematics.extend((0.0,) * TEAMMATE_KINEMATIC_DIM)
            loads.append(0.0)
            freshness.append(False)
            continue
        relative_position_w = tuple(a - b for a, b in zip(teammate.position_w, recipient.position_w, strict=True))
        relative_velocity_w = tuple(
            a - b for a, b in zip(teammate.linear_velocity_w, recipient.linear_velocity_w, strict=True)
        )
        kinematics.extend(_world_to_body(relative_position_w, recipient.orientation_wxyz))
        kinematics.extend(_world_to_body(relative_velocity_w, recipient.orientation_wxyz))
        loads.append(float(teammate.load_ratio))
        freshness.append(True)
    return kinematics, loads, freshness


def assemble_policy_observation(
    local_observation: Sequence[float],
    recipient: RobotCoordinationState,
    teammate_ids: Sequence[str],
    table: LatestStateTable,
    max_age_ms: float = 150.0,
    now_monotonic_ns: int | None = None,
) -> tuple[list[float], list[bool]]:
    """Append LiveKit-delivered teammate tokens to the 98-value local state."""
    if len(local_observation) != LOCAL_OBSERVATION_DIM:
        raise ValueError(
            f"Expected {LOCAL_OBSERVATION_DIM} local observation values, got {len(local_observation)}"
        )
    kinematics, loads, freshness = teammate_observation_parts(
        recipient,
        teammate_ids,
        table,
        max_age_ms=max_age_ms,
        now_monotonic_ns=now_monotonic_ns,
    )
    return [*local_observation, *kinematics, *loads], freshness


class LiveKitRobotStateClient:
    """One robot's LiveKit data-track publisher and teammate subscriber."""

    def __init__(
        self,
        robot_id: str,
        url: str,
        token: str,
        table: LatestStateTable | None = None,
        command_callers: Iterable[str] = (),
    ) -> None:
        robot_index(robot_id)
        self.robot_id = robot_id
        self.url = url
        self.token = token
        self.table = table or LatestStateTable()
        self.command_callers = frozenset(command_callers)
        self._sequence = 0
        self._room = None
        self._track = None
        self._subscriber_tasks: set[asyncio.Task] = set()
        self._command_lock = threading.Lock()
        self._commands: deque[dict[str, object]] = deque(maxlen=32)
        self._command_ids: deque[str] = deque(maxlen=128)

    async def connect(self) -> None:
        try:
            from livekit import rtc
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("Install the LiveKit transport with: pip install -e '.[livekit]'") from exc

        self._room = rtc.Room()

        @self._room.on("data_track_published")
        def on_data_track_published(track) -> None:
            if track.info.name != TRACK_NAME or track.publisher_identity == self.robot_id:
                return
            task = asyncio.create_task(self._subscribe(track))
            self._subscriber_tasks.add(task)
            task.add_done_callback(self._subscriber_tasks.discard)

        await self._room.connect(self.url, self.token)
        if self.command_callers:
            self._room.local_participant.register_rpc_method(SUPERVISORY_RPC_METHOD, self._receive_command)
        self._track = await self._room.local_participant.publish_data_track(name=TRACK_NAME)

    def _receive_command(self, invocation) -> str:
        """Validate and enqueue one high-level command; never actuate here."""
        if invocation.caller_identity not in self.command_callers:
            return json.dumps({"accepted": False, "reason": "caller is not authorized"})
        try:
            command = json.loads(invocation.payload)
        except (TypeError, json.JSONDecodeError):
            return json.dumps({"accepted": False, "reason": "payload is not valid JSON"})
        if not isinstance(command, dict):
            return json.dumps({"accepted": False, "reason": "payload must be an object"})
        command_id = command.get("command_id")
        intent = command.get("intent")
        if not isinstance(command_id, str) or not command_id or intent not in SUPERVISORY_COMMANDS:
            return json.dumps({"accepted": False, "reason": "invalid command id or intent"})
        with self._command_lock:
            if command_id in self._command_ids:
                return json.dumps({"accepted": True, "duplicate": True, "robot_id": self.robot_id})
            self._command_ids.append(command_id)
            self._commands.append(command)
        return json.dumps({"accepted": True, "queued": True, "robot_id": self.robot_id})

    def drain_commands(self) -> list[dict[str, object]]:
        """Return commands for the local deterministic supervisor exactly once."""
        with self._command_lock:
            commands = list(self._commands)
            self._commands.clear()
        return commands

    async def _subscribe(self, track) -> None:
        try:
            async for frame in track.subscribe():
                try:
                    state = RobotCoordinationState.decode(bytes(frame.payload))
                except ValueError:
                    continue
                if state.robot_id == track.publisher_identity:
                    self.table.update(state)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - SDK transport failures must not stop local control
            # A remote participant can leave at any time. Freshness gating
            # removes its last packet without interrupting local inference.
            return

    def publish(
        self,
        position_w: Sequence[float],
        linear_velocity_w: Sequence[float],
        orientation_wxyz: Sequence[float],
        load_ratio: float,
    ) -> RobotCoordinationState:
        if self._track is None:
            raise RuntimeError("LiveKit client is not connected")
        from livekit import rtc

        state = RobotCoordinationState(
            robot_id=self.robot_id,
            sequence=self._sequence,
            sender_time_ns=time.time_ns(),
            position_w=tuple(float(value) for value in position_w),
            linear_velocity_w=tuple(float(value) for value in linear_velocity_w),
            orientation_wxyz=tuple(float(value) for value in orientation_wxyz),
            load_ratio=float(load_ratio),
        )
        frame = rtc.DataTrackFrame(payload=state.encode(), user_timestamp=state.sender_time_ns // 1_000_000)
        self._track.try_push(frame)
        self.table.update(state)
        self._sequence = (self._sequence + 1) & 0xFFFFFFFF
        return state

    async def close(self) -> None:
        for task in tuple(self._subscriber_tasks):
            task.cancel()
        if self._subscriber_tasks:
            await asyncio.gather(*self._subscriber_tasks, return_exceptions=True)
        if self._room is not None:
            if self.command_callers:
                self._room.local_participant.unregister_rpc_method(SUPERVISORY_RPC_METHOD)
            await self._room.disconnect()
        self._room = None
        self._track = None
