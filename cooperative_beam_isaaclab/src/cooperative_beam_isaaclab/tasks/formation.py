"""Simulator-independent formation geometry for cooperative payload stations."""

from __future__ import annotations

import math


def even_stations(team_size: int, payload_length: float) -> tuple[float, ...]:
    """Spread stations over 72% of the payload's long axis."""
    if team_size < 2:
        raise ValueError("Cooperative transport requires at least two robots")
    half_span = 0.36 * payload_length
    interval = 2.0 * half_span / (team_size - 1)
    return tuple(-half_span + index * interval for index in range(team_size))


def alternating_sides(team_size: int) -> tuple[int, ...]:
    """Assign consecutive long-axis stations to opposite payload sides."""
    if team_size < 2:
        raise ValueError("Cooperative transport requires at least two robots")
    return tuple(-1 if index % 2 == 0 else 1 for index in range(team_size))


def robot_station_poses(
    stations: tuple[float, ...],
    payload_width: float,
    clearance: float = 0.50,
) -> tuple[tuple[float, float, float], ...]:
    """Return robot x/y/yaw poses facing inward from alternating sides."""
    sides = alternating_sides(len(stations))
    offset = clearance + payload_width / 2.0
    return tuple(
        (side * offset, station_y, 0.0 if side < 0 else math.pi)
        for side, station_y in zip(sides, stations, strict=True)
    )


def sling_station_x(team_size: int, payload_width: float) -> tuple[float, ...]:
    """Place each robot's sling pair on its near payload edge."""
    return tuple(side * payload_width / 2.0 for side in alternating_sides(team_size))


def side_balanced_load_ratios(station_x: tuple[float, ...]) -> tuple[float, ...]:
    """Split vertical load equally by side, then equally within each side.

    Odd teams have different station counts on the two sides. Equal per-robot
    loads would therefore create a roll torque, so robots on the less-populated
    side must intentionally carry more.
    """
    negative = sum(value < 0.0 for value in station_x)
    positive = sum(value > 0.0 for value in station_x)
    if negative == 0 or positive == 0:
        return tuple(1.0 / len(station_x) for _ in station_x)
    return tuple(0.5 / (negative if value < 0.0 else positive) for value in station_x)
