from __future__ import annotations

import math

import pytest

from cooperative_beam_isaaclab.tasks.formation import (
    alternating_sides,
    even_stations,
    robot_station_poses,
    side_balanced_load_ratios,
    sling_station_x,
)


def test_alternating_formation_faces_payload_from_both_sides() -> None:
    stations = even_stations(5, 3.0)
    poses = robot_station_poses(stations, 0.2)
    assert alternating_sides(5) == (-1, 1, -1, 1, -1)
    assert tuple(pose[1] for pose in poses) == pytest.approx(stations)
    assert tuple(pose[2] for pose in poses) == pytest.approx((0.0, math.pi, 0.0, math.pi, 0.0))
    assert all(pose[0] < 0.0 for pose in poses[::2])
    assert all(pose[0] > 0.0 for pose in poses[1::2])


def test_sling_edges_and_odd_team_load_targets_are_roll_balanced() -> None:
    station_x = sling_station_x(3, 0.2)
    expected = side_balanced_load_ratios(station_x)
    assert station_x == pytest.approx((-0.1, 0.1, -0.1))
    assert expected == pytest.approx((0.25, 0.5, 0.25))
    assert sum(ratio for ratio, x_value in zip(expected, station_x, strict=True) if x_value < 0.0) == pytest.approx(
        0.5
    )
    assert sum(ratio for ratio, x_value in zip(expected, station_x, strict=True) if x_value > 0.0) == pytest.approx(
        0.5
    )
