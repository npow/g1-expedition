from __future__ import annotations

import pytest
import torch

from cooperative_beam_isaaclab.tasks.trajectory import cuboid_inertia_tensor, payload_tracking_terms, rescue_trajectory


def test_cuboid_inertia_scales_with_payload_mass() -> None:
    masses = torch.tensor([[12.0], [24.0]])
    inertias = cuboid_inertia_tensor(masses, (1.0, 1.0, 1.0))
    assert inertias.shape == (2, 1, 9)
    assert torch.allclose(inertias[0, 0, [0, 4, 8]], torch.tensor([2.0, 2.0, 2.0]))
    assert torch.allclose(inertias[1], 2.0 * inertias[0])
    assert torch.count_nonzero(inertias[..., [1, 2, 3, 5, 6, 7]]) == 0


def test_rescue_trajectory_reaches_all_waypoints() -> None:
    start = torch.tensor([[0.0, 0.0, 0.075]]).repeat(4, 1)
    progress = torch.tensor([0.0, 0.30, 0.80, 1.0])
    position, yaw = rescue_trajectory(progress, start, 0.16, (0.85, 0.20), 0.14, 0.35)
    assert torch.allclose(position[0], torch.tensor([0.0, 0.0, 0.075]))
    assert torch.allclose(position[1], torch.tensor([0.0, 0.0, 0.235]), atol=1.0e-5)
    assert torch.allclose(position[2], torch.tensor([0.85, 0.20, 0.235]), atol=1.0e-5)
    assert torch.allclose(position[3], torch.tensor([0.85, 0.20, 0.14]), atol=1.0e-5)
    assert torch.allclose(yaw, torch.tensor([0.0, 0.0, 0.35, 0.35]), atol=1.0e-5)


def test_tracking_rewards_prefer_balanced_correct_pose() -> None:
    target = torch.zeros((2, 3))
    beam = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    up = torch.tensor([1.0, 0.5])
    heading = torch.tensor([1.0, 0.5])
    loads = torch.tensor([[50.0, 50.0, 50.0], [10.0, 50.0, 90.0]])
    terms = payload_tracking_terms(beam, target, up, heading, loads)
    assert terms["position"][0] > terms["position"][1]
    assert terms["level"][0] > terms["level"][1]
    assert terms["heading"][0] > terms["heading"][1]
    assert terms["load_balance"][0] > terms["load_balance"][1]


def test_load_balance_requires_support_and_respects_side_target() -> None:
    zeros = torch.zeros(1, 3)
    common = {
        "beam_position": torch.zeros(1, 3),
        "target_position": torch.zeros(1, 3),
        "beam_up_z": torch.ones(1),
        "heading_alignment": torch.ones(1),
    }
    unloaded = payload_tracking_terms(**common, team_loads=zeros)
    side_balanced = payload_tracking_terms(
        **common,
        team_loads=torch.tensor([[25.0, 50.0, 25.0]]),
        expected_load_ratios=torch.tensor([0.25, 0.50, 0.25]),
    )
    per_robot_equal = payload_tracking_terms(
        **common,
        team_loads=torch.tensor([[25.0, 25.0, 25.0]]),
        expected_load_ratios=torch.tensor([0.25, 0.50, 0.25]),
    )
    assert unloaded["load_balance"].item() == 0.0
    assert side_balanced["load_balance"].item() == pytest.approx(1.0)
    assert side_balanced["load_balance"] > per_robot_equal["load_balance"]
