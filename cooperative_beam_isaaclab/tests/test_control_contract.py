from __future__ import annotations

import torch

from cooperative_beam_isaaclab.tasks.control_contract import (
    AGILE_OBSERVATION_DIM,
    compose_agile_observation,
    scale_high_level_actions,
)


def test_high_level_action_scaling_has_physical_units() -> None:
    actions = torch.tensor(
        [
            [1.0, -1.0, 0.5, -1.0, 1.0, 0.0, -1.0, -1.0, 0.5, 1.0],
            [-2.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    commands, wrists = scale_high_level_actions(
        actions,
        velocity_scale=(0.55, 0.30, 0.65),
        hip_height_bounds=(0.64, 0.78),
        wrist_offset_scale=(0.10, 0.09, 0.38),
    )

    assert torch.allclose(commands[0], torch.tensor([0.55, -0.30, 0.325, 0.64]))
    assert torch.allclose(commands[1], torch.tensor([-0.55, 0.30, 0.0, 0.78]))
    assert torch.allclose(wrists[0, 0], torch.tensor([0.10, 0.0, -0.38]))
    assert torch.allclose(wrists[0, 1], torch.tensor([-0.10, 0.045, 0.38]))


def test_agile_observation_matches_frozen_policy_contract() -> None:
    batch = 5
    observation = compose_agile_observation(
        commands=torch.zeros(batch, 4),
        base_linear_velocity=torch.zeros(batch, 3),
        base_angular_velocity=torch.zeros(batch, 3),
        projected_gravity=torch.zeros(batch, 3),
        joint_position_relative=torch.zeros(batch, 29),
        joint_velocity=torch.full((batch, 29), 10.0),
        previous_leg_action=torch.zeros(batch, 12),
    )

    assert observation.shape == (batch, AGILE_OBSERVATION_DIM)
    # Joint velocities occupy the 42:71 slice and are scaled exactly as in Isaac Lab.
    assert torch.allclose(observation[:, 42:71], torch.ones(batch, 29))
