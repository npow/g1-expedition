"""Pure tensor helpers for the frozen AGILE/high-level controller boundary."""

from __future__ import annotations

import torch

HIGH_LEVEL_ACTION_DIM = 10
AGILE_COMMAND_DIM = 4
AGILE_OBSERVATION_DIM = 83
AGILE_LEG_ACTION_DIM = 12


def scale_high_level_actions(
    actions: torch.Tensor,
    velocity_scale: tuple[float, float, float],
    hip_height_bounds: tuple[float, float],
    wrist_offset_scale: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map normalized policy actions to AGILE commands and two wrist offsets.

    Args:
        actions: Tensor with final dimension 10, ordered as ``[vx, vy, wz,
            hip_height, left_xyz, right_xyz]`` and normally bounded by [-1, 1].

    Returns:
        AGILE commands with shape ``(..., 4)`` and wrist offsets with shape
        ``(..., 2, 3)``.
    """
    if actions.shape[-1] != HIGH_LEVEL_ACTION_DIM:
        raise ValueError(f"Expected {HIGH_LEVEL_ACTION_DIM} high-level actions, got {actions.shape[-1]}")

    bounded = torch.clamp(actions, -1.0, 1.0)
    velocity_scale_tensor = bounded.new_tensor(velocity_scale)
    hip_low, hip_high = hip_height_bounds

    commands = torch.empty((*bounded.shape[:-1], AGILE_COMMAND_DIM), device=bounded.device, dtype=bounded.dtype)
    commands[..., :3] = bounded[..., :3] * velocity_scale_tensor
    commands[..., 3] = hip_low + 0.5 * (bounded[..., 3] + 1.0) * (hip_high - hip_low)

    wrist_scale_tensor = bounded.new_tensor(wrist_offset_scale)
    wrist_offsets = bounded[..., 4:].reshape(*bounded.shape[:-1], 2, 3) * wrist_scale_tensor
    return commands, wrist_offsets


def compose_agile_observation(
    commands: torch.Tensor,
    base_linear_velocity: torch.Tensor,
    base_angular_velocity: torch.Tensor,
    projected_gravity: torch.Tensor,
    joint_position_relative: torch.Tensor,
    joint_velocity: torch.Tensor,
    previous_leg_action: torch.Tensor,
) -> torch.Tensor:
    """Compose the exact 83-value observation expected by NVIDIA AGILE.

    This mirrors Isaac Lab's ``AgileBasedLowerBodyAction`` and
    ``AgileTeacherPolicyObservationsCfg``. The joint tensors must contain the
    29 body joints in Isaac Lab asset order and the previous action must be the
    12 unscaled AGILE outputs.
    """
    observation = torch.cat(
        (
            commands,
            base_linear_velocity,
            base_angular_velocity,
            projected_gravity,
            joint_position_relative,
            0.1 * joint_velocity,
            previous_leg_action,
        ),
        dim=-1,
    )
    if observation.shape[-1] != AGILE_OBSERVATION_DIM:
        raise ValueError(
            f"AGILE observation must have {AGILE_OBSERVATION_DIM} values; got {observation.shape[-1]}"
        )
    return observation
