"""Simulator-independent target trajectory and reward helpers."""

from __future__ import annotations

import torch


def cuboid_inertia_tensor(
    masses: torch.Tensor,
    size: tuple[float, float, float],
) -> torch.Tensor:
    """Return row-major body-frame inertia tensors for uniform cuboids."""
    size_x, size_y, size_z = size
    inertias = torch.zeros((*masses.shape, 9), dtype=masses.dtype, device=masses.device)
    inertias[..., 0] = masses * (size_y**2 + size_z**2) / 12.0
    inertias[..., 4] = masses * (size_x**2 + size_z**2) / 12.0
    inertias[..., 8] = masses * (size_x**2 + size_y**2) / 12.0
    return inertias


def smoothstep(value: torch.Tensor) -> torch.Tensor:
    """Cubic smoothstep on a value already intended for the unit interval."""
    value = torch.clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def rescue_trajectory(
    progress: torch.Tensor,
    start_position: torch.Tensor,
    lift_height: float,
    carry_delta_xy: tuple[float, float],
    final_height: float,
    target_yaw: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate lift, carry/turn, and placement targets for every environment.

    The three phases occupy 30%, 50%, and 20% of an episode. The return values
    are target position and target yaw, both expressed in the local environment.
    """
    lift_alpha = smoothstep(progress / 0.30)
    carry_alpha = smoothstep((progress - 0.30) / 0.50)
    place_alpha = smoothstep((progress - 0.80) / 0.20)

    target = start_position.clone()
    lifted_z = start_position[:, 2] + lift_height
    target[:, 2] = start_position[:, 2] + lift_alpha * lift_height
    target[:, 0] += carry_alpha * carry_delta_xy[0]
    target[:, 1] += carry_alpha * carry_delta_xy[1]
    target[:, 2] = torch.where(progress >= 0.80, lifted_z + place_alpha * (final_height - lifted_z), target[:, 2])
    yaw = carry_alpha * target_yaw
    return target, yaw


def payload_tracking_terms(
    beam_position: torch.Tensor,
    target_position: torch.Tensor,
    beam_up_z: torch.Tensor,
    heading_alignment: torch.Tensor,
    team_loads: torch.Tensor,
    expected_load_ratios: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute bounded team-level shaping terms used by the environment."""
    position_error = torch.linalg.vector_norm(beam_position - target_position, dim=-1)
    position = torch.exp(-3.0 * position_error)
    level = torch.clamp(beam_up_z, 0.0, 1.0) ** 2
    heading = torch.clamp(heading_alignment, 0.0, 1.0) ** 2
    load_mean = team_loads.mean(dim=-1)
    load_cv = team_loads.std(dim=-1, unbiased=False) / torch.clamp(load_mean, min=5.0)
    load_sum = team_loads.sum(dim=-1, keepdim=True)
    actual_load_ratios = team_loads / torch.clamp(load_sum, min=1.0)
    if expected_load_ratios is None:
        expected_load_ratios = torch.full_like(team_loads, 1.0 / team_loads.shape[-1])
    else:
        expected_load_ratios = expected_load_ratios.to(device=team_loads.device, dtype=team_loads.dtype)
        expected_load_ratios = expected_load_ratios.expand_as(team_loads)
    relative_load_error = (actual_load_ratios - expected_load_ratios) / torch.clamp(
        expected_load_ratios, min=1.0e-5
    )
    load_target_rmse = torch.sqrt(torch.mean(torch.square(relative_load_error), dim=-1))
    # Zero tension is not balanced support. This gate closes the reward loophole
    # where every unloaded sling previously produced a perfect coefficient of variation.
    supporting = torch.clamp(load_mean / 5.0, 0.0, 1.0)
    load_balance = torch.exp(-2.0 * load_target_rmse) * supporting
    return {
        "position": position,
        "level": level,
        "heading": heading,
        "load_balance": load_balance,
        "position_error": position_error,
        "load_cv": load_cv,
        "load_target_rmse": load_target_rmse,
    }
