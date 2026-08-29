from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import torch

from cooperative_beam_isaaclab.tasks.parameter_sharing import load_actor_only
from cooperative_beam_isaaclab.tasks.teamhoi_model import teammate_attention_model


def test_teammate_attention_actor_supports_variable_teams() -> None:
    reference_shapes = None
    for team_size in (2, 3, 4, 5, 6):
        observation_dim = 98 + 7 * (team_size - 1)
        model = teammate_attention_model(
            observation_space=gym.spaces.Box(-float("inf"), float("inf"), shape=(observation_dim,)),
            state_space=gym.spaces.Box(
                -float("inf"),
                float("inf"),
                shape=(team_size * observation_dim + 1,),
            ),
            action_space=gym.spaces.Box(-1.0, 1.0, shape=(10,)),
            device="cpu",
        )
        mean, metadata = model.compute({"observations": torch.zeros(7, observation_dim)})
        assert model.num_teammates == team_size - 1
        assert mean.shape == (7, 10)
        assert metadata["log_std"].shape == (10,)
        state_shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
        if reference_shapes is None:
            reference_shapes = state_shapes
        else:
            assert state_shapes == reference_shapes


def test_actor_checkpoint_transfers_between_team_sizes(tmp_path: Path) -> None:
    def model_for(team_size: int):
        observation_dim = 98 + 7 * (team_size - 1)
        return teammate_attention_model(
            observation_space=gym.spaces.Box(-float("inf"), float("inf"), shape=(observation_dim,)),
            state_space=gym.spaces.Box(-float("inf"), float("inf"), shape=(team_size * observation_dim + 1,)),
            action_space=gym.spaces.Box(-1.0, 1.0, shape=(10,)),
            device="cpu",
        )

    source = model_for(2)
    target = model_for(5)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(0.125)
    checkpoint_path = tmp_path / "agent.pt"
    torch.save({"g1_0": {"policy": source.state_dict()}}, checkpoint_path)

    agent = SimpleNamespace(device="cpu", possible_agents=["g1_0"], policies={"g1_0": target})
    load_actor_only(agent, str(checkpoint_path))

    for source_parameter, target_parameter in zip(source.parameters(), target.parameters(), strict=True):
        assert torch.equal(source_parameter, target_parameter)
