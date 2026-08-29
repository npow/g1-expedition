"""A compact teammate-token actor adapted from TeamHOI's coordination design."""

from __future__ import annotations

import torch
from skrl.models.torch import GaussianMixin, Model
from torch import nn

LOCAL_OBSERVATION_DIM = 98
TEAMMATE_KINEMATIC_DIM = 6
TEAMMATE_TOKEN_DIM = TEAMMATE_KINEMATIC_DIM + 1


class TeammateAttentionPolicy(GaussianMixin, Model):
    """Shared actor that cross-attends from local state to teammate tokens."""

    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device=None,
        clip_actions=False,
        clip_mean_actions=False,
        clip_log_std=True,
        min_log_std=-5.0,
        max_log_std=1.0,
        initial_log_std=-0.9,
        token_dim=128,
        attention_heads=4,
        role="",
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_mean_actions=clip_mean_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction="sum",
            role=role,
        )

        observation_dim = int(observation_space.shape[0])
        teammate_width = observation_dim - LOCAL_OBSERVATION_DIM
        if teammate_width <= 0 or teammate_width % TEAMMATE_TOKEN_DIM:
            raise ValueError(
                f"Observation width {observation_dim} does not encode one or more "
                f"{TEAMMATE_TOKEN_DIM}-value teammate tokens after the "
                f"{LOCAL_OBSERVATION_DIM}-value local observation"
            )
        self.num_teammates = teammate_width // TEAMMATE_TOKEN_DIM

        self.local_encoder = nn.Sequential(
            nn.Linear(LOCAL_OBSERVATION_DIM, 256),
            nn.ELU(),
            nn.Linear(256, token_dim),
            nn.ELU(),
        )
        self.teammate_encoder = nn.Sequential(
            nn.Linear(TEAMMATE_KINEMATIC_DIM + 1, token_dim),
            nn.ELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.teammate_attention = nn.MultiheadAttention(token_dim, attention_heads, batch_first=True)
        self.policy_head = nn.Sequential(
            nn.Linear(2 * token_dim, 256),
            nn.ELU(),
            nn.Linear(256, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), float(initial_log_std)))

    def compute(self, inputs, role=""):
        observation = inputs["observations"]
        expected_width = LOCAL_OBSERVATION_DIM + self.num_teammates * TEAMMATE_TOKEN_DIM
        if observation.shape[-1] != expected_width:
            raise ValueError(f"Unexpected cooperative observation shape: {observation.shape}")

        local_token = self.local_encoder(observation[:, :LOCAL_OBSERVATION_DIM])
        teammate_kinematics = observation[
            :,
            LOCAL_OBSERVATION_DIM : LOCAL_OBSERVATION_DIM
            + self.num_teammates * TEAMMATE_KINEMATIC_DIM,
        ].reshape(-1, self.num_teammates, TEAMMATE_KINEMATIC_DIM)
        teammate_loads = observation[:, -self.num_teammates :].unsqueeze(-1)
        teammate_tokens = self.teammate_encoder(torch.cat((teammate_kinematics, teammate_loads), dim=-1))
        attended, _ = self.teammate_attention(
            local_token.unsqueeze(1),
            teammate_tokens,
            teammate_tokens,
            need_weights=False,
        )
        mean_actions = self.policy_head(torch.cat((local_token, attended.squeeze(1)), dim=-1))
        return mean_actions, {"log_std": self.log_std_parameter}


def teammate_attention_model(
    observation_space,
    state_space,
    action_space,
    device=None,
    return_source=False,
    **kwargs,
):
    """skrl runner-compatible model factory."""
    if return_source:
        return "TeammateAttentionPolicy(local query + variable exchangeable teammate tokens)"
    return TeammateAttentionPolicy(
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device=device,
        **kwargs,
    )
