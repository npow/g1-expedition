"""TeamHOI-style parameter sharing for skrl's multi-agent runner."""

from __future__ import annotations

import os
from typing import Any

import torch


def share_mappo_modules(agent: Any) -> None:
    """Tie every MAPPO station to one actor, critic, optimizer, and scheduler.

    skrl creates one model set per PettingZoo agent by default. The cooperative
    G1 stations have identical spaces, so sharing the modules gives us a single
    decentralized policy trained on experience from every station, as in
    TeamHOI. Memories and observation preprocessors remain per-station.
    """
    possible_agents = list(agent.possible_agents)
    if len(possible_agents) < 2:
        return

    primary = possible_agents[0]
    shared_policy = agent.policies[primary]
    shared_value = agent.values[primary]
    shared_optimizer = agent.optimizers[primary]
    shared_scheduler = agent.schedulers[primary]

    for uid in possible_agents:
        agent.policies[uid] = shared_policy
        agent.values[uid] = shared_value
        agent.models[uid]["policy"] = shared_policy
        agent.models[uid]["value"] = shared_value
        agent.optimizers[uid] = shared_optimizer
        agent.schedulers[uid] = shared_scheduler
        agent.checkpoint_modules[uid]["policy"] = shared_policy
        agent.checkpoint_modules[uid]["value"] = shared_value
        agent.checkpoint_modules[uid]["optimizer"] = shared_optimizer


def load_actor_only(agent: Any, checkpoint_path: str) -> None:
    """Load only the shared actor, allowing transfer across team sizes.

    skrl's full MAPPO checkpoint also contains a team-size-dependent critic and
    state scaler. The attention actor itself has identical parameters for any
    number of teammate tokens, so loading just its state is the correct
    zero-shot/generalization operation.
    """
    checkpoint = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)
    if "policy" in checkpoint:
        policy_state = checkpoint["policy"]
    else:
        source_agent = next(iter(checkpoint))
        policy_state = checkpoint[source_agent]["policy"]
    primary = agent.possible_agents[0]
    agent.policies[primary].load_state_dict(policy_state, strict=True)


def install_parameter_shared_runner(*, share_parameters: bool = True) -> None:
    """Install the custom actor factory and optionally tie MAPPO modules."""
    import skrl.utils.runner.torch as runner_module

    base_runner = runner_module.Runner
    if getattr(base_runner, "_cooperative_parameter_sharing", False):
        return

    class ParameterSharedRunner(base_runner):
        _cooperative_parameter_sharing = True

        def _component(self, name: str):
            if name.lower() == "teammateattentionpolicy":
                from .teamhoi_model import teammate_attention_model

                return teammate_attention_model
            return super()._component(name)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if share_parameters and hasattr(self.agent, "policies") and hasattr(self.agent, "values"):
                share_mappo_modules(self.agent)
                print(
                    "[INFO] TeamHOI-style parameter sharing enabled: "
                    f"one actor/central critic across {len(self.agent.possible_agents)} G1 stations"
                )
                actor_checkpoint = os.environ.get("COOP_ACTOR_CHECKPOINT")
                if actor_checkpoint:
                    load_actor_only(self.agent, actor_checkpoint)
                    print(f"[INFO] Loaded team-size-independent actor from: {actor_checkpoint}")

    runner_module.Runner = ParameterSharedRunner
