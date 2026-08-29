"""Gym registration for the cooperative G1 disaster-recovery tasks."""

import gymnasium as gym

from . import agents

CRATE_TASK_ID = "Isaac-Cooperative-G1-Rescue-Crate-Direct-v0"
TIMBER_TASK_ID = "Isaac-Cooperative-G1-Timber-Direct-v0"
GIRDER_TASK_ID = "Isaac-Cooperative-G1-Footbridge-Girder-Direct-v0"
TASK_ID = TIMBER_TASK_ID

TASKS = {
    CRATE_TASK_ID: "CooperativeCrateEnvCfg",
    TIMBER_TASK_ID: "CooperativeBeamEnvCfg",
    GIRDER_TASK_ID: "CooperativeGirderEnvCfg",
}

for task_id, config_class in TASKS.items():
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.cooperative_beam_env:CooperativeBeamEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.cooperative_beam_env_cfg:{config_class}",
            "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
            "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        },
    )

__all__ = ["CRATE_TASK_ID", "GIRDER_TASK_ID", "TASKS", "TASK_ID", "TIMBER_TASK_ID"]
