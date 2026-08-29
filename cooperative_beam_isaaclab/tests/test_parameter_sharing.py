from __future__ import annotations

from types import SimpleNamespace

from cooperative_beam_isaaclab.tasks.parameter_sharing import share_mappo_modules


def test_mappo_modules_are_tied_across_robot_stations() -> None:
    uids = ["g1_0", "g1_1", "g1_2"]
    policies = {uid: object() for uid in uids}
    values = {uid: object() for uid in uids}
    optimizers = {uid: object() for uid in uids}
    schedulers = {uid: object() for uid in uids}
    agent = SimpleNamespace(
        possible_agents=uids,
        policies=policies,
        values=values,
        optimizers=optimizers,
        schedulers=schedulers,
        models={uid: {"policy": policies[uid], "value": values[uid]} for uid in uids},
        checkpoint_modules={
            uid: {"policy": policies[uid], "value": values[uid], "optimizer": optimizers[uid]} for uid in uids
        },
    )

    share_mappo_modules(agent)

    assert len({id(agent.policies[uid]) for uid in uids}) == 1
    assert len({id(agent.values[uid]) for uid in uids}) == 1
    assert len({id(agent.optimizers[uid]) for uid in uids}) == 1
