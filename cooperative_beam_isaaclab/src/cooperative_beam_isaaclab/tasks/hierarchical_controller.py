"""Frozen AGILE locomotion with GPU-batched wrist inverse kinematics."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io.torchscript import load_torchscript_model
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from .control_contract import AGILE_LEG_ACTION_DIM, compose_agile_observation, scale_high_level_actions


class HierarchicalG1Controller:
    """Drive several G1s through one frozen lower-body policy and wrist IK.

    The AGILE network is evaluated once on a stacked ``num_envs * num_robots``
    batch. Only the 10 high-level commands are learned. This preserves AGILE's
    leg controller and leaves both seven-joint arms available for sling control.
    """

    def __init__(
        self,
        robots: list[Articulation],
        wrist_body_ids: list[list[int]],
        num_envs: int,
        device: str,
        cfg,
    ) -> None:
        self.robots = robots
        self.wrist_body_ids = wrist_body_ids
        self.num_envs = num_envs
        self.num_robots = len(robots)
        self.device = device
        self.cfg = cfg

        self.body_joint_ids, body_names = robots[0].find_joints(cfg.agile_body_joint_patterns)
        self.leg_joint_ids, leg_names = robots[0].find_joints(cfg.agile_leg_joint_patterns)
        if len(body_names) != 29:
            raise RuntimeError(f"AGILE requires 29 body joints, found {len(body_names)}: {body_names}")
        if len(leg_names) != cfg.agile_policy_output_dim:
            raise RuntimeError(
                f"AGILE requires {cfg.agile_policy_output_dim} leg joints, found {len(leg_names)}: {leg_names}"
            )

        self.arm_joint_ids: list[list[list[int]]] = []
        self.arm_joint_names: list[list[list[str]]] = []
        for robot in robots:
            robot_arm_ids: list[list[int]] = []
            robot_arm_names: list[list[str]] = []
            for side in ("left", "right"):
                patterns = [pattern.replace(".*", side) for pattern in cfg.arm_joint_patterns]
                ids, names = robot.find_joints(patterns)
                if len(ids) != 7:
                    raise RuntimeError(f"Expected seven {side} arm joints, found {len(ids)}: {names}")
                robot_arm_ids.append(ids)
                robot_arm_names.append(names)
            self.arm_joint_ids.append(robot_arm_ids)
            self.arm_joint_names.append(robot_arm_names)

        policy_file = retrieve_file_path(cfg.agile_policy_path)
        self.policy = load_torchscript_model(policy_file, device=device)
        self.policy.eval()

        ik_cfg = DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": cfg.wrist_ik_damping},
        )
        self.wrist_ik = [
            [DifferentialIKController(ik_cfg, num_envs=num_envs, device=device) for _ in range(2)]
            for _ in robots
        ]

        self.previous_leg_actions = torch.zeros(
            (num_envs, self.num_robots, AGILE_LEG_ACTION_DIM), device=device
        )
        self.commands = torch.zeros((num_envs, self.num_robots, 4), device=device)
        self.wrist_offsets = torch.zeros((num_envs, self.num_robots, 2, 3), device=device)
        self.nominal_wrist_positions_b = torch.zeros_like(self.wrist_offsets)
        self.needs_wrist_calibration = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.gravity_w = torch.tensor((0.0, 0.0, -1.0), device=device).repeat(num_envs, 1)

        print(f"[INFO] Loaded frozen AGILE policy: {policy_file}")
        print(f"[INFO] AGILE body joints ({len(body_names)}): {body_names}")
        print(f"[INFO] AGILE leg outputs ({len(leg_names)}): {leg_names}")

    def reset(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.previous_leg_actions[env_ids] = 0.0
        self.commands[env_ids] = 0.0
        self.wrist_offsets[env_ids] = 0.0
        self.needs_wrist_calibration[env_ids] = True
        for controllers in self.wrist_ik:
            for controller in controllers:
                controller.reset(env_ids)

    def apply(self, agent_actions: list[torch.Tensor]) -> None:
        """Apply a list of normalized high-level actions, one tensor per G1."""
        stacked_actions = torch.stack(agent_actions, dim=1)
        self.commands, self.wrist_offsets = scale_high_level_actions(
            stacked_actions,
            self.cfg.command_velocity_scale,
            self.cfg.command_hip_height,
            self.cfg.wrist_action_scale,
        )

        policy_inputs = []
        for robot_index, robot in enumerate(self.robots):
            root_quat = robot.data.root_quat_w.torch
            projected_gravity = quat_apply_inverse(root_quat, self.gravity_w)
            policy_inputs.append(
                compose_agile_observation(
                    self.commands[:, robot_index],
                    robot.data.root_lin_vel_b.torch,
                    robot.data.root_ang_vel_b.torch,
                    projected_gravity,
                    robot.data.joint_pos.torch[:, self.body_joint_ids]
                    - robot.data.default_joint_pos.torch[:, self.body_joint_ids],
                    robot.data.joint_vel.torch[:, self.body_joint_ids],
                    self.previous_leg_actions[:, robot_index],
                )
            )

        # Stack robot index next to environment index for a single GPU inference.
        policy_input = torch.stack(policy_inputs, dim=1).reshape(self.num_envs * self.num_robots, -1)
        with torch.inference_mode():
            leg_actions = self.policy.forward(policy_input)
        leg_actions = leg_actions.reshape(self.num_envs, self.num_robots, -1)
        if leg_actions.shape[-1] != AGILE_LEG_ACTION_DIM:
            raise RuntimeError(f"AGILE returned {leg_actions.shape[-1]} actions, expected {AGILE_LEG_ACTION_DIM}")
        self.previous_leg_actions.copy_(leg_actions)

        for robot_index, robot in enumerate(self.robots):
            # Default targets hold hands/waist. AGILE owns the legs and the
            # batched IK controllers own the two arms.
            targets = robot.data.default_joint_pos.torch.clone()
            targets[:, self.leg_joint_ids] += self.cfg.agile_policy_output_scale * leg_actions[:, robot_index]

            root_position = robot.data.root_pos_w.torch
            root_quat = robot.data.root_quat_w.torch
            wrist_positions_w = robot.data.body_pos_w.torch[:, self.wrist_body_ids[robot_index]]
            wrist_positions_b = quat_apply_inverse(
                root_quat[:, None, :].expand(-1, 2, -1).reshape(-1, 4),
                (wrist_positions_w - root_position[:, None, :]).reshape(-1, 3),
            ).reshape(self.num_envs, 2, 3)

            if torch.any(self.needs_wrist_calibration):
                env_ids = torch.nonzero(self.needs_wrist_calibration, as_tuple=False).squeeze(-1)
                self.nominal_wrist_positions_b[env_ids, robot_index] = wrist_positions_b[env_ids]

            wrist_targets_b = self.nominal_wrist_positions_b[:, robot_index] + self.wrist_offsets[:, robot_index]
            target_quat = root_quat[:, None, :].expand(-1, 2, -1).reshape(-1, 4)
            wrist_targets_w = root_position[:, None, :] + quat_apply(
                target_quat, wrist_targets_b.reshape(-1, 3)
            ).reshape(self.num_envs, 2, 3)

            for side_index in range(2):
                arm_ids = self.arm_joint_ids[robot_index][side_index]
                wrist_id = self.wrist_body_ids[robot_index][side_index]
                controller = self.wrist_ik[robot_index][side_index]
                wrist_pose_w = robot.data.body_pose_w.torch[:, wrist_id]
                controller.set_command(wrist_targets_w[:, side_index], ee_quat=wrist_pose_w[:, 3:7])
                jacobian_joint_ids = [joint_id + robot.num_base_dofs for joint_id in arm_ids]
                jacobian = robot.data.body_link_jacobian_w.torch[:, wrist_id, :, jacobian_joint_ids]
                targets[:, arm_ids] = controller.compute(
                    wrist_pose_w[:, :3],
                    wrist_pose_w[:, 3:7],
                    jacobian,
                    robot.data.joint_pos.torch[:, arm_ids],
                )

            limits = robot.data.joint_limits.torch
            targets = torch.clamp(targets, limits[..., 0], limits[..., 1])
            robot.set_joint_position_target_index(target=targets)

        self.needs_wrist_calibration[:] = False
