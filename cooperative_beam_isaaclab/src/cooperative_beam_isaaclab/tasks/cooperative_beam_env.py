"""Variable-team cooperative payload lifting and transport environment."""

from __future__ import annotations

import math
from collections.abc import Sequence

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_apply_inverse, sample_uniform

from .cooperative_beam_env_cfg import CooperativeBeamEnvCfg
from .formation import side_balanced_load_ratios
from .hierarchical_controller import HierarchicalG1Controller
from .trajectory import cuboid_inertia_tensor, payload_tracking_terms, rescue_trajectory


class CooperativeBeamEnv(DirectMARLEnv):
    """Cooperative load transport with a shared payload and decentralized actions.

    Two unilateral spring-damper cables connect each robot's wrists to its payload
    station. They behave like rescue slings: a cable can pull but never push. Every
    cable force is applied to the wrist and the exact equal-and-opposite wrench is
    applied to the timber, preserving the physical coupling between agents.
    """

    cfg: CooperativeBeamEnvCfg

    def __init__(self, cfg: CooperativeBeamEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        expected_actions = cfg.action_spaces[cfg.possible_agents[0]]
        self.num_robots = len(cfg.possible_agents)
        if (
            len(self.robots) != self.num_robots
            or len(cfg.sling_station_y) != self.num_robots
            or len(cfg.sling_station_x) != self.num_robots
        ):
            raise ValueError(
                "possible_agents, robot_cfgs, sling_station_y, and sling_station_x must have the same length: "
                f"{self.num_robots}, {len(self.robots)}, {len(cfg.sling_station_y)}, {len(cfg.sling_station_x)}"
            )

        self._wrist_body_ids: list[list[int]] = []
        for robot in self.robots:
            left_id = self._find_distal_arm_body(robot, "left")
            right_id = self._find_distal_arm_body(robot, "right")
            self._wrist_body_ids.append([left_id, right_id])

        self.actions = {
            agent: torch.zeros((self.num_envs, expected_actions), device=self.device) for agent in cfg.possible_agents
        }
        self._previous_actions = {agent: value.clone() for agent, value in self.actions.items()}
        self.controller = HierarchicalG1Controller(
            self.robots,
            self._wrist_body_ids,
            self.num_envs,
            self.device,
            cfg,
        )

        stations = torch.tensor(cfg.sling_station_y, device=self.device)
        station_x = torch.tensor(cfg.sling_station_x, device=self.device)
        hand_offset = cfg.sling_hand_separation
        self._beam_site_offsets_b = torch.zeros((self.num_robots, 2, 3), device=self.device)
        self._beam_site_offsets_b[:, :, 0] = station_x[:, None]
        # A G1 on +x faces through pi radians, reversing its world-frame left
        # and right directions. Swap that station's sites so its two slings do
        # not cross each other in front of the payload.
        station_side = torch.sign(station_x)
        self._beam_site_offsets_b[:, 0, 1] = stations - station_side * hand_offset
        self._beam_site_offsets_b[:, 1, 1] = stations + station_side * hand_offset
        self._expected_load_ratios = torch.tensor(
            side_balanced_load_ratios(tuple(cfg.sling_station_x)), device=self.device
        )

        self._sling_rest_lengths = torch.zeros((self.num_envs, self.num_robots, 2), device=self.device)
        self._sling_extensions = torch.zeros_like(self._sling_rest_lengths)
        self._sling_tensions = torch.zeros_like(self._sling_rest_lengths)
        self._sling_vectors_w = torch.zeros((self.num_envs, self.num_robots, 2, 3), device=self.device)
        self._needs_sling_calibration = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._payload_masses = torch.full((self.num_envs,), cfg.curriculum_start_mass, device=self.device)
        self._episode_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._cooperative_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._transport_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_jerk_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_jerk_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_transport_load_cv_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_transport_load_target_rmse_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_transport_load_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_peak_sling_tension = torch.zeros(self.num_envs, device=self.device)
        self._latest_position_error = torch.zeros(self.num_envs, device=self.device)
        self._last_failure_terms: dict[str, torch.Tensor] = {}
        self._prev_beam_velocity = torch.zeros((self.num_envs, 3), device=self.device)
        self._prev_beam_acceleration = torch.zeros((self.num_envs, 3), device=self.device)
        self._prev_beam_height = torch.full((self.num_envs,), cfg.beam_cfg.init_state.pos[2], device=self.device)
        self._latest_observations: dict[str, torch.Tensor] | None = None

        self._gravity_unit_w = torch.tensor((0.0, 0.0, -1.0), device=self.device).repeat(self.num_envs, 1)
        self._up_unit_w = torch.tensor((0.0, 0.0, 1.0), device=self.device).repeat(self.num_envs, 1)
        self._beam_long_axis_b = torch.tensor((0.0, 1.0, 0.0), device=self.device).repeat(self.num_envs, 1)

        for index, ids in enumerate(self._wrist_body_ids):
            print(f"[INFO] G1_{index} sling bodies: {[self.robots[index].body_names[i] for i in ids]}")

    def _setup_scene(self) -> None:
        self.robots = [Articulation(robot_cfg) for robot_cfg in self.cfg.robot_cfgs]
        self.beam = RigidObject(self.cfg.beam_cfg)

        target_cfg = sim_utils.CuboidCfg(
            size=(*self.cfg.drop_zone_size, 0.018),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.12, 0.42, 0.10),
                emissive_color=(0.02, 0.08, 0.015),
                roughness=0.8,
            ),
        )
        target_cfg.func(
            "/World/envs/env_0/RescueDropZone",
            target_cfg,
            translation=(self.cfg.carry_delta_xy[0], self.cfg.carry_delta_xy[1], 0.009),
            orientation=(0.0, 0.0, math.sin(self.cfg.target_yaw / 2.0), math.cos(self.cfg.target_yaw / 2.0)),
        )
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=0.9,
                    restitution=0.0,
                )
            ),
        )

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        for index, robot in enumerate(self.robots):
            self.scene.articulations[f"g1_{index}"] = robot
        self.scene.rigid_objects["payload"] = self.beam

        sim_utils.DomeLightCfg(intensity=2200.0, color=(0.78, 0.82, 0.88)).func(
            "/World/DomeLight", sim_utils.DomeLightCfg(intensity=2200.0, color=(0.78, 0.82, 0.88))
        )
        sun_cfg = sim_utils.DistantLightCfg(intensity=1800.0, color=(1.0, 0.91, 0.78), angle=0.35)
        sun_cfg.func(
            "/World/MountainSun",
            sun_cfg,
            orientation=(0.26, -0.18, -0.05, 0.947),
        )

    @staticmethod
    def _find_distal_arm_body(robot: Articulation, side: str) -> int:
        preferred = [
            f"{side}_wrist_yaw_link",
            f"{side}_hand_palm_link",
            f"{side}_palm_link",
            f"{side}_hand_link",
            f"{side}_rubber_hand",
        ]
        for body_name in preferred:
            if body_name in robot.body_names:
                return robot.body_names.index(body_name)
        candidates = [
            index for index, name in enumerate(robot.body_names) if side in name and ("wrist" in name or "hand" in name)
        ]
        if candidates:
            return candidates[-1]
        candidates = [index for index, name in enumerate(robot.body_names) if side in name and "elbow" in name]
        if candidates:
            return candidates[-1]
        raise RuntimeError(f"Could not find a distal {side} arm body. G1 bodies: {robot.body_names}")

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self._previous_actions = {agent: self.actions[agent].clone() for agent in self.cfg.possible_agents}
        self.actions = {agent: torch.clamp(actions[agent], -1.0, 1.0) for agent in self.cfg.possible_agents}

    def _apply_action(self) -> None:
        self.controller.apply([self.actions[agent] for agent in self.cfg.possible_agents])
        for robot in self.robots:
            robot.permanent_wrench_composer.reset()

        self.beam.permanent_wrench_composer.reset()
        self._apply_sling_wrenches()

    def _sling_geometry(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        beam_position = self.beam.data.root_pos_w.torch
        beam_quat = self.beam.data.root_quat_w.torch
        num_sites = 2 * self.num_robots
        local_offsets = self._beam_site_offsets_b.reshape(1, num_sites, 3).expand(self.num_envs, -1, -1)
        repeated_quat = beam_quat[:, None, :].expand(-1, num_sites, -1).reshape(-1, 4)
        offsets_w = quat_apply(repeated_quat, local_offsets.reshape(-1, 3)).reshape(
            self.num_envs, self.num_robots, 2, 3
        )
        beam_sites_w = beam_position[:, None, None, :] + offsets_w

        beam_linear_velocity = self.beam.data.root_lin_vel_w.torch[:, None, None, :]
        beam_angular_velocity = self.beam.data.root_ang_vel_w.torch[:, None, None, :]
        beam_site_velocities = beam_linear_velocity + torch.linalg.cross(
            beam_angular_velocity.expand_as(offsets_w), offsets_w, dim=-1
        )

        wrist_positions = torch.stack(
            [robot.data.body_pos_w.torch[:, ids] for robot, ids in zip(self.robots, self._wrist_body_ids, strict=True)],
            dim=1,
        )
        wrist_velocities = torch.stack(
            [
                robot.data.body_lin_vel_w.torch[:, ids]
                for robot, ids in zip(self.robots, self._wrist_body_ids, strict=True)
            ],
            dim=1,
        )
        cable_vectors = beam_sites_w - wrist_positions
        cable_lengths = torch.linalg.vector_norm(cable_vectors, dim=-1)
        cable_directions = cable_vectors / torch.clamp(cable_lengths[..., None], min=1.0e-5)
        relative_speed = torch.sum((beam_site_velocities - wrist_velocities) * cable_directions, dim=-1)
        return beam_sites_w, offsets_w, cable_vectors, cable_lengths, relative_speed

    def _apply_sling_wrenches(self) -> None:
        _, offsets_w, cable_vectors, cable_lengths, relative_speed = self._sling_geometry()
        if torch.any(self._needs_sling_calibration):
            env_ids = torch.nonzero(self._needs_sling_calibration, as_tuple=False).squeeze(-1)
            self._sling_rest_lengths[env_ids] = cable_lengths[env_ids] + self.cfg.sling_calibration_slack
            self._needs_sling_calibration[env_ids] = False

        extension = cable_lengths - self._sling_rest_lengths
        tension = torch.clamp(
            self.cfg.sling_stiffness * extension + self.cfg.sling_damping * relative_speed,
            min=0.0,
            max=self.cfg.sling_max_tension,
        )
        directions = cable_vectors / torch.clamp(cable_lengths[..., None], min=1.0e-5)
        forces_on_wrists = directions * tension[..., None]

        for index, robot in enumerate(self.robots):
            robot.permanent_wrench_composer.add_forces_and_torques_index(
                forces=forces_on_wrists[:, index],
                body_ids=self._wrist_body_ids[index],
                is_global=True,
            )

        forces_on_beam = -forces_on_wrists
        total_beam_force = forces_on_beam.sum(dim=(1, 2))
        total_beam_torque = torch.linalg.cross(offsets_w, forces_on_beam, dim=-1).sum(dim=(1, 2))
        self.beam.permanent_wrench_composer.add_forces_and_torques_index(
            forces=total_beam_force[:, None, :],
            torques=total_beam_torque[:, None, :],
            body_ids=[0],
            is_global=True,
        )

        self._sling_vectors_w = cable_vectors
        self._sling_extensions = torch.clamp(extension, min=0.0)
        self._sling_tensions = tension

    def _trajectory(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        progress = self.episode_length_buf.float() / float(self.max_episode_length)
        start = torch.tensor(self.cfg.beam_cfg.init_state.pos, device=self.device).repeat(self.num_envs, 1)
        transport_scale = self._transport_scale()
        target_local, target_yaw = rescue_trajectory(
            progress,
            start,
            self.cfg.lift_height,
            (
                transport_scale * self.cfg.carry_delta_xy[0],
                transport_scale * self.cfg.carry_delta_xy[1],
            ),
            self.cfg.final_beam_height,
            transport_scale * self.cfg.target_yaw,
        )
        return target_local + self.scene.env_origins, target_yaw, progress

    def _transport_scale(self) -> float:
        """Return the training curriculum scale or a fixed evaluation value."""
        if self.cfg.transport_scale_override is not None:
            return min(max(float(self.cfg.transport_scale_override), 0.0), 1.0)
        curriculum_span = max(
            self.cfg.transport_curriculum_end_steps - self.cfg.transport_curriculum_start_steps,
            1,
        )
        return min(
            max((self.common_step_counter - self.cfg.transport_curriculum_start_steps) / curriculum_span, 0.0),
            1.0,
        )

    def _robot_up_z(self) -> torch.Tensor:
        return torch.stack(
            [quat_apply(robot.data.root_quat_w.torch, self._up_unit_w)[:, 2] for robot in self.robots], dim=-1
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        target_position, _, progress = self._trajectory()
        beam_position = self.beam.data.root_pos_w.torch
        beam_velocity = self.beam.data.root_lin_vel_w.torch
        beam_quat = self.beam.data.root_quat_w.torch
        robot_positions = torch.stack([robot.data.root_pos_w.torch for robot in self.robots], dim=1)
        robot_velocities = torch.stack([robot.data.root_lin_vel_w.torch for robot in self.robots], dim=1)
        team_loads = self._sling_tensions.sum(dim=-1)
        load_sum = torch.clamp(team_loads.sum(dim=-1, keepdim=True), min=1.0)
        load_ratios = team_loads / load_sum

        observations: dict[str, torch.Tensor] = {}
        for index, (agent, robot) in enumerate(zip(self.cfg.possible_agents, self.robots, strict=True)):
            root_quat = robot.data.root_quat_w.torch
            joint_pos = robot.data.joint_pos.torch[:, self.controller.body_joint_ids]
            default_joint_pos = robot.data.default_joint_pos.torch[:, self.controller.body_joint_ids]
            joint_vel = robot.data.joint_vel.torch[:, self.controller.body_joint_ids]

            beam_relative = quat_apply_inverse(root_quat, beam_position - robot.data.root_pos_w.torch)
            beam_velocity_relative = quat_apply_inverse(root_quat, beam_velocity - robot.data.root_lin_vel_w.torch)
            target_relative = quat_apply_inverse(root_quat, target_position - robot.data.root_pos_w.torch)
            sling_vectors_local = quat_apply_inverse(
                root_quat[:, None, :].expand(-1, 2, -1).reshape(-1, 4),
                self._sling_vectors_w[:, index].reshape(-1, 3),
            ).reshape(self.num_envs, 6)

            teammate_features = []
            teammate_loads = []
            for teammate_index in range(self.num_robots):
                if teammate_index == index:
                    continue
                teammate_features.extend(
                    [
                        quat_apply_inverse(root_quat, robot_positions[:, teammate_index] - robot_positions[:, index]),
                        quat_apply_inverse(root_quat, robot_velocities[:, teammate_index] - robot_velocities[:, index]),
                    ]
                )
                teammate_loads.append(load_ratios[:, teammate_index : teammate_index + 1])

            obs = torch.cat(
                (
                    self.cfg.root_velocity_scale * robot.data.root_lin_vel_b.torch,
                    self.cfg.root_velocity_scale * robot.data.root_ang_vel_b.torch,
                    quat_apply_inverse(root_quat, self._gravity_unit_w),
                    joint_pos - default_joint_pos,
                    self.cfg.joint_velocity_scale * joint_vel,
                    beam_relative,
                    self.cfg.root_velocity_scale * beam_velocity_relative,
                    beam_quat,
                    sling_vectors_local,
                    load_ratios[:, index : index + 1],
                    target_relative,
                    progress[:, None],
                    self.actions[agent],
                    *teammate_features,
                    *teammate_loads,
                ),
                dim=-1,
            )
            if obs.shape[-1] != self.cfg.observation_spaces[agent]:
                raise RuntimeError(
                    f"Observation size is {obs.shape[-1]}, configured as {self.cfg.observation_spaces[agent]}"
                )
            observations[agent] = obs

        self._latest_observations = observations
        return observations

    def _get_states(self) -> torch.Tensor:
        observations = self._latest_observations or self._get_observations()
        state = torch.cat(
            [*(observations[agent] for agent in self.cfg.possible_agents), self._payload_masses[:, None]], dim=-1
        )
        if state.shape[-1] != self.cfg.state_space:
            raise RuntimeError(f"Central state size is {state.shape[-1]}, configured as {self.cfg.state_space}")
        return state

    def _task_failure_terms(self) -> dict[str, torch.Tensor]:
        robot_heights = torch.stack([robot.data.root_pos_w.torch[:, 2] for robot in self.robots], dim=-1)
        robot_up = self._robot_up_z()
        robot_fall = torch.any(robot_heights < self.cfg.minimum_robot_height, dim=-1)
        robot_tilt = torch.any(robot_up < 0.35, dim=-1)
        beam_drop = self.beam.data.root_pos_w.torch[:, 2] < self.cfg.minimum_beam_height
        sling_break = torch.any(self._sling_extensions > self.cfg.max_sling_extension, dim=(1, 2))
        return {
            "robot_fall": robot_fall,
            "robot_tilt": robot_tilt,
            "payload_drop": beam_drop,
            "sling_overextension": sling_break,
        }

    def _task_failure(self) -> torch.Tensor:
        return torch.stack(tuple(self._task_failure_terms().values()), dim=0).any(dim=0)

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        target_position, target_yaw, progress = self._trajectory()
        beam_position = self.beam.data.root_pos_w.torch
        beam_velocity = self.beam.data.root_lin_vel_w.torch
        beam_quat = self.beam.data.root_quat_w.torch
        beam_up_z = quat_apply(beam_quat, self._up_unit_w)[:, 2]
        beam_long_axis = quat_apply(beam_quat, self._beam_long_axis_b)
        desired_heading = torch.stack(
            (-torch.sin(target_yaw), torch.cos(target_yaw), torch.zeros_like(target_yaw)), dim=-1
        )
        heading_alignment = torch.sum(beam_long_axis * desired_heading, dim=-1)
        team_loads = self._sling_tensions.sum(dim=-1)
        transport_active = (progress >= 0.30) & (progress <= 0.80)
        all_robots_supporting = torch.all(team_loads > self.cfg.cooperative_tension_threshold, dim=-1)
        self._transport_steps += transport_active.long()
        self._cooperative_steps += (transport_active & all_robots_supporting).long()

        control_dt = self.cfg.sim.dt * self.cfg.decimation
        beam_acceleration = (beam_velocity - self._prev_beam_velocity) / control_dt
        beam_jerk = torch.linalg.vector_norm(
            (beam_acceleration - self._prev_beam_acceleration) / control_dt,
            dim=-1,
        )
        self._episode_jerk_sum += beam_jerk
        self._episode_jerk_steps += 1
        self._prev_beam_velocity = beam_velocity.clone()
        self._prev_beam_acceleration = beam_acceleration.clone()
        terms = payload_tracking_terms(
            beam_position,
            target_position,
            beam_up_z,
            heading_alignment,
            team_loads,
            self._expected_load_ratios,
        )
        self._episode_transport_load_cv_sum += terms["load_cv"] * transport_active.float()
        self._episode_transport_load_target_rmse_sum += terms["load_target_rmse"] * transport_active.float()
        self._episode_transport_load_steps += transport_active.long()
        self._episode_peak_sling_tension = torch.maximum(
            self._episode_peak_sling_tension,
            self._sling_tensions.amax(dim=(1, 2)),
        )
        self._latest_position_error = terms["position_error"].clone()

        lift_progress = torch.clamp(beam_position[:, 2] - self._prev_beam_height, -0.025, 0.025)
        action_rate = torch.stack(
            [
                torch.mean(torch.square(self.actions[agent] - self._previous_actions[agent]), dim=-1)
                for agent in self.cfg.possible_agents
            ],
            dim=-1,
        ).mean(dim=-1)
        sling_extension = torch.mean(torch.square(self._sling_extensions), dim=(1, 2))
        upright = torch.clamp(self._robot_up_z(), min=0.0).mean(dim=-1)
        failure = self._task_failure()

        final_phase = progress > 0.92
        success_now = (
            final_phase
            & (terms["position_error"] < self.cfg.success_position_tolerance)
            & (heading_alignment > self.cfg.success_heading_tolerance)
            & (beam_up_z > 0.90)
        )
        new_success = success_now & ~self._episode_succeeded
        self._episode_succeeded |= success_now

        team_reward = (
            self.cfg.reward_position * terms["position"]
            + self.cfg.reward_level * terms["level"]
            + self.cfg.reward_heading * terms["heading"]
            + self.cfg.reward_load_balance * terms["load_balance"]
            + self.cfg.reward_upright * upright
            + self.cfg.reward_lift_progress * lift_progress
            + self.cfg.reward_success * new_success.float()
            - self.cfg.penalty_sling_extension * sling_extension
            - self.cfg.penalty_action_rate * action_rate
            - self.cfg.penalty_termination * failure.float()
        )
        self._prev_beam_height = beam_position[:, 2].clone()

        log = self.extras.setdefault("log", {})
        log["Reward/position"] = terms["position"].mean()
        log["Reward/load_balance"] = terms["load_balance"].mean()
        log["Metrics/payload_position_error"] = terms["position_error"].mean()
        log["Metrics/payload_mass_kg"] = self._payload_masses.mean()
        log["Metrics/payload_kg_per_robot"] = self._payload_masses.mean() / self.num_robots
        log["Metrics/payload_kg_per_arm"] = self._payload_masses.mean() / (2 * self.num_robots)
        log["Metrics/transport_curriculum_scale"] = self._transport_scale()
        log["Metrics/max_sling_tension_n"] = self._sling_tensions.amax(dim=(1, 2)).mean()
        log["Metrics/load_coefficient_of_variation"] = terms["load_cv"].mean()
        log["Metrics/load_target_rmse"] = terms["load_target_rmse"].mean()
        log["Metrics/team_upright"] = upright.mean()
        return {agent: team_reward for agent in self.cfg.possible_agents}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self._last_failure_terms = self._task_failure_terms()
        failure = torch.stack(tuple(self._last_failure_terms.values()), dim=0).any(dim=0)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return (
            {agent: failure for agent in self.cfg.possible_agents},
            {agent: time_out for agent in self.cfg.possible_agents},
        )

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self.robots[0]._ALL_INDICES
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        episode_log: dict[str, float] = {}
        if hasattr(self, "_episode_succeeded"):
            episode_log["Metrics/success_rate"] = self._episode_succeeded[env_ids].float().mean().item()
            cooperative_ratio = self._cooperative_steps[env_ids].float() / torch.clamp(
                self._transport_steps[env_ids], min=1
            )
            mean_jerk = self._episode_jerk_sum[env_ids] / torch.clamp(
                self._episode_jerk_steps[env_ids], min=1
            )
            mean_transport_load_cv = self._episode_transport_load_cv_sum[env_ids] / torch.clamp(
                self._episode_transport_load_steps[env_ids], min=1
            )
            mean_transport_load_target_rmse = self._episode_transport_load_target_rmse_sum[env_ids] / torch.clamp(
                self._episode_transport_load_steps[env_ids], min=1
            )
            episode_log["Metrics/cooperative_time_ratio"] = cooperative_ratio.mean().item()
            episode_log["Metrics/mean_payload_jerk_mps3"] = mean_jerk.mean().item()
            episode_log["Metrics/final_payload_position_error_m"] = self._latest_position_error[env_ids].mean().item()
            episode_log["Metrics/mean_transport_load_cv"] = mean_transport_load_cv.mean().item()
            episode_log["Metrics/mean_transport_load_target_rmse"] = (
                mean_transport_load_target_rmse.mean().item()
            )
            episode_log["Metrics/episode_peak_sling_tension_n"] = (
                self._episode_peak_sling_tension[env_ids].mean().item()
            )
            episode_log["Metrics/episode_payload_kg_per_robot"] = (
                self._payload_masses[env_ids].mean().item() / self.num_robots
            )
            episode_log["Metrics/episode_payload_kg_per_arm"] = (
                self._payload_masses[env_ids].mean().item() / (2 * self.num_robots)
            )
        super()._reset_idx(env_ids)
        self.extras.setdefault("log", {}).update(episode_log)

        for agent, robot in zip(self.cfg.possible_agents, self.robots, strict=True):
            root_pose = robot.data.default_root_pose.torch[env_ids].clone()
            root_pose[:, :3] += self.scene.env_origins[env_ids]
            root_pose[:, :2] += sample_uniform(
                -self.cfg.reset_root_xy_noise,
                self.cfg.reset_root_xy_noise,
                (len(env_ids), 2),
                self.device,
            )
            root_velocity = torch.zeros((len(env_ids), 6), device=self.device)
            joint_position = robot.data.default_joint_pos.torch[env_ids].clone()
            joint_position[:, self.controller.body_joint_ids] += sample_uniform(
                -self.cfg.reset_joint_noise,
                self.cfg.reset_joint_noise,
                (len(env_ids), len(self.controller.body_joint_ids)),
                self.device,
            )
            joint_velocity = torch.zeros_like(joint_position)
            robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
            robot.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)
            robot.write_joint_position_to_sim_index(position=joint_position, env_ids=env_ids)
            robot.write_joint_velocity_to_sim_index(velocity=joint_velocity, env_ids=env_ids)
            robot.set_joint_position_target_index(target=joint_position, env_ids=env_ids)
            if hasattr(self, "actions"):
                self.actions[agent][env_ids] = 0.0
                self._previous_actions[agent][env_ids] = 0.0

        if hasattr(self, "controller"):
            self.controller.reset(env_ids)

        beam_pose = self.beam.data.default_root_pose.torch[env_ids].clone()
        beam_pose[:, :3] += self.scene.env_origins[env_ids]
        beam_pose[:, :2] += sample_uniform(
            -self.cfg.reset_beam_xy_noise,
            self.cfg.reset_beam_xy_noise,
            (len(env_ids), 2),
            self.device,
        )
        beam_velocity = torch.zeros((len(env_ids), 6), device=self.device)
        self.beam.write_root_pose_to_sim_index(root_pose=beam_pose, env_ids=env_ids)
        self.beam.write_root_velocity_to_sim_index(root_velocity=beam_velocity, env_ids=env_ids)

        curriculum_fraction = min(self.common_step_counter / max(self.cfg.curriculum_steps, 1), 1.0)
        current_max_mass = self.cfg.curriculum_start_mass + curriculum_fraction * (
            self.cfg.curriculum_end_mass - self.cfg.curriculum_start_mass
        )
        payload_masses = sample_uniform(
            self.cfg.curriculum_start_mass,
            current_max_mass,
            (len(env_ids), 1),
            self.device,
        )
        self.beam.set_masses_index(masses=payload_masses, body_ids=[0], env_ids=env_ids)
        self.beam.set_inertias_index(
            inertias=cuboid_inertia_tensor(payload_masses, self.cfg.payload_size),
            body_ids=[0],
            env_ids=env_ids,
        )

        if hasattr(self, "_payload_masses"):
            self._payload_masses[env_ids] = payload_masses[:, 0]
            self._needs_sling_calibration[env_ids] = True
            self._sling_rest_lengths[env_ids] = 0.0
            self._sling_extensions[env_ids] = 0.0
            self._sling_tensions[env_ids] = 0.0
            self._episode_succeeded[env_ids] = False
            self._cooperative_steps[env_ids] = 0
            self._transport_steps[env_ids] = 0
            self._episode_jerk_sum[env_ids] = 0.0
            self._episode_jerk_steps[env_ids] = 0
            self._episode_transport_load_cv_sum[env_ids] = 0.0
            self._episode_transport_load_target_rmse_sum[env_ids] = 0.0
            self._episode_transport_load_steps[env_ids] = 0
            self._episode_peak_sling_tension[env_ids] = 0.0
            self._latest_position_error[env_ids] = 0.0
            self._prev_beam_velocity[env_ids] = 0.0
            self._prev_beam_acceleration[env_ids] = 0.0
            self._prev_beam_height[env_ids] = beam_pose[:, 2]
            self._latest_observations = None
