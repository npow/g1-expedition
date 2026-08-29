"""Configurable disaster-recovery payload tasks for cooperative G1 teams."""

from __future__ import annotations

import math
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab_assets import G1_29DOF_CFG

from .formation import even_stations, robot_station_poses, sling_station_x

LOCAL_OBSERVATION_DIM = 98
TEAMMATE_TOKEN_DIM = 7
ACTION_DIM = 10


def team_observation_dim(team_size: int) -> int:
    """Return one actor's observation width for a variable-size team."""
    if team_size < 2:
        raise ValueError("Cooperative transport requires at least two robots")
    return LOCAL_OBSERVATION_DIM + (team_size - 1) * TEAMMATE_TOKEN_DIM


def _agents(team_size: int) -> list[str]:
    return [f"g1_{index}" for index in range(team_size)]


def _action_spaces(team_size: int) -> dict[str, int]:
    return {agent: ACTION_DIM for agent in _agents(team_size)}


def _observation_spaces(team_size: int) -> dict[str, int]:
    width = team_observation_dim(team_size)
    return {agent: width for agent in _agents(team_size)}


def _g1_cfg(index: int, x_position: float, y_position: float, yaw: float) -> ArticulationCfg:
    # AGILE was trained for Isaac Lab's 29-DoF G1 body. Using this exact asset is
    # part of the frozen policy contract; the older G1_CFG has incompatible
    # torso, elbow, and finger joints.
    cfg = G1_29DOF_CFG.replace(prim_path=f"/World/envs/env_.*/G1_{index}")
    cfg.init_state.pos = (x_position, y_position, 0.74)
    cfg.init_state.rot = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
    cfg.init_state.joint_pos.update(
        {
            "left_shoulder_pitch_joint": 0.25,
            "right_shoulder_pitch_joint": 0.25,
            "left_shoulder_roll_joint": 0.16,
            "right_shoulder_roll_joint": -0.16,
            ".*_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.55,
            "right_elbow_joint": -0.55,
            ".*_wrist_.*_joint": 0.0,
            "waist_.*_joint": 0.0,
        }
    )
    return cfg


def _robot_cfgs(stations: Sequence[float], payload_width: float) -> list[ArticulationCfg]:
    station_tuple = tuple(stations)
    return [
        _g1_cfg(index, x_position, y_position, yaw)
        for index, (x_position, y_position, yaw) in enumerate(
            robot_station_poses(station_tuple, payload_width)
        )
    ]


def _payload_cfg(
    prim_name: str,
    size: tuple[float, float, float],
    mass: float,
    color: tuple[float, float, float],
    roughness: float,
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{prim_name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=2.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=roughness),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, size[2] / 2.0 + 0.015),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )


TIMBER_STATIONS = (-0.72, 0.0, 0.72)
TIMBER_SIZE = (0.16, 2.20, 0.12)


@configclass
class CooperativeBeamEnvCfg(DirectMARLEnvCfg):
    """Three G1s moving a fallen timber; the backwards-compatible default task."""

    decimation = 4
    episode_length_s = 12.0
    possible_agents = _agents(3)
    # Hierarchical action per robot:
    #   4 AGILE commands [vx, vy, yaw rate, hip height]
    #   6 left/right wrist position offsets for batched differential IK
    action_spaces = _action_spaces(3)
    observation_spaces = _observation_spaces(3)
    state_space = 3 * team_observation_dim(3) + 1

    sim: SimulationCfg = SimulationCfg(
        # Match the 200 Hz physics / 50 Hz policy cadence used by Isaac Lab's
        # reference AGILE loco-manipulation environment.
        dt=1.0 / 200.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=0.9,
            restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=6.0, replicate_physics=True)

    payload_label = "fallen timber"
    payload_size = TIMBER_SIZE
    sling_station_y = TIMBER_STATIONS
    robot_cfgs: list[ArticulationCfg] = _robot_cfgs(TIMBER_STATIONS, TIMBER_SIZE[0])
    beam_cfg: RigidObjectCfg = _payload_cfg(
        "Timber",
        TIMBER_SIZE,
        mass=15.0,
        color=(0.24, 0.095, 0.025),
        roughness=0.88,
    )
    drop_zone_size = (0.55, 2.50)

    agile_body_joint_patterns = [
        ".*_shoulder_.*_joint",
        ".*_elbow_joint",
        ".*_wrist_.*_joint",
        ".*_hip_.*_joint",
        ".*_knee_joint",
        ".*_ankle_.*_joint",
        "waist_.*_joint",
    ]
    agile_leg_joint_patterns = [".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"]
    agile_policy_path = f"{ISAACLAB_NUCLEUS_DIR}/Policies/Agile/agile_locomotion.pt"
    agile_policy_input_dim = 83
    agile_policy_output_dim = 12
    agile_policy_output_scale = 0.25
    command_velocity_scale = (0.55, 0.30, 0.65)
    command_hip_height = (0.64, 0.78)

    wrist_action_scale = (0.10, 0.09, 0.38)
    wrist_ik_damping = 0.08
    arm_joint_patterns = [
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_pitch_joint",
        ".*_wrist_roll_joint",
        ".*_wrist_yaw_joint",
    ]
    joint_velocity_scale = 0.08
    root_velocity_scale = 0.35

    sling_hand_separation = 0.13
    sling_station_x = sling_station_x(len(TIMBER_STATIONS), TIMBER_SIZE[0])
    sling_calibration_slack = 0.015
    sling_stiffness = 620.0
    sling_damping = 32.0
    # Per-sling transient guard. This remains far above the nominal static
    # loads in every profile but prevents one cable shock from toppling a G1.
    sling_max_tension = 80.0
    max_sling_extension = 0.40
    cooperative_tension_threshold = 5.0

    curriculum_start_mass = 8.0
    curriculum_end_mass = 18.0
    # common_step_counter counts vector control steps, not transitions across
    # all environments. These values fit inside a 10k x 24-step MAPPO run.
    curriculum_steps = 180_000
    transport_curriculum_start_steps = 24_000
    transport_curriculum_end_steps = 150_000
    # Evaluation can pin the requested displacement independently of a fresh
    # environment's common_step_counter. None preserves the training curriculum.
    transport_scale_override: float | None = None

    # A 16 cm lift clears the 12 cm timber while remaining inside the reusable
    # AGILE + wrist-IK controller's tested workspace.
    lift_height = 0.16
    carry_delta_xy = (0.85, 0.20)
    final_beam_height = 0.14
    target_yaw = 0.35
    success_position_tolerance = 0.14
    success_heading_tolerance = 0.90

    reset_joint_noise = 0.035
    reset_root_xy_noise = 0.025
    reset_beam_xy_noise = 0.025
    minimum_robot_height = 0.45
    minimum_beam_height = 0.025

    reward_position = 5.0
    reward_level = 1.2
    reward_heading = 0.6
    reward_load_balance = 1.0
    reward_upright = 1.0
    reward_lift_progress = 18.0
    reward_success = 20.0
    penalty_sling_extension = 1.5
    penalty_action_rate = 0.04
    penalty_termination = 12.0

    def configure_team_size(self, team_size: int) -> None:
        """Override team size before constructing the environment.

        This is used by scaling evaluations to vary robot count while holding
        payload geometry and mass fixed. Training task defaults remain the
        recognizable 2/3/5-robot configurations declared below.
        """
        stations = even_stations(team_size, self.payload_size[1])
        self.possible_agents = _agents(team_size)
        self.action_spaces = _action_spaces(team_size)
        self.observation_spaces = _observation_spaces(team_size)
        self.state_space = team_size * team_observation_dim(team_size) + 1
        self.sling_station_y = stations
        self.sling_station_x = sling_station_x(team_size, self.payload_size[0])
        self.robot_cfgs = _robot_cfgs(stations, self.payload_size[0])


CRATE_STATIONS = (-0.30, 0.30)
CRATE_SIZE = (0.55, 1.00, 0.36)


@configclass
class CooperativeCrateEnvCfg(CooperativeBeamEnvCfg):
    """Two G1s moving a dense mountain-rescue equipment crate."""

    possible_agents = _agents(2)
    action_spaces = _action_spaces(2)
    observation_spaces = _observation_spaces(2)
    state_space = 2 * team_observation_dim(2) + 1

    payload_label = "rescue equipment crate"
    payload_size = CRATE_SIZE
    sling_station_y = CRATE_STATIONS
    robot_cfgs: list[ArticulationCfg] = _robot_cfgs(CRATE_STATIONS, CRATE_SIZE[0])
    beam_cfg: RigidObjectCfg = _payload_cfg(
        "RescueEquipmentCrate",
        CRATE_SIZE,
        mass=7.0,
        color=(0.72, 0.22, 0.035),
        roughness=0.70,
    )
    drop_zone_size = (0.90, 1.35)
    sling_station_x = sling_station_x(len(CRATE_STATIONS), CRATE_SIZE[0])
    curriculum_start_mass = 4.0
    curriculum_end_mass = 10.0
    final_beam_height = CRATE_SIZE[2] / 2.0 + 0.08


GIRDER_STATIONS = (-1.18, -0.59, 0.0, 0.59, 1.18)
GIRDER_SIZE = (0.22, 3.25, 0.20)


@configclass
class CooperativeGirderEnvCfg(CooperativeBeamEnvCfg):
    """Five G1s clearing a collapsed steel footbridge girder."""

    possible_agents = _agents(5)
    action_spaces = _action_spaces(5)
    observation_spaces = _observation_spaces(5)
    state_space = 5 * team_observation_dim(5) + 1

    payload_label = "collapsed footbridge girder"
    payload_size = GIRDER_SIZE
    sling_station_y = GIRDER_STATIONS
    robot_cfgs: list[ArticulationCfg] = _robot_cfgs(GIRDER_STATIONS, GIRDER_SIZE[0])
    beam_cfg: RigidObjectCfg = _payload_cfg(
        "CollapsedFootbridgeGirder",
        GIRDER_SIZE,
        mass=24.0,
        color=(0.24, 0.28, 0.31),
        roughness=0.40,
    )
    drop_zone_size = (0.65, 3.60)
    sling_station_x = sling_station_x(len(GIRDER_STATIONS), GIRDER_SIZE[0])
    curriculum_start_mass = 15.0
    curriculum_end_mass = 30.0
    success_position_tolerance = 0.18


__all__ = [
    "CooperativeBeamEnvCfg",
    "CooperativeCrateEnvCfg",
    "CooperativeGirderEnvCfg",
    "team_observation_dim",
]
