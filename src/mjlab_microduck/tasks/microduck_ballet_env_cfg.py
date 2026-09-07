"""Microduck Ballet V1 — commanded, one-legged pirouette.

This is deliberately a one-leg-hold-first skill with a timed deployment turn:

    twist = [active, free_leg_side, turn]

V1 fixes ``free_leg_side=+1`` (left leg lifted, right leg supporting). A4 holds
the observed turn command at zero through iteration 1200, then synchronously
ramps command and reward to 0.2 and 0.4 rad/s. The runtime switches back to a
standing policy after the timed skill.

The environment is derived from BallKick rather than rebuilt from mjlab's
base.  BallKick is the closest proven sim2real recipe: full ground-contact
robot, a named support foot, unified 61D observations, standing starts, BAM,
encoder/IMU noise and delayed push/CoM curricula.  The ball entity and all
ball-specific terms are removed below.

V1 is intentionally conservative: one fixed support side and a moderate fixed
yaw rate.  It is a training scaffold, not a claim that the untrained policy is
hardware safe.  Add side conditioning, reverse turns and faster rotation only
after single-support balance converges.
"""

from copy import deepcopy

from mjlab.managers import CurriculumTermCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_ball_kick_env_cfg import (
    VELOCITY_PUSH_RANGE,
    MicroduckBallKickRlCfg,
    make_microduck_ball_kick_env_cfg,
)

EPISODE_LENGTH_S = 12.0
COMMAND_DWELL_S = (EPISODE_LENGTH_S, EPISODE_LENGTH_S)
TURN_RATE = 0.4
TURN_START_ITER = 1200
TURN_FULL_ITER = 1600

# Left leg is the free/display leg; the right leg is the support/pivot leg.
FREE_LEG_SIDE = 1.0
FREE_FOOT_SITE = "left_foot"
SUPPORT_SENSOR = "support_foot_ground_contact"
FREE_SENSOR = "free_foot_ground_contact"
SUPPORT_FOOT_SITE = "right_foot"

STAND_Z = 0.115
FREE_FOOT_Z = 0.055

_FREE_LEG_JOINTS = [0, 1, 2, 3, 4]
_NECK_JOINTS = [5, 6, 7, 8]
_FREE_HIP_YAW = [0]
_SUPPORT_HIP_YAW = [9]

# Actuator order is the deployed 14D action order.  This table is deliberately
# exhaustive: every joint is either assigned an intended role or explicitly
# held at HOME.  The support-leg balance joints remain free because lateral
# weight transfer cannot be achieved with a rigid support chain.
BALLET_JOINT_ROLES = (
    (0, "left_hip_yaw", "hold_home"),
    (1, "left_hip_roll", "free_leg_pose"),
    (2, "left_hip_pitch", "free_leg_pose"),
    (3, "left_knee", "free_leg_pose"),
    (4, "left_ankle", "free_leg_pose"),
    (5, "neck_pitch", "hold_home"),
    (6, "head_pitch", "hold_home"),
    (7, "head_yaw", "hold_home"),
    (8, "head_roll", "hold_home"),
    (9, "right_hip_yaw", "body_turn_only"),
    (10, "right_hip_roll", "support_balance"),
    (11, "right_hip_pitch", "support_balance"),
    (12, "right_knee", "support_balance"),
    (13, "right_ankle", "support_balance"),
)

# A readable lifted-leg silhouette, not a human anatomical ballet pose.  The
# support leg is deliberately omitted so it remains free to balance and pivot.
FREE_LEG_TARGET = {
    1: -0.25,  # hip roll: move it away from the support leg
    2: -0.75,  # hip pitch
    3: 0.85,   # flex knee for ground clearance
    4: 0.15,   # tuck the foot rather than drive the ankle stop
}


def make_microduck_ballet_env_cfg(play: bool = False):
    """Build the fixed-side Ballet V1 environment."""

    # kick_foot="left" makes BallKick's named support sensor select the RIGHT
    # foot, which is exactly Ballet V1's support side.
    cfg = make_microduck_ball_kick_env_cfg(play=play, kick_foot="left")
    cfg.episode_length_s = EPISODE_LENGTH_S

    # Remove the prop while retaining the proven full-collision robot recipe.
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.events.pop("reset_ball", None)
    for group in ("actor", "critic"):
        cfg.observations[group].terms.pop("ball_position", None)
        cfg.observations[group].terms.pop("ball_velocity", None)

    free_foot_ground = ContactSensorCfg(
        name=FREE_SENSOR,
        primary=ContactMatch(
            mode="geom",
            pattern=r"^left_foot_collision$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
    )
    cfg.scene.sensors = (*cfg.scene.sensors, free_foot_ground)

    # The observed turn command starts at zero and is raised by curriculum at
    # exactly the same steps as the turn rewards.  The policy never receives a
    # latent turn request while it is supposed to learn pure single support.
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = COMMAND_DWELL_S
    command.debug_vis = False
    cfg.commands["twist"] = microduck_mdp.BalletCommandCfg(
        **{
            **vars(command),
            "active": 1.0,
            "free_leg_side": FREE_LEG_SIDE,
            "turn": 0.0,
        }
    )

    # Preserve the shared runtime layout explicitly: twist(3) + zero head(4)
    # + zero body(6), for a 61D actor observation and 14D action.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding,
            params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding,
            params={"dim": 6},
        )

    for name in (
        "ball_forward_velocity",
        "ball_speed_overshoot",
        "support_foot_grounded",
        "pose_stand_legs",
    ):
        cfg.rewards.pop(name, None)

    # The neck is not an actuator for this skill.  Gaussian pose accuracy plus
    # L1 position, raw-action and velocity costs reject both a static head
    # counterweight and the rotating-head shortcut seen in model_750.pt.
    cfg.rewards["pose_stand_neck"].weight = 1.5
    cfg.rewards["pose_stand_neck"].params.update(
        {"std": 0.20, "joint_indices": _NECK_JOINTS}
    )
    cfg.rewards["neck_home_l1"] = RewardTermCfg(
        func=microduck_mdp.pose_l1_penalty,
        weight=3.0,
        params={"joint_indices": _NECK_JOINTS, "target_overrides": None},
    )
    cfg.rewards["neck_action_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_action_l2,
        weight=-0.20,
        params={"joint_indices": _NECK_JOINTS},
    )
    cfg.rewards["neck_joint_vel_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_joint_vel_l2,
        weight=-0.02,
    )
    cfg.rewards["height_stand"].weight = 2.0
    cfg.rewards["height_stand"].params["std"] = 0.04
    cfg.rewards["height_stand_l1"] = RewardTermCfg(
        func=microduck_mdp.height_l1_penalty,
        weight=20.0,
        params={
            "target_height": STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["upright_linear"] = RewardTermCfg(
        func=microduck_mdp.body_upright_linear,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # Dense bootstrap: form and hold the one-leg silhouette.
    cfg.rewards["unique_support"] = RewardTermCfg(
        func=microduck_mdp.ballet_unique_support,
        weight=2.0,
        params={
            "support_sensor": SUPPORT_SENSOR,
            "free_sensor": FREE_SENSOR,
            "command_name": "twist",
        },
    )
    # Direct the trunk/pelvis over the support foot.  A3 rewarded whole-robot
    # CoM, which could be improved by moving the head instead of the hips.
    support_site_cfg = SceneEntityCfg(
        "robot", site_names=(SUPPORT_FOOT_SITE,)
    )
    cfg.rewards["trunk_over_support"] = RewardTermCfg(
        func=microduck_mdp.ballet_trunk_over_support,
        weight=3.0,
        params={
            "command_name": "twist",
            "std": 0.04,
            "asset_cfg": support_site_cfg,
        },
    )
    cfg.rewards["trunk_over_support_l1"] = RewardTermCfg(
        func=microduck_mdp.ballet_trunk_over_support_l1,
        weight=10.0,
        params={"command_name": "twist", "asset_cfg": support_site_cfg},
    )
    cfg.rewards["contact_violation"] = RewardTermCfg(
        func=microduck_mdp.ballet_contact_violation,
        weight=-2.0,
        params={
            "support_sensor": SUPPORT_SENSOR,
            "free_sensor": FREE_SENSOR,
            "command_name": "twist",
        },
    )
    cfg.rewards["free_foot_height"] = RewardTermCfg(
        func=microduck_mdp.ballet_free_foot_height,
        weight=2.5,
        params={
            "target_height": FREE_FOOT_Z,
            "std": 0.025,
            "command_name": "twist",
            "asset_cfg": SceneEntityCfg("robot", site_names=(FREE_FOOT_SITE,)),
        },
    )
    cfg.rewards["free_leg_pose"] = RewardTermCfg(
        func=microduck_mdp.ballet_commanded_pose,
        weight=2.5,
        params={
            "when_active": True,
            "command_name": "twist",
            "std": 0.35,
            "joint_indices": _FREE_LEG_JOINTS,
            "target_overrides": FREE_LEG_TARGET,
        },
    )
    cfg.rewards["free_leg_pose_l1"] = RewardTermCfg(
        func=microduck_mdp.ballet_commanded_pose_l1,
        weight=1.0,
        params={
            "when_active": True,
            "command_name": "twist",
            "joint_indices": _FREE_LEG_JOINTS,
            "target_overrides": FREE_LEG_TARGET,
        },
    )
    # These two hip-yaw joints have distinct roles: the free hip-yaw never
    # participates, while the support hip-yaw is released only with the turn
    # command.  All other leg joints are necessary for pose or balance.
    cfg.rewards["free_hip_yaw_home_l1"] = RewardTermCfg(
        func=microduck_mdp.pose_l1_penalty,
        weight=2.0,
        params={"joint_indices": _FREE_HIP_YAW, "target_overrides": None},
    )
    cfg.rewards["free_hip_yaw_action_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_action_l2,
        weight=-0.20,
        params={"joint_indices": _FREE_HIP_YAW},
    )
    cfg.rewards["support_hip_yaw_home_l1"] = RewardTermCfg(
        func=microduck_mdp.ballet_home_pose_l1_when_not_turning,
        weight=2.0,
        params={
            "joint_indices": _SUPPORT_HIP_YAW,
            "command_name": "twist",
        },
    )
    cfg.rewards["support_hip_yaw_action_l2"] = RewardTermCfg(
        func=microduck_mdp.ballet_joint_action_l2_when_not_turning,
        weight=-0.20,
        params={
            "joint_indices": _SUPPORT_HIP_YAW,
            "command_name": "twist",
        },
    )

    # Pirouette objective.  Track yaw rate directly rather than prescribing a
    # joint trajectory; the L1 companion keeps a useful gradient before the
    # Gaussian target becomes reachable.
    cfg.rewards["turn_rate_track"] = RewardTermCfg(
        func=microduck_mdp.ballet_turn_rate_track,
        weight=0.0,
        params={
            "command_name": "twist",
            "std": 0.30,
            "upright_std": 0.25,
            "neck_std": 0.20,
            "support_sensor": SUPPORT_SENSOR,
            "free_sensor": FREE_SENSOR,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["turn_rate_l1"] = RewardTermCfg(
        func=microduck_mdp.ballet_turn_rate_l1,
        weight=0.0,
        params={
            "command_name": "twist",
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["planar_drift"] = RewardTermCfg(
        func=microduck_mdp.ballet_planar_drift,
        weight=-0.5,
        params={
            "command_name": "twist",
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Discourage violent vertical shocks while entering the one-leg pose.
    cfg.rewards["gentle_transition"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.002,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # angular_momentum penalises all three axes and would directly fight yaw.
    cfg.rewards.pop("angular_momentum", None)

    # First-difference smoothing does not directly suppress rapid alternating
    # corrections.  Add a small second-difference cost for the ten leg actions;
    # its weight is introduced only after the turn has begun to form.
    cfg.rewards["leg_action_acceleration"] = RewardTermCfg(
        func=microduck_mdp.leg_action_acceleration_l2,
        weight=0.0,
    )

    # Start with BallKick's permissive stability weights, then tighten posture
    # after the policy has discovered single support and yaw motion.
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["body_ang_vel"].weight = -0.05

    # A4 gives deterministic single support substantially longer to settle.
    # Crucially, the observed command and both reward terms change together.
    cfg.curriculum["turn_command"] = CurriculumTermCfg(
        func=microduck_mdp.ballet_turn_command_curriculum,
        params={
            "command_name": "twist",
            "turn_stages": [
                {"step": 0, "turn": 0.0},
                {"step": TURN_START_ITER * 24, "turn": 0.2},
                {"step": TURN_FULL_ITER * 24, "turn": TURN_RATE},
            ],
        },
    )
    cfg.curriculum["turn_rate_track_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "turn_rate_track",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": TURN_START_ITER * 24, "weight": 3.0},
                {"step": TURN_FULL_ITER * 24, "weight": 6.0},
            ],
        },
    )
    cfg.curriculum["turn_rate_l1_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "turn_rate_l1",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": TURN_START_ITER * 24, "weight": 0.5},
                {"step": 1800 * 24, "weight": 0.25},
            ],
        },
    )

    cfg.curriculum["upright_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "upright",
            "weight_stages": [
                {"step": 0, "weight": 2.0},
                {"step": 1000 * 24, "weight": 3.0},
                {"step": TURN_FULL_ITER * 24, "weight": 4.0},
            ],
        },
    )
    cfg.curriculum["body_ang_vel_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "body_ang_vel",
            "weight_stages": [
                {"step": 0, "weight": -0.05},
                {"step": 1000 * 24, "weight": -0.10},
                {"step": TURN_FULL_ITER * 24, "weight": -0.20},
            ],
        },
    )
    cfg.curriculum["leg_action_acceleration_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "leg_action_acceleration",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 1400 * 24, "weight": -0.02},
                {"step": 1800 * 24, "weight": -0.05},
            ],
        },
    )

    # Repeated balance corrections and yaw injection need some motion.  Keep
    # smoothing, but do not inherit BallKick's strong final -1.0 blocker.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 1000 * 24, "weight": -0.2},
                {"step": 1400 * 24, "weight": -0.35},
                {"step": 1800 * 24, "weight": -0.5},
            ],
        },
    )

    # BallKick hardens the policy while its kick is already forming.  Ballet's
    # single-support equilibrium is harder to discover, so preserve the small
    # reset-time randomization but postpone stronger randomization and pushes
    # until after the one-leg pose and first turn stage.
    if "com_range" in cfg.curriculum:
        cfg.curriculum["com_range"].params["range_stages"] = [
            {"step": 0, "range": 0.003},
            {"step": 1600 * 24, "range": 0.005},
            {"step": 2000 * 24, "range": 0.01},
            {"step": 2600 * 24, "range": 0.015},
        ]
    if "head_com_range" in cfg.curriculum:
        cfg.curriculum["head_com_range"].params["range_stages"] = [
            {"step": 0, "range": 0.003},
            {"step": 1800 * 24, "range": 0.005},
            {"step": 2400 * 24, "range": 0.01},
        ]
    if "push_magnitude" in cfg.curriculum:
        cfg.curriculum["push_magnitude"].params["push_stages"] = [
            {
                "step": 0,
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
            },
            {
                "step": 1800 * 24,
                "velocity_range": {
                    "x": (-0.08, 0.08),
                    "y": (-0.08, 0.08),
                },
            },
            {
                "step": 2600 * 24,
                "velocity_range": {
                    "x": VELOCITY_PUSH_RANGE,
                    "y": VELOCITY_PUSH_RANGE,
                },
            },
        ]

    return cfg


MicroduckBalletRlCfg = deepcopy(MicroduckBallKickRlCfg)
MicroduckBalletRlCfg.experiment_name = "ballet_right_support_pirouette_a4_joint_roles"
MicroduckBalletRlCfg.run_name = "ballet_right_support_pirouette_a4_joint_roles"
MicroduckBalletRlCfg.max_iterations = 6_000
MicroduckBalletRlCfg.algorithm.symmetry_cfg = None
