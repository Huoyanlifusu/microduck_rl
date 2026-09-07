"""Microduck Ballet V1 — commanded, one-legged pirouette.

This is deliberately a *perpetual skill with an unwind*, not a clocked dance:

    twist = [active, free_leg_side, turn]

V1 fixes ``free_leg_side=+1`` (left leg lifted, right leg supporting) and a
positive yaw rate.  The active flag flips during training.  That teaches the
same network to enter the one-legged turn and, critically, to brake, lower the
free leg and return to a two-foot HOME stand when robotd starts its unwind at
an arbitrary rotation phase.

The environment is derived from BallKick rather than rebuilt from mjlab's
base.  BallKick is the closest proven sim2real recipe: full ground-contact
robot, a named support foot, unified 61D observations, standing starts, BAM,
encoder/IMU noise and delayed push/CoM curricula.  The ball entity and all
ball-specific terms are removed below.

V1 is intentionally conservative: one fixed support side and a moderate fixed
yaw rate.  It is a training scaffold, not a claim that the untrained policy is
hardware safe.  Add side conditioning, reverse turns and faster rotation only
after single-support balance and arbitrary-phase unwind converge.
"""

from copy import deepcopy

from mjlab.managers import CurriculumTermCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_ball_kick_env_cfg import (
    MicroduckBallKickRlCfg,
    make_microduck_ball_kick_env_cfg,
)

EPISODE_LENGTH_S = 12.0
COMMAND_DWELL_S = (3.0, 6.0)
ACTIVE_PROB = 0.65
TURN_RATE = 0.8

# Left leg is the free/display leg; the right leg is the support/pivot leg.
FREE_LEG_SIDE = 1.0
FREE_FOOT_SITE = "left_foot"
SUPPORT_SENSOR = "support_foot_ground_contact"
FREE_SENSOR = "free_foot_ground_contact"

STAND_Z = 0.115
FREE_FOOT_Z = 0.055

_ALL_JOINTS = list(range(14))
_FREE_LEG_JOINTS = [0, 1, 2, 3, 4]

# A readable lifted-leg silhouette, not a human anatomical ballet pose.  The
# support leg is deliberately omitted so it remains free to balance and pivot.
FREE_LEG_TARGET = {
    0: 0.10,   # hip yaw: open the free leg slightly
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

    # Ballet command: binary active flag, fixed free-leg side and fixed turn.
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
            "active_prob": ACTIVE_PROB,
            "free_leg_side": FREE_LEG_SIDE,
            "turn": TURN_RATE,
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
        "pose_stand_neck",
        "height_stand",
    ):
        cfg.rewards.pop(name, None)

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
    cfg.rewards["contact_violation"] = RewardTermCfg(
        func=microduck_mdp.ballet_contact_violation,
        weight=-1.0,
        params={
            "support_sensor": SUPPORT_SENSOR,
            "free_sensor": FREE_SENSOR,
            "command_name": "twist",
        },
    )
    cfg.rewards["free_foot_height"] = RewardTermCfg(
        func=microduck_mdp.ballet_free_foot_height,
        weight=1.5,
        params={
            "target_height": FREE_FOOT_Z,
            "std": 0.025,
            "command_name": "twist",
            "asset_cfg": SceneEntityCfg("robot", site_names=(FREE_FOOT_SITE,)),
        },
    )
    cfg.rewards["free_leg_pose"] = RewardTermCfg(
        func=microduck_mdp.ballet_commanded_pose,
        weight=1.5,
        params={
            "when_active": True,
            "command_name": "twist",
            "std": 0.35,
            "joint_indices": _FREE_LEG_JOINTS,
            "target_overrides": FREE_LEG_TARGET,
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
            "std": 0.45,
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

    # Unwind targets.  The flag is resampled mid-episode, so these train safe
    # recovery from arbitrary rotation phases rather than only at reset.
    cfg.rewards["idle_pose"] = RewardTermCfg(
        func=microduck_mdp.ballet_commanded_pose,
        weight=4.0,
        params={
            "when_active": False,
            "command_name": "twist",
            "std": 0.30,
            "joint_indices": _ALL_JOINTS,
            "target_overrides": None,
        },
    )
    cfg.rewards["idle_height"] = RewardTermCfg(
        func=microduck_mdp.ballet_commanded_height,
        weight=2.0,
        params={
            "target_height": STAND_Z,
            "std": 0.025,
            "when_active": False,
            "command_name": "twist",
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Smooth the free-leg lowering and the return to two-foot HOME.  The helper
    # is self-negating, hence the positive weight.
    cfg.rewards["gentle_transition"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.002,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # angular_momentum penalises all three axes and would directly fight yaw.
    cfg.rewards.pop("angular_momentum", None)

    # First learn right-foot balance and the display pose, then introduce a
    # moderate turn.  The target remains continuous after stage 600; active=0
    # samples train braking and the HOME unwind at arbitrary rotation phases.
    cfg.curriculum["turn_rate_track_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "turn_rate_track",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 300 * 24, "weight": 3.0},
                {"step": 600 * 24, "weight": 6.0},
            ],
        },
    )
    cfg.curriculum["turn_rate_l1_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "turn_rate_l1",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 300 * 24, "weight": 0.5},
                {"step": 1000 * 24, "weight": 0.25},
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
                {"step": 500 * 24, "weight": -0.2},
                {"step": 1000 * 24, "weight": -0.35},
                {"step": 1500 * 24, "weight": -0.5},
            ],
        },
    )

    return cfg


MicroduckBalletRlCfg = deepcopy(MicroduckBallKickRlCfg)
MicroduckBalletRlCfg.experiment_name = "ballet_right_support_pirouette_v1"
MicroduckBalletRlCfg.run_name = "ballet_right_support_pirouette_v1"
MicroduckBalletRlCfg.max_iterations = 6_000
MicroduckBalletRlCfg.algorithm.symmetry_cfg = None
