"""Microduck Ballet V1 — commanded, one-legged low hopping.

This is deliberately a *perpetual skill with an unwind*, not a clocked dance:

    twist = [active, free_leg_side, turn]

V1 fixes ``free_leg_side=+1`` (left leg lifted, right leg supporting) and
``turn=0``.  The active flag flips during training.  That teaches the same
network to enter the one-legged hop and, critically, to land and return to a
two-foot HOME stand when robotd starts its unwind at an arbitrary hop phase.

The environment is derived from BallKick rather than rebuilt from mjlab's
base.  BallKick is the closest proven sim2real recipe: full ground-contact
robot, a named support foot, unified 61D observations, standing starts, BAM,
encoder/IMU noise and delayed push/CoM curricula.  The ball entity and all
ball-specific terms are removed below.

V1 is intentionally conservative: low hops, one fixed side, no yaw turn.  It
is a training scaffold, not a claim that the untrained policy is hardware
safe.  Add side conditioning and turning only after the basic contact cycle
and arbitrary-phase unwind converge.
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

# Left leg is the free/display leg; the right leg is the support/hop leg.
FREE_LEG_SIDE = 1.0
FREE_FOOT_SITE = "left_foot"
SUPPORT_SENSOR = "support_foot_ground_contact"
FREE_SENSOR = "free_foot_ground_contact"

STAND_Z = 0.115
HOP_Z = 0.135
FREE_FOOT_Z = 0.055

_ALL_JOINTS = list(range(14))
_FREE_LEG_JOINTS = [0, 1, 2, 3, 4]

# A readable lifted-leg silhouette, not a human anatomical ballet pose.  The
# support leg is deliberately omitted so it remains free to compress/push off.
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

    # Ballet command: exact binary active flag, fixed free-leg side and no turn.
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
        "pose_stand_neck",
        "height_stand",
    ):
        cfg.rewards.pop(name, None)

    # Dense bootstrap: form the one-leg silhouette before asking for flight.
    cfg.rewards["unique_support"] = RewardTermCfg(
        func=microduck_mdp.ballet_unique_support,
        weight=2.0,
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

    # Hop discovery: a broad height target provides dense gradient; the
    # stateful completion pulse says which contact cycle actually counts.
    cfg.rewards["hop_height"] = RewardTermCfg(
        func=microduck_mdp.ballet_commanded_height,
        weight=0.0,
        params={
            "target_height": HOP_Z,
            "std": 0.020,
            "when_active": True,
            "command_name": "twist",
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["hop_cycle"] = RewardTermCfg(
        func=microduck_mdp.ballet_hop_cycle,
        weight=0.0,
        params={
            "support_sensor": SUPPORT_SENSOR,
            "free_sensor": FREE_SENSOR,
            "command_name": "twist",
            "min_air_time": 0.04,
            "max_air_time": 0.40,
        },
    )

    # Unwind targets.  The flag is resampled mid-episode, so these train safe
    # recovery from arbitrary points in the cycle rather than only at reset.
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

    # Style/sim2real pressure.  This function is self-negating, hence the
    # positive weight.  Keep it small: takeoff and landing necessarily have az.
    cfg.rewards["gentle_landing"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.002,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # First learn weight shift/lift, then ask for low flight and finally make a
    # valid same-foot landing economically important.
    cfg.curriculum["hop_height_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "hop_height",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 300 * 24, "weight": 1.0},
                {"step": 600 * 24, "weight": 2.0},
            ],
        },
    )
    cfg.curriculum["hop_cycle_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "hop_cycle",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 600 * 24, "weight": 5.0},
                {"step": 1000 * 24, "weight": 10.0},
            ],
        },
    )

    return cfg


MicroduckBalletRlCfg = deepcopy(MicroduckBallKickRlCfg)
MicroduckBalletRlCfg.experiment_name = "ballet_right_support_v1"
MicroduckBalletRlCfg.run_name = "ballet_right_support_v1"
MicroduckBalletRlCfg.max_iterations = 6_000
MicroduckBalletRlCfg.algorithm.symmetry_cfg = None
