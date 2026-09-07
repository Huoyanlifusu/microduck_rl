from copy import deepcopy
from types import SimpleNamespace

import torch

from mjlab_microduck.robot.microduck_constants import MICRODUCK_BACKLASH_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.backlash import make_backlash_variant
from mjlab_microduck.tasks.microduck_ballet_env_cfg import (
    COMMAND_DWELL_S,
    FREE_LEG_SIDE,
    FREE_SENSOR,
    SUPPORT_SENSOR,
    TURN_RATE,
    MicroduckBalletRlCfg,
    make_microduck_ballet_env_cfg,
)


def test_ballet_command_matches_the_runtime_skill_contract():
    cfg = make_microduck_ballet_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.BalletCommandCfg)
    assert cmd.free_leg_side == FREE_LEG_SIDE == 1.0
    assert cmd.turn == TURN_RATE == 0.4
    assert cmd.resampling_time_range == COMMAND_DWELL_S == (2.0, 2.6)


def test_ballet_keeps_the_unified_actor_observation_layout():
    cfg = make_microduck_ballet_env_cfg()
    terms = cfg.observations["actor"].terms
    assert "base_lin_vel" not in terms
    assert terms["head_command"].params["dim"] == 4
    assert terms["body_command"].params["dim"] == 6
    assert "ball_position" not in terms
    assert "ball_velocity" not in terms


def test_ballet_removes_the_ball_and_wires_both_feet():
    cfg = make_microduck_ballet_env_cfg()
    assert list(cfg.scene.entities) == ["robot"]
    assert "reset_ball" not in cfg.events
    sensor_names = {sensor.name for sensor in cfg.scene.sensors}
    assert SUPPORT_SENSOR in sensor_names
    assert FREE_SENSOR in sensor_names


def test_ballet_reward_stack_requires_one_leg_turn_and_safe_unwind():
    cfg = make_microduck_ballet_env_cfg()
    rewards = cfg.rewards
    for name in (
        "unique_support",
        "contact_violation",
        "free_foot_height",
        "free_leg_pose",
        "turn_rate_track",
        "turn_rate_l1",
        "planar_drift",
        "idle_pose",
        "idle_height",
        "gentle_transition",
        "leg_action_acceleration",
    ):
        assert name in rewards
    # trunk_vertical_accel_penalty is self-negating: positive weight is the
    # only sign that makes shocks costly.
    assert rewards["gentle_transition"].weight > 0.0
    assert rewards["idle_pose"].params["when_active"] is False
    assert rewards["turn_rate_track"].params["upright_std"] == 0.25
    assert "angular_momentum" not in rewards


def test_turn_curriculum_starts_after_single_leg_balance():
    cfg = make_microduck_ballet_env_cfg()
    track = cfg.curriculum["turn_rate_track_weight"].params["weight_stages"]
    bootstrap = cfg.curriculum["turn_rate_l1_weight"].params["weight_stages"]
    action_rate = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    upright = cfg.curriculum["upright_weight"].params["weight_stages"]
    body_ang_vel = cfg.curriculum["body_ang_vel_weight"].params["weight_stages"]
    action_acc = cfg.curriculum["leg_action_acceleration_weight"].params[
        "weight_stages"
    ]
    assert track[0]["weight"] == 0.0
    assert bootstrap[0]["weight"] == 0.0
    assert track[-1]["weight"] == 6.0
    assert bootstrap[-1]["weight"] == 0.25
    assert action_rate[-1]["weight"] == -0.5
    assert upright[-1]["weight"] == 4.0
    assert body_ang_vel[-1]["weight"] == -0.2
    assert action_acc[-1]["weight"] == -0.05


def test_ballet_is_asymmetric_and_has_a_backlash_build():
    assert MicroduckBalletRlCfg.algorithm.symmetry_cfg is None
    base = make_microduck_ballet_env_cfg()
    backlash = make_backlash_variant(deepcopy(base), MICRODUCK_BACKLASH_ROBOT_CFG)
    for group in ("actor", "critic"):
        terms = backlash.observations[group].terms
        assert terms["joint_pos"].func is microduck_mdp.joint_pos_rel_backlash
        assert terms["joint_vel"].func is microduck_mdp.joint_vel_rel_backlash


def test_turn_rate_tracks_command_and_brakes_during_unwind():
    command = torch.tensor([[1.0, 1.0, TURN_RATE]])
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_ang_vel_b=torch.tensor([[0.0, 0.0, TURN_RATE]]),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )
    )
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        command_manager=SimpleNamespace(get_command=lambda _: command),
        scene={"robot": robot},
    )

    track = microduck_mdp.ballet_turn_rate_track(env, std=0.30, upright_std=0.25)
    assert track.item() == 1.0
    assert microduck_mdp.ballet_turn_rate_l1(env).item() == 0.0

    robot.data.root_link_quat_w[:] = torch.tensor([[0.9848, 0.1736, 0.0, 0.0]])
    tilted_track = microduck_mdp.ballet_turn_rate_track(
        env, std=0.30, upright_std=0.25
    )
    assert tilted_track.item() < 0.4
    robot.data.root_link_quat_w[:] = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    command[:, 0] = 0.0
    braking_track = microduck_mdp.ballet_turn_rate_track(
        env, std=0.30, upright_std=0.25
    )
    assert braking_track.item() < 0.2
    robot.data.root_link_ang_vel_b[:, 2] = 0.0
    track = microduck_mdp.ballet_turn_rate_track(env, std=0.30, upright_std=0.25)
    assert track.item() == 1.0


def test_contact_violation_rejects_flight_and_free_foot_touchdown():
    command = torch.tensor([[1.0, 1.0, TURN_RATE]])
    support = SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[1.0]])))
    free = SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[0.0]])))
    sensors = {SUPPORT_SENSOR: support, FREE_SENSOR: free}
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        command_manager=SimpleNamespace(get_command=lambda _: command),
        scene=SimpleNamespace(sensors=sensors),
    )

    assert microduck_mdp.ballet_contact_violation(
        env, SUPPORT_SENSOR, FREE_SENSOR
    ).item() == 0.0
    support.data.found.zero_()
    assert microduck_mdp.ballet_contact_violation(
        env, SUPPORT_SENSOR, FREE_SENSOR
    ).item() == 1.0
    support.data.found.fill_(1.0)
    free.data.found.fill_(1.0)
    assert microduck_mdp.ballet_contact_violation(
        env, SUPPORT_SENSOR, FREE_SENSOR
    ).item() == 1.0
