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
    SUPPORT_FOOT_SITE,
    TURN_RATE,
    MicroduckBalletRlCfg,
    make_microduck_ballet_env_cfg,
)


def test_ballet_command_matches_the_runtime_skill_contract():
    cfg = make_microduck_ballet_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.BalletCommandCfg)
    assert cmd.active == 1.0
    assert cmd.free_leg_side == FREE_LEG_SIDE == 1.0
    assert cmd.turn == TURN_RATE == 0.4
    assert cmd.resampling_time_range == COMMAND_DWELL_S == (12.0, 12.0)


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


def test_ballet_reward_stack_prioritizes_weight_transfer_and_one_leg_hold():
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
        "gentle_transition",
        "leg_action_acceleration",
        "pose_stand_neck",
        "height_stand",
        "height_stand_l1",
        "upright_linear",
        "com_over_support",
        "com_over_support_l1",
        "free_leg_pose_l1",
    ):
        assert name in rewards
    # trunk_vertical_accel_penalty is self-negating: positive weight is the
    # only sign that makes shocks costly.
    assert rewards["gentle_transition"].weight > 0.0
    assert "idle_pose" not in rewards
    assert "idle_height" not in rewards
    assert rewards["turn_rate_track"].params["upright_std"] == 0.25
    assert rewards["turn_rate_track"].params["neck_std"] == 0.20
    assert rewards["pose_stand_neck"].weight == 1.5
    assert rewards["pose_stand_neck"].params["joint_indices"] == [5, 6, 7, 8]
    assert rewards["height_stand"].weight == 2.0
    assert rewards["height_stand"].params["std"] == 0.04
    assert rewards["height_stand_l1"].weight == 20.0
    assert rewards["upright_linear"].weight == 2.0
    assert rewards["com_over_support"].weight == 3.0
    assert rewards["com_over_support"].params["std"] == 0.04
    assert rewards["com_over_support"].params["asset_cfg"].site_names == (
        SUPPORT_FOOT_SITE,
    )
    assert rewards["com_over_support_l1"].weight == 10.0
    assert rewards["free_foot_height"].weight == 2.5
    assert rewards["free_leg_pose"].weight == 2.5
    assert rewards["free_leg_pose_l1"].weight == 1.0
    assert rewards["contact_violation"].weight == -2.0
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
    assert track[1]["step"] == 800 * 24
    assert bootstrap[1]["step"] == 800 * 24
    assert action_rate[1]["step"] == 800 * 24
    assert track[-1]["weight"] == 6.0
    assert bootstrap[-1]["weight"] == 0.25
    assert action_rate[-1]["weight"] == -0.5
    assert upright[-1]["weight"] == 4.0
    assert body_ang_vel[-1]["weight"] == -0.2
    assert action_acc[-1]["weight"] == -0.05


def test_com_over_support_supplies_dense_weight_transfer_direction():
    command = torch.tensor([[1.0, 1.0, TURN_RATE]])
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_com_pos_w=torch.tensor([[0.0, 0.0, 0.14]]),
            site_pos_w=torch.tensor([[[0.0, 0.0, 0.01]]]),
        )
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_command=lambda _: command),
        scene={"robot": robot},
    )
    asset_cfg = SimpleNamespace(name="robot", site_ids=[0])

    centered = microduck_mdp.ballet_com_over_support(env, asset_cfg=asset_cfg)
    assert centered.item() == 1.0
    assert microduck_mdp.ballet_com_over_support_l1(
        env, asset_cfg=asset_cfg
    ).item() == 0.0

    robot.data.root_com_pos_w[:, 1] = 0.04
    offset = microduck_mdp.ballet_com_over_support(
        env, std=0.04, asset_cfg=asset_cfg
    )
    assert torch.isclose(offset, torch.exp(torch.tensor([-1.0]))).all()
    offset_l1 = microduck_mdp.ballet_com_over_support_l1(
        env, asset_cfg=asset_cfg
    )
    assert torch.isclose(offset_l1, torch.tensor([-0.04])).all()


def test_ballet_is_asymmetric_and_has_a_backlash_build():
    assert MicroduckBalletRlCfg.algorithm.symmetry_cfg is None
    base = make_microduck_ballet_env_cfg()
    backlash = make_backlash_variant(deepcopy(base), MICRODUCK_BACKLASH_ROBOT_CFG)
    for group in ("actor", "critic"):
        terms = backlash.observations[group].terms
        assert terms["joint_pos"].func is microduck_mdp.joint_pos_rel_backlash
        assert terms["joint_vel"].func is microduck_mdp.joint_vel_rel_backlash


def test_turn_rate_tracks_active_command_and_retains_zero_target_fallback():
    command = torch.tensor([[1.0, 1.0, TURN_RATE]])
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_ang_vel_b=torch.tensor([[0.0, 0.0, TURN_RATE]]),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            joint_pos=torch.zeros(1, 14),
            default_joint_pos=torch.zeros(1, 14),
        ),
        find_joints=lambda _: (list(range(14)), [f"joint_{i}" for i in range(14)]),
    )
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        command_manager=SimpleNamespace(get_command=lambda _: command),
        scene={"robot": robot},
    )

    track = microduck_mdp.ballet_turn_rate_track(
        env, std=0.30, upright_std=0.25, neck_std=0.20
    )
    assert track.item() == 1.0
    assert microduck_mdp.ballet_turn_rate_l1(env).item() == 0.0

    robot.data.joint_pos[:, 5] = 0.4
    bowed_track = microduck_mdp.ballet_turn_rate_track(
        env, std=0.30, upright_std=0.25, neck_std=0.20
    )
    assert bowed_track.item() < 0.4
    robot.data.joint_pos.zero_()

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
