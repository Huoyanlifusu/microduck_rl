from copy import deepcopy
from types import SimpleNamespace

import torch

from mjlab_microduck.robot.microduck_constants import MICRODUCK_BACKLASH_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.backlash import make_backlash_variant
from mjlab_microduck.tasks.microduck_ballet_env_cfg import (
    ACTIVE_PROB,
    COMMAND_DWELL_S,
    FREE_LEG_SIDE,
    FREE_SENSOR,
    SUPPORT_SENSOR,
    MicroduckBalletRlCfg,
    make_microduck_ballet_env_cfg,
)


def test_ballet_command_matches_the_runtime_skill_contract():
    cfg = make_microduck_ballet_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.BalletCommandCfg)
    assert cmd.active_prob == ACTIVE_PROB
    assert cmd.free_leg_side == FREE_LEG_SIDE == 1.0
    assert cmd.turn == 0.0
    assert cmd.resampling_time_range == COMMAND_DWELL_S


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


def test_ballet_reward_stack_requires_a_valid_hop_and_safe_unwind():
    cfg = make_microduck_ballet_env_cfg()
    rewards = cfg.rewards
    for name in (
        "unique_support",
        "free_foot_height",
        "free_leg_pose",
        "hop_height",
        "hop_cycle",
        "idle_pose",
        "idle_height",
        "gentle_landing",
    ):
        assert name in rewards
    assert rewards["hop_cycle"].params["support_sensor"] == SUPPORT_SENSOR
    assert rewards["hop_cycle"].params["free_sensor"] == FREE_SENSOR
    # trunk_vertical_accel_penalty is self-negating: positive weight is the
    # only sign that makes shocks costly.
    assert rewards["gentle_landing"].weight > 0.0
    assert rewards["idle_pose"].params["when_active"] is False


def test_hop_curriculum_starts_after_single_leg_balance():
    cfg = make_microduck_ballet_env_cfg()
    height = cfg.curriculum["hop_height_weight"].params["weight_stages"]
    cycle = cfg.curriculum["hop_cycle_weight"].params["weight_stages"]
    assert height[0]["weight"] == 0.0
    assert cycle[0]["weight"] == 0.0
    assert height[-1]["weight"] > 0.0
    assert cycle[-1]["weight"] > height[-1]["weight"]


def test_ballet_is_asymmetric_and_has_a_backlash_build():
    assert MicroduckBalletRlCfg.algorithm.symmetry_cfg is None
    base = make_microduck_ballet_env_cfg()
    backlash = make_backlash_variant(deepcopy(base), MICRODUCK_BACKLASH_ROBOT_CFG)
    for group in ("actor", "critic"):
        terms = backlash.observations[group].terms
        assert terms["joint_pos"].func is microduck_mdp.joint_pos_rel_backlash
        assert terms["joint_vel"].func is microduck_mdp.joint_vel_rel_backlash


def test_hop_cycle_pays_once_for_same_support_foot_landing():
    command = torch.tensor([[1.0, 1.0, 0.0]])
    support = SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[1.0]])))
    free = SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[0.0]])))
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        step_dt=0.02,
        episode_length_buf=torch.tensor([2]),
        command_manager=SimpleNamespace(get_command=lambda _: command),
        scene=SimpleNamespace(sensors={SUPPORT_SENSOR: support, FREE_SENSOR: free}),
    )

    def reward():
        return microduck_mdp.ballet_hop_cycle(
            env,
            support_sensor=SUPPORT_SENSOR,
            free_sensor=FREE_SENSOR,
            min_air_time=0.04,
        )

    assert reward().item() == 0.0  # arm from unique support
    support.data.found.zero_()
    assert reward().item() == 0.0  # first 20 ms of flight
    assert reward().item() == 0.0  # valid 40 ms flight, still airborne
    support.data.found.fill_(1.0)
    assert reward().item() == 1.0  # land on the same support foot
    assert reward().item() == 0.0  # no repeated reward while planted

    support.data.found.zero_()
    reward()
    reward()
    free.data.found.fill_(1.0)
    assert reward().item() == 0.0  # free-foot landing invalidates the attempt
