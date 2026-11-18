"""Environment configurations for SuperMimic representations."""

from __future__ import annotations

from typing import cast

import mjlab.envs.mdp as mdp
import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.utils.lab_api.math import quat_error_magnitude
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from supermimic.representations import (
  PROPRIO_NOISE_HALF_WIDTHS,
  assemble_anchor_window,
  motion_anchor_window,
  motion_goal_window,
)
from supermimic.utils.tracking_reward import tracking_kernel

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _make_asset_obs_fn(attr_name: str, *, subtract_origins: bool = False):
  """Factory for observation terms that read a single asset data attribute."""

  def fn(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
  ) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    val = getattr(asset.data, attr_name)
    return val - env.scene.env_origins if subtract_origins else val

  return fn


root_pos = _make_asset_obs_fn("root_link_pos_w", subtract_origins=True)
root_ori = _make_asset_obs_fn("root_link_quat_w")
root_lin_vel = _make_asset_obs_fn("root_link_lin_vel_w")
root_ang_vel = _make_asset_obs_fn("root_link_ang_vel_w")
joint_pos = _make_asset_obs_fn("joint_pos")
joint_vel = _make_asset_obs_fn("joint_vel")


# ---------------------------------------------------------------------------
# Maximal coordinate observation terms
# ---------------------------------------------------------------------------


def _make_body_obs_fn(attr_name: str, *, subtract_origins: bool = False):
  """Factory for observation terms that read a MotionCommand body attribute."""

  def fn(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    val = getattr(command, attr_name)
    if subtract_origins:
      val = val - env.scene.env_origins.unsqueeze(1)
    return val.reshape(env.num_envs, -1)

  return fn


body_pos_w = _make_body_obs_fn("robot_body_pos_w", subtract_origins=True)
body_quat_w = _make_body_obs_fn("robot_body_quat_w")
body_lin_vel_w = _make_body_obs_fn("robot_body_lin_vel_w")
body_ang_vel_w = _make_body_obs_fn("robot_body_ang_vel_w")


def motion_command_window(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  """Goal command: joint targets over a future window of reference frames."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return motion_goal_window(command.motion, command.time_steps)


def _motion_anchor_rel(
  env: ManagerBasedRlEnv, command_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
  """Windowed motion anchor as ``(rel_pos, rel_ori)`` in the robot root frame.

  ``env_origins`` is added so the motion anchor shares ``robot_anchor_pos_w``'s
  world frame before the relative transform.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  target_pos, target_quat = motion_anchor_window(command.motion, command.time_steps)
  target_pos = target_pos + env.scene.env_origins[:, None, :]
  return assemble_anchor_window(
    command.robot_anchor_pos_w, command.robot_anchor_quat_w, target_pos, target_quat
  )


def motion_anchor_pos_b(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  """Windowed anchor translation relative to the robot root (world-model only)."""
  return _motion_anchor_rel(env, command_name)[0]


def motion_anchor_ori_b(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  """Windowed anchor orientation (rot6d) relative to the robot root."""
  return _motion_anchor_rel(env, command_name)[1]


# ---------------------------------------------------------------------------
# Configuration factories
# ---------------------------------------------------------------------------


def _add_term(terms: dict[str, ObservationTermCfg], name: str, func, **params) -> None:
  """Insert a proprio term keyed by ``name``; noise pulled from the shared spec."""
  half = PROPRIO_NOISE_HALF_WIDTHS[name]
  kwargs: dict = {"func": func}
  if params:
    kwargs["params"] = params
  if half > 0:
    kwargs["noise"] = Unoise(n_min=-half, n_max=half)
  terms[name] = ObservationTermCfg(**kwargs)


def _sensor_actor_group() -> ObservationGroupCfg:
  """Sensor-only proprio for the actor — what the deployed robot reads.

  Layout: ``[projected_gravity(3), base_ang_vel(3), joint_pos_rel(N),
  joint_vel(N), actions(A)]``. Motion targets live in the ``goal`` group.
  ``enable_corruption=False``: noise is owned by
  ``representations.proprio_noise_bounds`` so train and BPTT rollout
  share one noise spec.
  """
  terms: dict[str, ObservationTermCfg] = {}
  _add_term(terms, "projected_gravity", mdp.projected_gravity)
  _add_term(terms, "base_ang_vel", mdp.builtin_sensor, sensor_name="robot/imu_ang_vel")
  _add_term(terms, "joint_pos", mdp.joint_pos_rel, biased=True)
  _add_term(terms, "joint_vel", mdp.joint_vel_rel)
  _add_term(terms, "actions", mdp.last_action)
  return ObservationGroupCfg(
    terms=terms, concatenate_terms=True, enable_corruption=False
  )


def _goal_obs_group() -> ObservationGroupCfg:
  """Goal group, layout ``[command, rel_pos, rel_ori]`` matching wm_goal.

  ``motion_anchor_pos_b`` (rel_pos) is world-model only; the deployed actor drops
  it by name (see the exporter), so pos and ori stay separate terms.
  """
  return ObservationGroupCfg(
    terms={
      "command": ObservationTermCfg(func=motion_command_window),
      "motion_anchor_pos_b": ObservationTermCfg(func=motion_anchor_pos_b),
      "motion_anchor_ori_b": ObservationTermCfg(func=motion_anchor_ori_b),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )


def gencoord_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create generalized-coordinate world model environment configuration based on G1 tracking."""
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)

  cfg.commands["motion"].anchor_body_name = "pelvis"

  cfg.observations["actor"] = _sensor_actor_group()

  # Add full state observation group
  cfg.observations["full_state"] = ObservationGroupCfg(
    terms={
      "root_pos": ObservationTermCfg(func=root_pos),
      "root_ori": ObservationTermCfg(func=root_ori),
      "joint_pos": ObservationTermCfg(func=joint_pos),
      "root_lin_vel": ObservationTermCfg(func=root_lin_vel),
      "root_ang_vel": ObservationTermCfg(func=root_ang_vel),
      "joint_vel": ObservationTermCfg(func=joint_vel),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  # Add goal observation group
  cfg.observations["goal"] = _goal_obs_group()

  cfg.events = {
    name: term
    for name, term in cfg.events.items()
    if name in {"foot_friction", "base_com", "encoder_bias"}
  }

  cfg.rewards.pop("self_collisions", None)
  cfg.rewards["joint_limit"].weight = -2.0
  # cfg.rewards.pop("motion_global_root_pos", None)
  # cfg.rewards.pop("motion_global_root_ori", None)

  cfg.terminations.pop("ee_body_pos", None)

  return cfg


def maxcoord_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create maximal-coordinate world model environment configuration.

  Extends the base gencoord_env_cfg by replacing the full_state and goal
  observation groups with per-body maximal coordinate versions.
  """
  cfg = gencoord_env_cfg(play=play)

  # Override full state: per-body maximal coordinates
  # Layout: [body_pos(B*3), body_quat(B*4), body_lin_vel(B*3), body_ang_vel(B*3)]
  cfg.observations["full_state"] = ObservationGroupCfg(
    terms={
      "body_pos_w": ObservationTermCfg(
        func=body_pos_w,
        params={"command_name": "motion"},
      ),
      "body_quat_w": ObservationTermCfg(
        func=body_quat_w,
        params={"command_name": "motion"},
      ),
      "body_lin_vel_w": ObservationTermCfg(
        func=body_lin_vel_w,
        params={"command_name": "motion"},
      ),
      "body_ang_vel_w": ObservationTermCfg(
        func=body_ang_vel_w,
        params={"command_name": "motion"},
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  cfg.observations["goal"] = _goal_obs_group()

  return cfg


# ---------------------------------------------------------------------------
# Real-time causal tracking (rt_4cam)
# ---------------------------------------------------------------------------
# Simulator-side twins of the four pose-tracking terms with the far-field kernel
# of ``tracking_kernel``. ``TrackingRewardCfg.from_env_rewards`` reads their
# params, so the world-model reward scores the same function.


def motion_global_root_position_error_kernel(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  far_weight: float,
  far_range: float,
  height_scale: float,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  err = torch.square(command.anchor_pos_w - command.robot_anchor_pos_w)
  err_sq = err[:, 0] + err[:, 1] + height_scale * err[:, 2]
  return tracking_kernel(err_sq, std, far_weight, far_range)


def motion_global_root_orientation_error_kernel(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  far_weight: float,
  far_range: float,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  err_sq = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return tracking_kernel(err_sq, std, far_weight, far_range)


def motion_relative_body_position_error_kernel(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  far_weight: float,
  far_range: float,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  err_sq = torch.sum(
    torch.square(command.body_pos_relative_w - command.robot_body_pos_w), dim=-1
  )
  return tracking_kernel(err_sq.mean(-1), std, far_weight, far_range)


def motion_relative_body_orientation_error_kernel(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  far_weight: float,
  far_range: float,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  err_sq = (
    quat_error_magnitude(command.body_quat_relative_w, command.robot_body_quat_w) ** 2
  )
  return tracking_kernel(err_sq.mean(-1), std, far_weight, far_range)


# Hypotheses of the rt_4cam design, to be confirmed or refuted in closed loop.
RT4CAM_FAR_WEIGHT = 0.3
RT4CAM_FAR_RANGE = 4.0
RT4CAM_ROOT_HEIGHT_SCALE = 0.25
RT4CAM_ACTION_RATE_WEIGHT = -0.5
RT4CAM_PUSH_INTERVAL_S = (3.0, 5.0)


def rt4cam_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Gencoord tracking env for the real-time causal tracker.

  Differences from ``gencoord_env_cfg``:
  - pose-tracking rewards use the far-field kernel, and the root height error is
    down-weighted;
  - the action-rate penalty is stronger;
  - the base task's velocity pushes are kept (training only), less often.

  Foot friction, torso CoM and encoder bias randomization are inherited. Actuator
  delay is not randomized: the world model reads only the current action, so a
  delayed action would be an unobserved input to it.
  """
  cfg = gencoord_env_cfg(play=play)
  far = {"far_weight": RT4CAM_FAR_WEIGHT, "far_range": RT4CAM_FAR_RANGE}

  def pose_term(name: str, func, **extra) -> None:
    old = cfg.rewards[name]
    params = {"command_name": "motion", "std": old.params["std"], **far, **extra}
    cfg.rewards[name] = RewardTermCfg(func=func, weight=old.weight, params=params)

  pose_term(
    "motion_global_root_pos",
    motion_global_root_position_error_kernel,
    height_scale=RT4CAM_ROOT_HEIGHT_SCALE,
  )
  pose_term("motion_global_root_ori", motion_global_root_orientation_error_kernel)
  pose_term("motion_body_pos", motion_relative_body_position_error_kernel)
  pose_term("motion_body_ori", motion_relative_body_orientation_error_kernel)
  cfg.rewards["action_rate_l2"].weight = RT4CAM_ACTION_RATE_WEIGHT

  if not play:
    push = unitree_g1_flat_tracking_env_cfg(play=False).events["push_robot"]
    push.interval_range_s = RT4CAM_PUSH_INTERVAL_S
    cfg.events["push_robot"] = push

  return cfg
