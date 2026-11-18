"""Gaussian-kernel tracking reward on world-frame bodies.

Mirrors ``mjlab.tasks.tracking.tracking_env_cfg.rewards``: each tracking term is
``weight * exp(-err_sq / std**2)``, plus the action-rate and joint-limit
penalties; body pos/ori targets are re-anchored
to the predicted root xy + yaw, matching
``MotionCommand.update_relative_body_poses``.

``TrackingRewardCfg`` carries the weights and kernel shapes. Its defaults are
the upstream constants; ``from_env_rewards`` reads them from the env's reward
terms so the world-model reward and the simulator reward cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_inv, quat_mul, yaw_quat

from supermimic.utils.helpers import rotation_error_magnitude, small_matmul


@dataclass(frozen=True)
class KernelCfg:
  """One tracking term: ``weight * tracking_kernel(err_sq, std, ...)``."""

  weight: float
  std: float
  far_weight: float = 0.0
  far_range: float = 4.0


@dataclass(frozen=True)
class TrackingRewardCfg:
  root_pos: KernelCfg = KernelCfg(0.5, 0.3)
  root_ori: KernelCfg = KernelCfg(0.5, 0.4)
  body_pos: KernelCfg = KernelCfg(1.0, 0.3)
  body_ori: KernelCfg = KernelCfg(1.0, 0.4)
  body_lin_vel: KernelCfg = KernelCfg(1.0, 1.0)
  body_ang_vel: KernelCfg = KernelCfg(1.0, 3.14)
  root_height_scale: float = 1.0
  """Scale on the squared height (z) part of the root position error."""
  action_rate_weight: float = -0.1
  joint_limit_weight: float = -2.0

  @classmethod
  def from_env_rewards(cls, rewards: Mapping[str, Any]) -> TrackingRewardCfg:
    """Read the reward terms of a tracking env cfg. A missing term weighs 0."""

    def kernel(name: str) -> KernelCfg:
      term = rewards.get(name)
      if term is None:
        return KernelCfg(0.0, 1.0)
      params = term.params
      return KernelCfg(
        weight=term.weight,
        std=params["std"],
        far_weight=params.get("far_weight", 0.0),
        far_range=params.get("far_range", 4.0),
      )

    def weight(name: str) -> float:
      term = rewards.get(name)
      return 0.0 if term is None else term.weight

    root_pos = rewards.get("motion_global_root_pos")
    return cls(
      root_pos=kernel("motion_global_root_pos"),
      root_ori=kernel("motion_global_root_ori"),
      body_pos=kernel("motion_body_pos"),
      body_ori=kernel("motion_body_ori"),
      body_lin_vel=kernel("motion_body_lin_vel"),
      body_ang_vel=kernel("motion_body_ang_vel"),
      root_height_scale=(
        1.0 if root_pos is None else root_pos.params.get("height_scale", 1.0)
      ),
      action_rate_weight=weight("action_rate_l2"),
      joint_limit_weight=weight("joint_limit"),
    )


UPSTREAM_REWARD_CFG = TrackingRewardCfg()


def tracking_kernel(
  err_sq: torch.Tensor, std: float, far_weight: float = 0.0, far_range: float = 4.0
) -> torch.Tensor:
  """Tracking kernel in ``[0, 1]`` of a squared error.

  ``far_weight == 0`` is the Gaussian ``exp(-err_sq / std**2)``. Otherwise it is
  blended with a far-field ramp ``max(0, 1 - huber(x) / huber(far_range))``,
  ``x = sqrt(err_sq) / std``: quadratic near zero, then linear, so the pull
  toward the target keeps a constant, bounded slope out to ``far_range`` stds
  where the Gaussian's has vanished. The ramp is floored at zero on purpose: a
  negative step reward would make ending the episode (survival weight -> 0)
  better than staying alive far from the target.
  """
  near = torch.exp(-err_sq / std**2)
  if far_weight == 0.0:
    return near
  scaled_sq = err_sq / std**2
  x = torch.sqrt(scaled_sq + 1e-12)
  huber = torch.where(x < 1.0, 0.5 * scaled_sq, x - 0.5)
  far = (1.0 - huber / (far_range - 0.5)).clamp(min=0.0)
  return (1.0 - far_weight) * near + far_weight * far


def _reanchor_targets(
  target_pos: torch.Tensor,
  target_quat: torch.Tensor,
  pred_root_pos: torch.Tensor,
  pred_root_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Re-anchor target body poses to predicted root xy + yaw.

  Target bodies are expressed in a frame whose xy + yaw come from the
  predicted root and whose z + roll/pitch come from the target root.
  """
  anchor_pos = target_pos[..., 0:1, :]

  delta_pos = torch.cat([pred_root_pos[..., :2], target_pos[..., 0, 2:3]], dim=-1)
  delta_mat = matrix_from_quat(
    yaw_quat(quat_mul(pred_root_quat, quat_inv(target_quat[..., 0, :])))
  ).unsqueeze(-3)

  rel_pos = target_pos - anchor_pos
  tgt_pos = (
    delta_pos.unsqueeze(-2) + small_matmul(delta_mat, rel_pos.unsqueeze(-1))[..., 0]
  )
  tgt_mat = small_matmul(delta_mat, matrix_from_quat(target_quat))
  return tgt_pos, tgt_mat


def tracking_reward(
  aux_state: torch.Tensor,
  pred_pos: torch.Tensor,
  pred_rot_mat: torch.Tensor,
  pred_lin_vel: torch.Tensor,
  pred_ang_vel: torch.Tensor,
  joint_state: torch.Tensor,
  action: torch.Tensor,
  last_action: torch.Tensor,
  target_pos: torch.Tensor,
  target_quat: torch.Tensor,
  target_lin_vel: torch.Tensor,
  target_ang_vel: torch.Tensor,
  soft_joint_pos_limits: torch.Tensor,
  dt: float,
  cfg: TrackingRewardCfg = UPSTREAM_REWARD_CFG,
) -> torch.Tensor:
  """Gaussian-kernel reward matching the motion-tracking env.

  ``pred_*`` are the predicted bodies in the world frame and ``aux_state`` the
  predicted root pose the targets are re-anchored to.
  """
  tgt_pos, tgt_mat = _reanchor_targets(
    target_pos, target_quat, aux_state[..., :3], aux_state[..., 3:7]
  )

  root_err = (pred_pos[..., 0, :] - target_pos[..., 0, :]) ** 2
  root_pos_err = (
    root_err[..., 0] + root_err[..., 1] + cfg.root_height_scale * root_err[..., 2]
  )
  terms: list[tuple[torch.Tensor, KernelCfg]] = [
    (root_pos_err, cfg.root_pos),
    (
      rotation_error_magnitude(
        pred_rot_mat[..., 0, :, :], matrix_from_quat(target_quat[..., 0, :])
      )
      ** 2,
      cfg.root_ori,
    ),
    (torch.sum((pred_pos - tgt_pos) ** 2, dim=-1).mean(-1), cfg.body_pos),
    ((rotation_error_magnitude(pred_rot_mat, tgt_mat) ** 2).mean(-1), cfg.body_ori),
    (
      torch.sum((pred_lin_vel - target_lin_vel) ** 2, dim=-1).mean(-1),
      cfg.body_lin_vel,
    ),
    (
      torch.sum((pred_ang_vel - target_ang_vel) ** 2, dim=-1).mean(-1),
      cfg.body_ang_vel,
    ),
  ]
  reward = sum(
    k.weight * tracking_kernel(err, k.std, k.far_weight, k.far_range)
    for err, k in terms
  )
  action_rate = torch.sum((action - last_action) ** 2, dim=-1)
  joint_pos, _ = joint_state.chunk(2, dim=-1)
  lo, hi = soft_joint_pos_limits.unbind(-1)
  joint_limit = ((lo - joint_pos).clamp(min=0) + (joint_pos - hi).clamp(min=0)).sum(-1)
  return dt * (
    reward + cfg.action_rate_weight * action_rate + cfg.joint_limit_weight * joint_limit
  )
