"""Tests for the tracking kernel and the reward config read from env cfgs."""

from __future__ import annotations

import torch

from supermimic.tasks.env_cfgs import (
  RT4CAM_ACTION_RATE_WEIGHT,
  RT4CAM_FAR_RANGE,
  RT4CAM_FAR_WEIGHT,
  RT4CAM_ROOT_HEIGHT_SCALE,
  gencoord_env_cfg,
  rt4cam_env_cfg,
)
from supermimic.utils.tracking_reward import (
  UPSTREAM_REWARD_CFG,
  KernelCfg,
  TrackingRewardCfg,
  tracking_kernel,
)

STD = 0.4


def _errors() -> torch.Tensor:
  return torch.linspace(0.0, 8.0 * STD, 400)


def test_without_far_field_the_kernel_is_the_gaussian():
  err_sq = _errors() ** 2
  torch.testing.assert_close(tracking_kernel(err_sq, STD), torch.exp(-err_sq / STD**2))


def test_far_field_kernel_is_bounded_and_monotone():
  kernel = tracking_kernel(_errors() ** 2, STD, far_weight=0.3, far_range=4.0)
  assert kernel[0].item() == 1.0
  assert kernel.min().item() >= 0.0
  assert bool((kernel[1:] <= kernel[:-1] + 1e-7).all())
  # Past far_range stds the ramp has reached its floor: nothing left but the
  # (vanished) Gaussian, so the step reward never goes negative.
  assert kernel[-1].item() < 1e-6


def test_far_field_keeps_a_bounded_pull_where_the_gaussian_has_none():
  def slope(far_weight: float, err: float) -> float:
    e = torch.tensor(err, requires_grad=True)
    tracking_kernel(e**2, STD, far_weight=far_weight, far_range=4.0).backward()
    return e.grad.item()

  far = 3.0 * STD
  assert abs(slope(0.0, far)) < 1e-3 / STD
  assert slope(0.3, far) < -0.05 / STD
  # In the linear zone the far-field slope is constant: 0.3 / (3.5 * std).
  for err in (1.5 * STD, 2.5 * STD, 3.5 * STD):
    ramp = slope(0.3, err) - 0.7 * slope(0.0, err) / 1.0
    assert abs(ramp + 0.3 / (3.5 * STD)) < 1e-4
  # At zero error the gradient is finite (and zero).
  assert abs(slope(0.3, 0.0)) < 1e-6


def test_upstream_env_cfg_reproduces_the_upstream_reward_constants():
  cfg = TrackingRewardCfg.from_env_rewards(gencoord_env_cfg().rewards)
  assert cfg == UPSTREAM_REWARD_CFG
  assert cfg.root_pos == KernelCfg(0.5, 0.3)
  assert cfg.action_rate_weight == -0.1
  assert cfg.joint_limit_weight == -2.0


def test_rt4cam_env_cfg_reward_and_events():
  env_cfg = rt4cam_env_cfg()
  cfg = TrackingRewardCfg.from_env_rewards(env_cfg.rewards)
  for name in ("root_pos", "root_ori", "body_pos", "body_ori"):
    kernel = getattr(cfg, name)
    upstream = getattr(UPSTREAM_REWARD_CFG, name)
    assert (kernel.weight, kernel.std) == (upstream.weight, upstream.std)
    assert kernel.far_weight == RT4CAM_FAR_WEIGHT
    assert kernel.far_range == RT4CAM_FAR_RANGE
  # Velocity terms stay Gaussian.
  assert cfg.body_lin_vel == UPSTREAM_REWARD_CFG.body_lin_vel
  assert cfg.body_ang_vel == UPSTREAM_REWARD_CFG.body_ang_vel
  assert cfg.root_height_scale == RT4CAM_ROOT_HEIGHT_SCALE
  assert cfg.action_rate_weight == RT4CAM_ACTION_RATE_WEIGHT
  # The positive weights (the reward ceiling the critic is clamped to) are unchanged.
  positive = sum(t.weight for t in env_cfg.rewards.values() if t.weight > 0)
  upstream = sum(t.weight for t in gencoord_env_cfg().rewards.values() if t.weight > 0)
  assert positive == upstream

  assert set(env_cfg.events) == {
    "foot_friction",
    "base_com",
    "encoder_bias",
    "push_robot",
  }
  assert "push_robot" not in rt4cam_env_cfg(play=True).events
  assert "push_robot" not in gencoord_env_cfg().events
