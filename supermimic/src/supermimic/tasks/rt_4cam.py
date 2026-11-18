"""Real-time causal tracking task slot (``rt_4cam``).

The slot turns the last ``K`` raw single-frame IK poses of a markerless
multi-camera reference stream into the goal one encoder consumes. It never sees
future reference frames or reference velocities.

Contract (shared by training and deployment through :func:`build_causal_goal`):

- Input, one sample per control tick (50 Hz), newest last:
  ``joint_pos`` ``(B, K, J)`` rad in the robot's joint order, ``root_pos``
  ``(B, K, 3)`` m and ``root_quat`` ``(B, K, 4)`` wxyz of the reference pelvis in
  any gravity-aligned z-up world frame, plus the robot's measured
  ``robot_joint_pos`` ``(B, J)`` rad.
- Output: ``(B, K * (J + 7) + J)`` goal, laid out as :func:`goal_layout` returns.
- Only the window's own increments, tilt and height are used, so the absolute
  horizontal position and heading of the reference are deliberately not
  observable: horizontal drift against the reference is not corrected.
- Time base: frame ``k`` is the reference at control tick ``t - (K - 1 - k)``.
  A missing camera frame must be filled by repeating the previous one, which is
  what the training degradation models; an empty window is not a valid input.

Training feeds the encoder a degraded copy of the clean clip
(:class:`Rt4CamStream`) while reward and termination keep scoring the clean
reference in the world-model env.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.tasks.tracking.mdp.commands import MotionLoader

TASK_NAME = "rt_4cam"

# Joint-name prefixes of the limbs an occlusion spike bends as one group.
_LIMB_PREFIXES: tuple[tuple[str, ...], ...] = (
  ("left_hip", "left_knee", "left_ankle"),
  ("right_hip", "right_knee", "right_ankle"),
  ("waist",),
  ("left_shoulder", "left_elbow", "left_wrist"),
  ("right_shoulder", "right_elbow", "right_wrist"),
)

type StreamState = dict[str, torch.Tensor]


@dataclass(frozen=True)
class Rt4CamTaskCfg:
  """Causal reference window and the degradation applied to it in training."""

  history_frames: int = 8
  """Reference frames in the window, newest last (K)."""
  joint_noise_std: float = 0.05
  """Per-frame, per-joint Gaussian noise on reference joint angles (rad)."""
  height_offset: float = 0.05
  """Half-width of the uniform pelvis-height bias held for a whole episode (m)."""
  xy_drift_std: float = 0.01
  """Per-frame std of the random walk added to the reference root xy (m)."""
  hold_prob: float = 0.15
  """Per-frame probability that the frame is dropped and the previous repeats."""
  spike_prob: float = 0.02
  """Per-frame probability that one limb jumps (occlusion glitch)."""
  spike_angle: float = 0.44
  """Largest joint offset of a limb spike (rad, about 25 degrees)."""
  spike_persist_prob: float = 0.5
  """Per-frame probability that an active spike survives to the next frame."""


def goal_dim(history_frames: int, num_joints: int) -> int:
  return history_frames * (num_joints + 7) + num_joints


def goal_layout(history_frames: int, num_joints: int) -> dict[str, tuple[int, int]]:
  """``name -> (start, stop)`` columns of the goal vector."""
  widths = {
    "ref_joint_pos": history_frames * num_joints,
    "ref_projected_gravity": history_frames * 3,
    "ref_root_height": history_frames,
    "ref_local_xy_yaw": history_frames * 3,
    "joint_pos_error": num_joints,
  }
  layout: dict[str, tuple[int, int]] = {}
  offset = 0
  for name, width in widths.items():
    layout[name] = (offset, offset + width)
    offset += width
  return layout


def _yaw(quat: torch.Tensor) -> torch.Tensor:
  """Heading angle of a wxyz quaternion about world z."""
  w, x, y, z = quat.unbind(-1)
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def build_causal_goal(
  joint_pos: torch.Tensor,
  root_pos: torch.Tensor,
  root_quat: torch.Tensor,
  robot_joint_pos: torch.Tensor,
) -> torch.Tensor:
  """Goal from the reference window and the robot's measured joint positions.

  Horizontal position and heading are expressed relative to the window's oldest
  frame, so the goal carries no absolute room coordinates.
  """
  batch = joint_pos.shape[0]
  down = torch.zeros_like(root_pos)
  down[..., 2] = -1.0
  gravity = quat_apply_inverse(root_quat, down)

  yaw = _yaw(root_quat)
  yaw0 = yaw[:, :1]
  delta = root_pos[..., :2] - root_pos[:, :1, :2]
  cos0, sin0 = torch.cos(yaw0), torch.sin(yaw0)
  local_x = cos0 * delta[..., 0] + sin0 * delta[..., 1]
  local_y = -sin0 * delta[..., 0] + cos0 * delta[..., 1]
  dyaw = yaw - yaw0
  dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))
  local = torch.stack([local_x, local_y, dyaw], dim=-1)

  return torch.cat(
    [
      joint_pos.reshape(batch, -1),
      gravity.reshape(batch, -1),
      root_pos[..., 2],
      local.reshape(batch, -1),
      joint_pos[:, -1] - robot_joint_pos,
    ],
    dim=-1,
  )


def limb_masks(joint_names: list[str]) -> torch.Tensor:
  """``(num_limbs, J)`` 0/1 masks grouping joints into legs, waist and arms."""
  masks = torch.zeros(len(_LIMB_PREFIXES), len(joint_names))
  for limb, prefixes in enumerate(_LIMB_PREFIXES):
    for joint, name in enumerate(joint_names):
      if name.startswith(prefixes):
        masks[limb, joint] = 1.0
  if not bool((masks.sum(0) == 1.0).all()):
    raise ValueError(f"joints not covered by exactly one limb: {joint_names}")
  return masks


class Rt4CamStream:
  """Degraded causal reference stream, kept as a per-env sliding window.

  One call per control tick pushes the clip frame at ``motion_times`` through
  the noise processes and into the window, the way raw camera IK frames arrive
  online. The noise is causal and persistent: a frame keeps the corruption it
  arrived with for as long as it stays in the window.

  State (plain dict so it traces under ``torch.compile``):
    ``joint_pos`` ``(B, K, J)``, ``root_pos`` ``(B, K, 3)``, ``root_quat``
    ``(B, K, 4)``: the degraded window, newest last.
    ``height_offset`` ``(B, 1)``, ``xy_drift`` ``(B, 2)``, ``spike`` ``(B, J)``:
    the noise processes.
  """

  def __init__(self, cfg: Rt4CamTaskCfg, joint_names: list[str], device: str) -> None:
    if cfg.history_frames < 1:
      raise ValueError(f"history_frames must be >= 1, got {cfg.history_frames}")
    self.cfg = cfg
    self._limbs = limb_masks(joint_names).to(device)

  def _window_frames(
    self, motion: MotionLoader, motion_times: torch.Tensor
  ) -> torch.Tensor:
    """``(B, K)`` clip frames ending at ``motion_times``, clamped to the start
    of each env's clip so the window never reads the previous clip."""
    ends = motion.motion_ends
    clip = torch.searchsorted(ends, motion_times, right=True).clamp(
      max=ends.numel() - 1
    )
    starts = torch.cat([ends.new_zeros(1), ends[:-1]])[clip]
    back = torch.arange(self.cfg.history_frames - 1, -1, -1, device=motion_times.device)
    return torch.maximum(motion_times[:, None] - back, starts[:, None])

  def _degrade(
    self,
    joint_pos: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    state: StreamState,
    hold_from: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, StreamState]:
    """Advance the noise processes one tick and corrupt one clean frame."""
    cfg = self.cfg
    batch = joint_pos.shape[0]
    device = joint_pos.device

    xy_drift = state["xy_drift"] + (
      torch.randn_like(state["xy_drift"]) * cfg.xy_drift_std
    )

    limb = torch.randint(self._limbs.shape[0], (batch,), device=device)
    direction = torch.randn_like(joint_pos) * self._limbs[limb]
    direction = direction / direction.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    started = torch.rand(batch, 1, device=device) < cfg.spike_prob
    survives = torch.rand(batch, 1, device=device) < cfg.spike_persist_prob
    spike = torch.where(
      started,
      direction * cfg.spike_angle,
      torch.where(survives, state["spike"], torch.zeros_like(joint_pos)),
    )

    noisy_joint_pos = (
      joint_pos + torch.randn_like(joint_pos) * cfg.joint_noise_std + spike
    )
    offset = torch.cat([xy_drift, state["height_offset"]], dim=-1)
    noisy_root_pos = root_pos + offset
    noisy_root_quat = root_quat

    if hold_from is not None:
      held = torch.rand(batch, 1, device=device) < cfg.hold_prob
      noisy_joint_pos = torch.where(held, hold_from[0], noisy_joint_pos)
      noisy_root_pos = torch.where(held, hold_from[1], noisy_root_pos)
      noisy_root_quat = torch.where(held, hold_from[2], noisy_root_quat)

    processes = {
      "height_offset": state["height_offset"],
      "xy_drift": xy_drift,
      "spike": spike,
    }
    return noisy_joint_pos, noisy_root_pos, noisy_root_quat, processes

  def init(self, motion: MotionLoader, motion_times: torch.Tensor) -> StreamState:
    """Fresh window ending at ``motion_times`` with newly drawn noise processes."""
    cfg = self.cfg
    batch = motion_times.shape[0]
    device = motion_times.device
    num_joints = motion.joint_pos.shape[-1]
    frames = self._window_frames(motion, motion_times)

    processes: StreamState = {
      "height_offset": (torch.rand(batch, 1, device=device) * 2 - 1)
      * cfg.height_offset,
      "xy_drift": torch.zeros(batch, 2, device=device),
      "spike": torch.zeros(batch, num_joints, device=device),
    }
    window: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for k in range(cfg.history_frames):
      idx = frames[:, k]
      joint_pos, root_pos, root_quat, processes = self._degrade(
        motion.joint_pos[idx],
        motion.root_pos_w[idx],
        motion.root_quat_w[idx],
        processes,
        window[-1] if window else None,
      )
      window.append((joint_pos, root_pos, root_quat))
    return {
      "joint_pos": torch.stack([f[0] for f in window], dim=1),
      "root_pos": torch.stack([f[1] for f in window], dim=1),
      "root_quat": torch.stack([f[2] for f in window], dim=1),
      **processes,
    }

  def advance(
    self, state: StreamState, motion: MotionLoader, motion_times: torch.Tensor
  ) -> StreamState:
    """Push the clip frame at ``motion_times`` into the window."""
    newest = (
      state["joint_pos"][:, -1],
      state["root_pos"][:, -1],
      state["root_quat"][:, -1],
    )
    joint_pos, root_pos, root_quat, processes = self._degrade(
      motion.joint_pos[motion_times],
      motion.root_pos_w[motion_times],
      motion.root_quat_w[motion_times],
      state,
      newest,
    )
    return {
      "joint_pos": torch.cat([state["joint_pos"][:, 1:], joint_pos[:, None]], dim=1),
      "root_pos": torch.cat([state["root_pos"][:, 1:], root_pos[:, None]], dim=1),
      "root_quat": torch.cat([state["root_quat"][:, 1:], root_quat[:, None]], dim=1),
      **processes,
    }

  def clean(self, motion: MotionLoader, motion_times: torch.Tensor) -> StreamState:
    """The undegraded window ending at ``motion_times`` (evaluation, tests)."""
    frames = self._window_frames(motion, motion_times)
    return {
      "joint_pos": motion.joint_pos[frames],
      "root_pos": motion.root_pos_w[frames],
      "root_quat": motion.root_quat_w[frames],
    }
