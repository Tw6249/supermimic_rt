"""Tests for the real-time causal tracking slot (rt_4cam)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

from supermimic.envs.multi_task_world_model_env import MultiTaskWorldModelEnv
from supermimic.models import MultiEncoderAE
from supermimic.runners.exporter import _SingleEncoderOnnx, _SplitInputOnnx
from supermimic.runners.run_config import (
  RT4CAM_SLOT_MIX,
  default_runner_cfg,
  load_runner_cfg,
)
from supermimic.runners.supermimic_runner import SuperMimicRunner, build_actor
from supermimic.tasks.rt_4cam import (
  TASK_NAME,
  Rt4CamStream,
  Rt4CamTaskCfg,
  build_causal_goal,
  goal_dim,
  goal_layout,
  limb_masks,
)
from supermimic.tasks.task_spec import (
  GoalContext,
  TasksCfg,
  TaskSpec,
  make_realtime_4cam_task,
  resolve_task_specs,
)
from tests.conftest import (
  MOTION_GOAL_DIM,
  make_mock_buffer,
  make_mock_command_env,
  make_mock_dynamics,
  make_mock_wm_representation,
)

JOINT_NAMES = [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
]
J = len(JOINT_NAMES)
K = 8
CLEAN = Rt4CamTaskCfg(
  joint_noise_std=0.0,
  height_offset=0.0,
  xy_drift_std=0.0,
  hold_prob=0.0,
  spike_prob=0.0,
)


def _motion(clip_lengths: tuple[int, ...] = (30, 20)) -> SimpleNamespace:
  """Two-clip stand-in for MotionLoader with distinct, smooth frames."""
  total = sum(clip_lengths)
  t = torch.arange(total, dtype=torch.float32)
  yaw = 0.05 * t
  zeros = torch.zeros(total)
  return SimpleNamespace(
    joint_pos=torch.randn(total, J),
    root_pos_w=torch.stack([0.02 * t, 0.01 * t, 0.75 + 0.001 * t], dim=-1),
    root_quat_w=quat_from_euler_xyz(0.1 * torch.sin(t), zeros, yaw),
    motion_ends=torch.tensor(clip_lengths).cumsum(0),
  )


def _stream(cfg: Rt4CamTaskCfg = CLEAN) -> Rt4CamStream:
  return Rt4CamStream(cfg, JOINT_NAMES, "cpu")


def _window(batch: int = 4):
  joint_pos = torch.randn(batch, K, J)
  root_pos = torch.randn(batch, K, 3)
  root_quat = quat_from_euler_xyz(
    0.3 * torch.randn(batch, K), 0.3 * torch.randn(batch, K), torch.randn(batch, K)
  )
  return joint_pos, root_pos, root_quat


# ---------------------------------------------------------------------------
# Goal
# ---------------------------------------------------------------------------


def test_goal_dim_and_layout_match_the_design():
  assert goal_dim(K, J) == 317
  layout = goal_layout(K, J)
  assert list(layout) == [
    "ref_joint_pos",
    "ref_projected_gravity",
    "ref_root_height",
    "ref_local_xy_yaw",
    "joint_pos_error",
  ]
  assert layout["ref_joint_pos"] == (0, K * J)
  assert layout["joint_pos_error"] == (317 - J, 317)


def test_goal_carries_no_absolute_horizontal_pose():
  joint_pos, root_pos, root_quat = _window()
  robot_q = torch.randn(4, J)
  goal = build_causal_goal(joint_pos, root_pos, root_quat, robot_q)

  # Move the whole window by a rigid horizontal transform: shift xy, turn about z.
  angle = torch.tensor(1.3)
  cos, sin = torch.cos(angle), torch.sin(angle)
  turned = root_pos.clone()
  turned[..., 0] = cos * root_pos[..., 0] - sin * root_pos[..., 1] + 5.0
  turned[..., 1] = sin * root_pos[..., 0] + cos * root_pos[..., 1] - 2.0
  zero = torch.zeros(())
  turn = quat_from_euler_xyz(zero, zero, angle).expand(4, K, 4)
  moved = build_causal_goal(joint_pos, turned, quat_mul(turn, root_quat), robot_q)

  torch.testing.assert_close(moved, goal, atol=1e-4, rtol=0)


def test_goal_blocks():
  joint_pos, root_pos, root_quat = _window()
  robot_q = torch.randn(4, J)
  goal = build_causal_goal(joint_pos, root_pos, root_quat, robot_q)
  layout = goal_layout(K, J)

  def block(name):
    lo, hi = layout[name]
    return goal[:, lo:hi]

  assert goal.shape == (4, 317)
  torch.testing.assert_close(block("ref_joint_pos"), joint_pos.flatten(1))
  torch.testing.assert_close(block("ref_root_height"), root_pos[..., 2])
  torch.testing.assert_close(block("joint_pos_error"), joint_pos[:, -1] - robot_q)
  # The oldest frame is the window's horizontal origin.
  local = block("ref_local_xy_yaw").unflatten(1, (K, 3))
  torch.testing.assert_close(local[:, 0], torch.zeros(4, 3), atol=1e-6, rtol=0)
  gravity = block("ref_projected_gravity").unflatten(1, (K, 3))
  torch.testing.assert_close(gravity.norm(dim=-1), torch.ones(4, K), atol=1e-5, rtol=0)


def test_upright_reference_projects_gravity_straight_down():
  joint_pos = torch.zeros(1, K, J)
  root_pos = torch.zeros(1, K, 3)
  zeros = torch.zeros(1, K)
  root_quat = quat_from_euler_xyz(zeros, zeros, torch.linspace(0, 1, K)[None])
  goal = build_causal_goal(joint_pos, root_pos, root_quat, torch.zeros(1, J))
  lo, hi = goal_layout(K, J)["ref_projected_gravity"]
  expected = torch.tensor([0.0, 0.0, -1.0]).repeat(1, K, 1)
  torch.testing.assert_close(
    goal[:, lo:hi].unflatten(1, (K, 3)), expected, atol=1e-6, rtol=0
  )


# ---------------------------------------------------------------------------
# Stream
# ---------------------------------------------------------------------------


def test_limb_masks_partition_the_joints():
  masks = limb_masks(JOINT_NAMES)
  assert masks.shape == (5, J)
  assert masks.sum(0).tolist() == [1.0] * J
  assert masks.sum(1).tolist() == [6.0, 6.0, 3.0, 7.0, 7.0]
  with pytest.raises(ValueError):
    limb_masks(["mystery_joint"])


def test_noise_free_stream_is_the_clean_window():
  motion = _motion()
  times = torch.tensor([12, 29, 45])
  state = _stream().init(motion, times)
  for k in range(K):
    frame = times - (K - 1 - k)
    torch.testing.assert_close(state["joint_pos"][:, k], motion.joint_pos[frame])
    torch.testing.assert_close(state["root_pos"][:, k], motion.root_pos_w[frame])
    torch.testing.assert_close(state["root_quat"][:, k], motion.root_quat_w[frame])


def test_window_never_reads_the_previous_clip():
  motion = _motion((30, 20))
  # Frame 32 is the third frame of the second clip, which starts at 30.
  state = _stream().init(motion, torch.tensor([32, 3]))
  expected_second = torch.tensor([30, 30, 30, 30, 30, 30, 31, 32])
  expected_first = torch.tensor([0, 0, 0, 0, 0, 1, 2, 3])
  torch.testing.assert_close(state["joint_pos"][0], motion.joint_pos[expected_second])
  torch.testing.assert_close(state["joint_pos"][1], motion.joint_pos[expected_first])


def test_stream_is_causal():
  """Frames after ``motion_times`` never reach the window."""
  motion = _motion()
  times = torch.tensor([12])
  before = _stream().init(motion, times)
  motion.joint_pos[13:] += 100.0
  motion.root_pos_w[13:] += 100.0
  after = _stream().init(motion, times)
  for key in ("joint_pos", "root_pos", "root_quat"):
    torch.testing.assert_close(after[key], before[key])


def test_advance_slides_the_window_and_keeps_old_corruption():
  torch.manual_seed(0)
  motion = _motion()
  stream = _stream(Rt4CamTaskCfg())
  state = stream.init(motion, torch.tensor([12, 40]))
  nxt = stream.advance(state, motion, torch.tensor([13, 41]))
  for key in ("joint_pos", "root_pos", "root_quat"):
    torch.testing.assert_close(nxt[key][:, :-1], state[key][:, 1:])
  torch.testing.assert_close(nxt["height_offset"], state["height_offset"])


def test_degradation_magnitudes():
  torch.manual_seed(0)
  motion = _motion((2000,))
  batch = 4096
  times = torch.randint(K, 2000, (batch,))
  cfg = Rt4CamTaskCfg(hold_prob=0.0, spike_prob=0.0)
  state = _stream(cfg).init(motion, times)
  clean = _stream().clean(motion, times)

  joint_err = state["joint_pos"] - clean["joint_pos"]
  assert abs(joint_err.std().item() - cfg.joint_noise_std) < 0.002

  # One height bias for the whole window, uniform in +-height_offset.
  dz = state["root_pos"][..., 2] - clean["root_pos"][..., 2]
  torch.testing.assert_close(dz, dz[:, :1].expand(-1, K), atol=1e-6, rtol=0)
  assert dz.abs().max().item() <= cfg.height_offset + 1e-6
  assert dz.abs().max().item() > 0.9 * cfg.height_offset

  # Horizontal drift is a random walk: std grows with the square root of steps.
  dxy = state["root_pos"][..., :2] - clean["root_pos"][..., :2]
  assert abs(dxy[:, 0].std().item() - cfg.xy_drift_std) < 0.001
  assert abs(dxy[:, -1].std().item() - cfg.xy_drift_std * math.sqrt(K)) < 0.003

  torch.testing.assert_close(state["root_quat"], clean["root_quat"])


def test_dropped_frames_repeat_the_previous_frame():
  motion = _motion()
  state = _stream(replace(CLEAN, hold_prob=1.0)).init(motion, torch.tensor([20]))
  oldest = motion.joint_pos[20 - (K - 1)]
  torch.testing.assert_close(state["joint_pos"][0], oldest.expand(K, J))

  torch.manual_seed(0)
  held = _stream(replace(CLEAN, hold_prob=0.15)).init(
    _motion((2000,)), torch.randint(K, 2000, (4096,))
  )
  repeated = (held["joint_pos"][:, 1:] == held["joint_pos"][:, :-1]).all(-1)
  assert abs(repeated.float().mean().item() - 0.15) < 0.01


def test_spike_bends_exactly_one_limb():
  torch.manual_seed(0)
  motion = _motion()
  cfg = replace(CLEAN, spike_prob=1.0)
  times = torch.full((64,), 20)
  state = _stream(cfg).init(motion, times)
  offset = (state["joint_pos"] - _stream().clean(motion, times)["joint_pos"])[:, -1]
  masks = limb_masks(JOINT_NAMES)
  touched = (offset.abs() > 1e-6).float()
  # Every env: the bent joints are all of one limb, the largest at spike_angle.
  per_limb = touched @ masks.T
  assert bool(((per_limb > 0).sum(-1) == 1).all())
  torch.testing.assert_close(
    offset.abs().amax(-1), torch.full((64,), cfg.spike_angle), atol=1e-5, rtol=0
  )


# ---------------------------------------------------------------------------
# Task spec, world-model env, runner, actor, export
# ---------------------------------------------------------------------------


def test_task_spec_contract():
  task = make_realtime_4cam_task(CLEAN, JOINT_NAMES)
  assert task.name == TASK_NAME
  assert task.goal_dim == 317
  assert task.stateful

  motion = _motion()
  times = torch.tensor([12, 40])
  robot_q = torch.randn(2, J)
  state = task.init_goal_state(motion, times)
  goal = task.extract_training_goal(
    motion, times, torch.zeros(2, 7), GoalContext(robot_q, state)
  )
  assert goal.shape == (2, 317)
  torch.testing.assert_close(goal[:, -J:], motion.joint_pos[times] - robot_q)
  with pytest.raises(ValueError):
    task.extract_training_goal(motion, times, torch.zeros(2, 7), GoalContext(robot_q))


def test_resolve_task_specs_adds_the_slot_only_with_joint_names():
  rep = make_mock_wm_representation(1)
  names = [ts.name for ts in resolve_task_specs(TasksCfg(), rep, num_goals=87)]
  assert names == ["motion", "velocity"]
  names = [
    ts.name
    for ts in resolve_task_specs(TasksCfg(), rep, num_goals=87, joint_names=JOINT_NAMES)
  ]
  assert names == ["motion", "velocity", TASK_NAME]


def test_default_runner_cfg():
  rt = default_runner_cfg("SuperMimic-RT4Cam-G1")
  assert rt.actor_obs == "proprio"
  assert rt.history_length > 1
  assert rt.slot_mix == RT4CAM_SLOT_MIX
  assert abs(sum(rt.slot_mix.values()) - 1.0) < 1e-9
  upstream = default_runner_cfg("SuperMimic-GenCoord-G1")
  assert upstream.slot_mix == {"motion": 1.0, "velocity": 0.0}
  assert upstream.actor_obs == "rep_state"


def test_run_config_survives_the_json_round_trip():
  """evaluate.py rebuilds the runner cfg from the run_config.json a run writes."""
  cfg = default_runner_cfg("SuperMimic-RT4Cam-G1")
  cfg.tasks = replace(cfg.tasks, rt_4cam=replace(cfg.tasks.rt_4cam, hold_prob=0.3))
  raw = json.loads(json.dumps({"runner_cfg": asdict(cfg)}))
  assert load_runner_cfg(raw) == cfg


def test_causal_slot_requires_the_proprio_sensor():
  cfg = default_runner_cfg("SuperMimic-RT4Cam-G1")
  cfg.actor_obs = "rep_state"
  wm = SimpleNamespace(
    representation=SimpleNamespace(rep_state_dim=68),
    num_actions=J,
    num_goals=670,
    joint_names=JOINT_NAMES,
  )
  with pytest.raises(RuntimeError, match="proprio"):
    build_actor(cfg, wm, "cpu")


BATCH = 8


def _wm_env(
  cfg: Rt4CamTaskCfg = CLEAN,
) -> tuple[MultiTaskWorldModelEnv, SimpleNamespace]:
  rep = make_mock_wm_representation(BATCH)
  robot_q = torch.randn(BATCH, J)
  rep.measured_joint_pos.side_effect = lambda sensor, joint_state: robot_q
  env = make_mock_command_env()
  motion = _motion((10,))
  motion.body_state = (
    env.unwrapped.command_manager.get_term.return_value.motion.body_state
  )
  env.unwrapped.command_manager.get_term.return_value.motion = motion
  motion_goal = torch.randn(BATCH, MOTION_GOAL_DIM)
  specs = [
    TaskSpec(
      name="motion",
      goal_dim=MOTION_GOAL_DIM,
      extract_training_goal=lambda m, mt, aux: motion_goal,
    ),
    make_realtime_4cam_task(cfg, JOINT_NAMES),
  ]
  wm_env = MultiTaskWorldModelEnv(
    dynamics=make_mock_dynamics(),
    representation=rep,
    env=env,
    batch_size=BATCH,
    replay_buffer=make_mock_buffer(),
    task_specs=specs,
    history_length=1,
    termination_softness=0.02,
    device="cpu",
  )
  return wm_env, SimpleNamespace(motion=motion, robot_q=robot_q)


def test_wm_env_emits_the_causal_goal():
  wm_env, ctx = _wm_env()
  obs = wm_env.reset()
  assert obs["goal_rt_4cam"].shape == (BATCH, 317)
  assert obs["goal_motion"].shape == (BATCH, MOTION_GOAL_DIM)
  # The mock buffer starts every env at frame 0.
  torch.testing.assert_close(
    obs["goal_rt_4cam"][:, -J:], ctx.motion.joint_pos[0] - ctx.robot_q
  )

  obs, _, _, _ = wm_env.step(torch.zeros(BATCH, 29))
  assert wm_env.motion_times.tolist() == [1] * BATCH
  torch.testing.assert_close(
    obs["goal_rt_4cam"][:, -J:], ctx.motion.joint_pos[1] - ctx.robot_q
  )
  # The window's newest reference pose is the frame the reward scores next.
  lo, hi = goal_layout(K, J)["ref_joint_pos"]
  newest = obs["goal_rt_4cam"][:, lo:hi].unflatten(1, (K, J))[:, -1]
  torch.testing.assert_close(newest, ctx.motion.joint_pos[1].expand(BATCH, J))


def test_wm_env_redraws_the_noise_at_reset():
  torch.manual_seed(0)
  wm_env, _ = _wm_env(Rt4CamTaskCfg())
  first = wm_env.reset()["goal_rt_4cam"]
  second = wm_env.reset()["goal_rt_4cam"]
  assert not torch.allclose(first, second)


def test_runner_restarts_the_stream_only_for_reset_envs():
  torch.manual_seed(0)
  motion = _motion((2000,))
  # Joint noise and the height bias only: the stand-in clip's frames are
  # unrelated to each other, so a repeated frame would not stay near its own.
  cfg = Rt4CamTaskCfg(hold_prob=0.0, spike_prob=0.0)
  task = make_realtime_4cam_task(cfg, JOINT_NAMES)
  runner = SuperMimicRunner.__new__(SuperMimicRunner)
  runner.motion = motion
  runner._sim_goal_states = {}
  runner._sim_goal_restart = torch.ones(4, dtype=torch.bool)

  first = runner._advance_sim_goal_state(task, torch.tensor([100, 200, 300, 400]))
  # Env 1 was reset to a new frame; the others advanced one tick.
  runner._sim_goal_restart = torch.tensor([False, True, False, False])
  second = runner._advance_sim_goal_state(task, torch.tensor([101, 900, 301, 401]))

  kept = torch.tensor([0, 2, 3])
  torch.testing.assert_close(
    second["joint_pos"][kept, :-1], first["joint_pos"][kept, 1:]
  )
  torch.testing.assert_close(
    second["height_offset"][kept], first["height_offset"][kept]
  )
  assert not torch.allclose(second["joint_pos"][1, :-1], first["joint_pos"][1, 1:])
  # The restarted window holds frames of the new location only.
  clean = Rt4CamStream(CLEAN, JOINT_NAMES, "cpu").clean(motion, torch.tensor([900]))
  assert (second["joint_pos"][1] - clean["joint_pos"][0]).abs().max() < 1.0


OBS_DIM = 3 + 3 + J + J + J
TERM_WIDTHS = [3, 3, J, J, J]
HISTORY = 4


def _actor(slot_mix: dict[str, float]) -> MultiEncoderAE:
  specs = [
    TaskSpec(name="motion", goal_dim=MOTION_GOAL_DIM, extract_training_goal=None),
    make_realtime_4cam_task(CLEAN, JOINT_NAMES),
  ]
  return MultiEncoderAE(
    obs_dim=OBS_DIM,
    action_dim=J,
    task_specs=specs,
    slot_mix=slot_mix,
    latent_dim=32,
    hidden_dims=(64, 64),
    latent_bc_weight=0.1,
    history_length=HISTORY,
  )


def test_actor_pulls_the_causal_latent_toward_the_motion_anchor():
  from tensordict import TensorDict

  actor = _actor({"motion": 0.25, TASK_NAME: 0.75}).train()
  batch = 16
  obs = TensorDict(
    {
      "history": torch.randn(batch, HISTORY, OBS_DIM),
      "goal_motion": torch.randn(batch, MOTION_GOAL_DIM),
      "goal_rt_4cam": torch.randn(batch, 317),
    },
    batch_size=[batch],
  )
  assert actor(obs).shape == (batch, J)
  aux = actor.auxiliary_loss()
  assert torch.all(aux[:4] == 0.0)  # motion slot is the anchor
  assert torch.all(aux[4:] > 0.0)
  aux.sum().backward()
  assert actor.encoders[TASK_NAME][0].weight.grad.abs().sum() > 0
  assert actor.encoders["motion"][0].weight.grad is None
  assert actor.decoder[0].weight.grad is None


def test_exported_module_matches_the_actor():
  from tensordict import TensorDict

  torch.manual_seed(0)
  actor = _actor({"motion": 0.0, TASK_NAME: 1.0}).eval()
  batch = 5
  history = torch.randn(batch, HISTORY, OBS_DIM)
  goal = torch.randn(batch, 317)
  obs = TensorDict(
    {
      "history": history,
      "goal_motion": torch.zeros(batch, MOTION_GOAL_DIM),
      "goal_rt_4cam": goal,
    },
    batch_size=[batch],
  )
  expected = actor(obs)

  # The deployed graph takes mjlab's term-major history: each sensor term's H
  # frames are contiguous.
  term_major = torch.cat(
    [chunk.flatten(1) for chunk in history.split(TERM_WIDTHS, dim=-1)], dim=-1
  )
  module = _SplitInputOnnx(_SingleEncoderOnnx(actor, TASK_NAME, TERM_WIDTHS))
  torch.testing.assert_close(module(term_major, goal), expected, atol=1e-5, rtol=0)
  # The latent the decoder reads is bounded.
  latent = torch.tanh(
    actor.encoders[TASK_NAME](actor.goal_normalizers[TASK_NAME](goal * 1e3))
  )
  assert latent.abs().max() <= 1.0
