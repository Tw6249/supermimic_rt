"""Check an exported ``policy_rt_4cam.onnx`` against the checkpoint it came from.

Rebuilds the actor from ``actor.pt`` (sizes are read from the state dict), runs
random inputs through the deployed slice in PyTorch and through the ONNX graph
(``onnx.reference``, no onnxruntime needed), and reports the largest action
difference. Exits non-zero when it exceeds ``--tolerance``.

Run from the supermimic checkout:
  uv run python ../tools/check_rt4cam_onnx.py --run-dir logs/<task>/<timestamp>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import tyro
from onnx.reference import ReferenceEvaluator

from supermimic.models import MultiEncoderAE
from supermimic.runners.exporter import _SingleEncoderOnnx, _SplitInputOnnx
from supermimic.tasks.rt_4cam import TASK_NAME
from supermimic.tasks.task_spec import TaskSpec


def _rebuild_actor(state: dict[str, torch.Tensor]) -> tuple[MultiEncoderAE, int]:
  names = sorted({k.split(".")[1] for k in state if k.startswith("encoders.")})
  # Slot order is the order the encoders were registered in.
  names = [n for n in ("motion", "velocity", TASK_NAME) if n in names]
  specs = [
    TaskSpec(
      name=n,
      goal_dim=state[f"encoders.{n}.0.weight"].shape[1],
      extract_training_goal=None,
    )
    for n in names
  ]
  linear = sorted(
    int(k.split(".")[1])
    for k in state
    if k.startswith("decoder.") and k.endswith(".weight") and state[k].dim() == 2
  )
  hidden = tuple(state[f"decoder.{i}.weight"].shape[0] for i in linear[:-1])
  action_dim = state[f"decoder.{linear[-1]}.weight"].shape[0]
  latent_dim = state[f"encoders.{names[0]}.{linear[-1]}.weight"].shape[0]
  obs_dim = state["obs_normalizer._mean"].shape[-1]
  decoder_in = state["decoder.0.weight"].shape[1]
  history = (decoder_in - latent_dim) // obs_dim
  actor = MultiEncoderAE(
    obs_dim=obs_dim,
    action_dim=action_dim,
    task_specs=specs,
    slot_mix={names[0]: 1.0},
    latent_dim=latent_dim,
    hidden_dims=hidden,
    latent_bc_weight=0.0,
    history_length=history,
  )
  actor.load_state_dict(state)
  return actor.eval(), action_dim


def main(run_dir: Path, tolerance: float = 1e-4, samples: int = 8) -> None:
  state = torch.load(run_dir / "actor.pt", map_location="cpu")
  actor, num_joints = _rebuild_actor(state)
  term_widths = [3, 3, num_joints, num_joints, num_joints]
  single = _SingleEncoderOnnx(actor, TASK_NAME, term_widths)
  module = _SplitInputOnnx(single)

  model = onnx.load(str(run_dir / f"policy_{TASK_NAME}.onnx"))
  onnx.checker.check_model(model)
  inputs = {
    i.name: [d.dim_value for d in i.type.tensor_type.shape.dim]
    for i in model.graph.input
  }
  meta = {p.key: p.value for p in model.metadata_props}
  print(f"[INFO] inputs: {inputs}")
  print(f"[INFO] goal_history_frames: {meta.get('goal_history_frames')}")
  print(f"[INFO] goal_layout: {meta.get('goal_layout')}")
  assert inputs["obs_history"] == [1, single.hist_size]
  assert inputs["goal"] == [1, single.goal_dim]

  session = ReferenceEvaluator(model)
  torch.manual_seed(0)
  worst = 0.0
  for _ in range(samples):
    history = torch.randn(1, single.hist_size)
    goal = torch.randn(1, single.goal_dim)
    with torch.no_grad():
      expected = module(history, goal).numpy()
    got = session.run(None, {"obs_history": history.numpy(), "goal": goal.numpy()})[0]
    worst = max(worst, float(np.abs(got - expected).max()))
  print(f"[INFO] max |onnx - torch| over {samples} samples: {worst:.3e}")
  if worst > tolerance:
    sys.exit(f"[FAIL] exceeds tolerance {tolerance:g}")
  print("[OK] exported graph matches the checkpoint")


if __name__ == "__main__":
  tyro.cli(main)
