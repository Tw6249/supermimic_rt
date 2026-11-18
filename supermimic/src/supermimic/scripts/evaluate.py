"""Closed-loop evaluation of a local SuperMimic run in the simulator.

Loads the checkpoint a training run wrote under ``logs/`` (no W&B needed), rolls
the actor out through one task slot's encoder, and reports falls, tracking
errors against the clean reference and jitter (see
``SuperMimicRunner.evaluate``). The env is the task's play configuration: no
pushes, no reset-pose randomization, startup domain randomization kept, resets
drawn uniformly over the clip.

``--reference degraded`` feeds the ``rt_4cam`` slot the same corrupted stream
it is trained on; ``clean`` feeds it the undegraded window. ``--noise-scale``
multiplies the magnitudes of the degraded stream (joint noise, height bias,
drift step, spike angle) to probe beyond the training distribution; frame-drop
and spike probabilities stay as trained. The ``motion`` slot reads the clip's
future window and ignores both flags.

  uv run src/supermimic/scripts/evaluate.py --run-dir logs/<task>/<timestamp> \\
    --motion-file "/path/clips/*.npz" --slot rt_4cam --reference degraded
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import mediapy as media
import mjlab.tasks  # noqa: F401
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.runners.run_config import check_representation_matches, load_runner_cfg
from supermimic.runners.supermimic_runner import SuperMimicRunner
from supermimic.scripts.train_utils import setup_device
from supermimic.tasks.rt_4cam import TASK_NAME as RT_4CAM


@dataclass
class EvalConfig:
  run_dir: Path
  motion_file: str
  slot: str = RT_4CAM
  reference: Literal["degraded", "clean"] = "degraded"
  noise_scale: float = 1.0
  shuffle_goal: bool = False
  """Control: feed every env another env's goal. See ``SuperMimicRunner.evaluate``."""
  num_envs: int = 1024
  num_steps: int = 1000
  seed: int = 0
  device: str | None = None
  video_file: Path | None = None
  """Also record env 0 to this .mp4. Use a few envs; resets then start at the
  clip's first frame so the video is the clip from its beginning."""
  tag: str | None = None
  """Suffix of the result file; defaults to the motion file's stem."""


def main() -> None:
  args = tyro.cli(EvalConfig, config=mjlab.TYRO_FLAGS)
  device = setup_device(args.device)
  torch.manual_seed(args.seed)

  run_config = json.loads((args.run_dir / "run_config.json").read_text())
  task_id = run_config["task_id"]
  cfg = load_runner_cfg(run_config)
  check_representation_matches(task_id, cfg.representation)
  if cfg.slot_mix.get(args.slot, 0.0) <= 0.0:
    raise SystemExit(f"slot '{args.slot}' was not trained in this run: {cfg.slot_mix}")

  # Evaluation settings that do not change what the actor is: no reset-sampler
  # state, a short replay buffer (unused), and an uncorrupted stream on request.
  sampling_mode = "start" if args.video_file else "uniform"
  cfg.sampling_mode = sampling_mode
  cfg.buffer_size = 16
  if args.reference == "clean":
    cfg.tasks = replace(
      cfg.tasks,
      rt_4cam=replace(
        cfg.tasks.rt_4cam,
        joint_noise_std=0.0,
        height_offset=0.0,
        xy_drift_std=0.0,
        hold_prob=0.0,
        spike_prob=0.0,
      ),
    )
  elif args.noise_scale != 1.0:
    trained = cfg.tasks.rt_4cam
    cfg.tasks = replace(
      cfg.tasks,
      rt_4cam=replace(
        trained,
        joint_noise_std=trained.joint_noise_std * args.noise_scale,
        height_offset=trained.height_offset * args.noise_scale,
        xy_drift_std=trained.xy_drift_std * args.noise_scale,
        spike_angle=trained.spike_angle * args.noise_scale,
      ),
    )

  env_cfg = load_env_cfg(task_id, play=True)
  motion_file = os.path.expanduser(args.motion_file)
  env_cfg.commands["motion"].motion_file = (
    [p.strip() for p in motion_file.split(",")] if "," in motion_file else motion_file
  )
  env_cfg.commands["motion"].sampling_mode = sampling_mode
  env_cfg.seed = args.seed
  env_cfg.scene.num_envs = args.num_envs
  render_mode = "rgb_array" if args.video_file else None
  if args.video_file:
    env_cfg.viewer.width, env_cfg.viewer.height = 1280, 720
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
  )
  frames: list = []

  runner = SuperMimicRunner(env=env, cfg=cfg, task_id=task_id, device=device)
  runner.load_checkpoint(args.run_dir)
  metrics = runner.evaluate(
    args.num_steps,
    args.slot,
    shuffle_goal=args.shuffle_goal,
    on_step=(lambda: frames.append(env.unwrapped.render()))
    if args.video_file
    else None,
  )
  if args.video_file:
    args.video_file.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(
      str(args.video_file), frames, fps=round(1 / env.unwrapped.step_dt)
    )
    print(f"[INFO] wrote {args.video_file}")

  reference = args.reference if args.slot == RT_4CAM else "clip"
  if reference == "degraded" and args.noise_scale != 1.0:
    reference = f"degraded_x{args.noise_scale:g}"
  if args.shuffle_goal:
    reference += "_shuffled"
  result = {
    "run_dir": str(args.run_dir),
    "task_id": task_id,
    "motion_file": args.motion_file,
    "slot": args.slot,
    "reference": reference,
    "noise_scale": args.noise_scale if reference.startswith("degraded") else None,
    "num_envs": args.num_envs,
    "num_steps": args.num_steps,
    "seed": args.seed,
    "shuffle_goal": args.shuffle_goal,
    "metrics": metrics,
  }
  tag = args.tag or Path(motion_file.split(",")[0]).stem.replace("*", "all")
  out = args.run_dir / f"eval_{args.slot}_{result['reference']}_{tag}.json"
  out.write_text(json.dumps(result, indent=2))
  print(json.dumps(metrics, indent=2))
  print(f"[INFO] wrote {out}")
  env.close()


if __name__ == "__main__":
  main()
