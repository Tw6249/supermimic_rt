"""Convert retargeted G1 motion CSVs to mjlab motion ``.npz`` files, locally.

Same conversion as ``mjlab.scripts.csv_to_npz`` (resample to the control rate,
finite-difference velocities) without the W&B upload.

Input: a directory of CSVs, one row per frame:
  ``[root_pos xyz (m), root_quat xyzw, joint_pos (29, rad)]`` in the joint order
  of ``mjlab.scripts.csv_to_npz.JOINT_NAMES``, sampled at ``--input-fps``.
Output: ``<output-dir>/<clip>.npz`` holding ``fps``, ``joint_names``,
  ``joint_pos``, ``joint_vel``, ``root_pos_w``, ``root_quat_w`` (wxyz),
  ``root_lin_vel_w``, ``root_ang_vel_w`` at ``--output-fps`` — what
  ``mjlab.tasks.tracking.mdp.commands.MotionLoader`` reads.
A clip whose CSV does not parse aborts the run; nothing is skipped silently.

Run from the supermimic checkout:
  uv run python -I ../tools/convert_csv_motions.py --input-dir ... --output-dir ...
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tyro
from mjlab.scripts.csv_to_npz import JOINT_NAMES, MotionLoader


def main(
  input_dir: Path,
  output_dir: Path,
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  overwrite: bool = False,
) -> None:
  files = sorted(input_dir.glob("*.csv"))
  if not files:
    raise SystemExit(f"no CSV files in {input_dir}")
  output_dir.mkdir(parents=True, exist_ok=True)
  expected_columns = 7 + len(JOINT_NAMES)
  total_frames = 0
  for path in files:
    out = output_dir / f"{path.stem}.npz"
    if out.exists() and not overwrite:
      total_frames += int(np.load(out)["joint_pos"].shape[0])
      continue
    with path.open() as fh:
      columns = len(fh.readline().split(","))
    if columns != expected_columns:
      raise SystemExit(f"{path}: {columns} columns, expected {expected_columns}")
    motion = MotionLoader(
      motion_file=str(path),
      input_fps=input_fps,
      output_fps=output_fps,
      device="cpu",
    )
    arrays = motion.arrays()
    if not all(np.isfinite(v).all() for k, v in arrays.items() if k != "joint_names"):
      raise SystemExit(f"{path}: non-finite values after conversion")
    np.savez(out, **arrays)
    total_frames += int(arrays["joint_pos"].shape[0])
  print(
    f"[INFO] {len(files)} clips, {total_frames} frames "
    f"({total_frames / output_fps / 60:.1f} min at {output_fps:g} Hz) in {output_dir}"
  )


if __name__ == "__main__":
  tyro.cli(main)
