"""Print one markdown table row per eval_*.json under the given run dirs.

  uv run python -I ../tools/summarize_evals.py logs/<task>/<timestamp> [...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

COLUMNS = (
  ("falls/min", "falls_per_min", 1.0, "{:.3f}"),
  ("joint err mean (rad)", "joint_pos_err_mean", 1.0, "{:.3f}"),
  ("body pos mean / p99 (cm)", ("body_pos_err_mean", "body_pos_err_p99"), 100.0, "{:.1f}"),
  ("root xy mean (m)", "root_xy_err_mean", 1.0, "{:.2f}"),
  ("height p99 (cm)", "root_height_err_p99", 100.0, "{:.1f}"),
  ("target step p50 / p99 / max (deg)", ("target_step_p50", "target_step_p99", "target_step_max"), 57.2958, "{:.1f}"),
  ("joint acc p50 / p99 (rad/s^2)", ("joint_acc_p50", "joint_acc_p99"), 1.0, "{:.0f}"),
)


def cell(metrics: dict, keys, scale: float, fmt: str) -> str:
  keys = (keys,) if isinstance(keys, str) else keys
  return " / ".join(
    fmt.format(metrics[k] * scale) if k in metrics else "n/a" for k in keys
  )


def main() -> None:
  header = ["run", "clip", "slot", "reference"] + [c[0] for c in COLUMNS]
  print("| " + " | ".join(header) + " |")
  print("|" + "---|" * len(header))
  for run_dir in sys.argv[1:]:
    run = json.loads((Path(run_dir) / "run_config.json").read_text())
    for path in sorted(Path(run_dir).glob("eval_*.json")):
      r = json.loads(path.read_text())
      row = [
        run["task_id"].split("-")[1],
        Path(r["motion_file"]).stem,
        r["slot"],
        r["reference"],
      ] + [cell(r["metrics"], k, s, f) for _, k, s, f in COLUMNS]
      print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
  main()
