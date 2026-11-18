#!/usr/bin/env bash
# Closed-loop evaluation matrix for one local run: every trained slot, on each
# motion file given. Results land next to the checkpoint as eval_*.json.
#
#   tools/eval_matrix.sh <run_dir> <motion.npz> [<motion.npz> ...]
#
# Run from the supermimic checkout. Slots are read from the run's run_config.json.
set -euo pipefail
run_dir=$1; shift
slots=$(uv run python -c "
import json, sys
mix = json.load(open(sys.argv[1] + '/run_config.json'))['runner_cfg']['slot_mix']
print(' '.join(k for k, v in mix.items() if v > 0))" "$run_dir" | tail -1)
for motion in "$@"; do
  for slot in $slots; do
    refs="clean"
    [ "$slot" = "rt_4cam" ] && refs="degraded clean"
    for ref in $refs; do
      WANDB_MODE=offline uv run src/supermimic/scripts/evaluate.py \
        --run-dir "$run_dir" --motion-file "$motion" --slot "$slot" \
        --reference "$ref" --num-envs 1024 --num-steps 1000 2>&1 |
        grep -E "^\[INFO\] wrote|Traceback|Error" || true
    done
  done
done
