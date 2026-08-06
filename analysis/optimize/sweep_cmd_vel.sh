#!/usr/bin/env bash
#
# Grid sweep over --cmd-vel-x (the walk task's forward commanded velocity —
# see objective.py::_apply_cmd_vel_override() and cmaes_walk.py's
# --cmd-vel-x flag). Runs a full CMA-ES optimization once per speed, naming
# each wandb run "cmd_vel_<speed>".
#
# Usage: ./sweep_cmd_vel.sh
#
# START/STOP below bound the slice of CMD_VELS this run covers (both
# inclusive). Edit them directly to resume after a crash, or to run two
# instances in parallel across separate tmux panes — just make sure each
# pane also gets its own RESULTS_SUFFIX so they don't clobber each other's
# cmaes_log/best_params.json files. Leave STOP blank to run through the
# last value.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

MAXITER=15
POPSIZE=100
WORKERS=8

# Edit this list freely.
CMD_VELS=(0.5 1.0 1.5)

# Start point: skip every value before this one. Edit directly, or override
# via env var. Leave blank to run from the very start.
START="${START-}"
started=false
[ -z "$START" ] && started=true

# Stop point: end the sweep right after this value (inclusive). Leave blank
# (the default) to run through the last value.
STOP="${STOP-}"

RESULTS_DIR="$HERE/results${RESULTS_SUFFIX:-}"
CMAES_LOG_DIR="$RESULTS_DIR/cmaes_log"
BEST_PARAMS_FILE="$RESULTS_DIR/best_params.json"
mkdir -p "$RESULTS_DIR"

for v in "${CMD_VELS[@]}"; do
  if [ "$started" = false ]; then
    if [ "$v" = "$START" ]; then
      started=true
    else
      continue
    fi
  fi

  run_name="cmd_vel_${v}"
  echo "═══════════════════════════════════════════════════════"
  echo "  Running ${run_name}  (--cmd-vel-x ${v})"
  echo "═══════════════════════════════════════════════════════"

  python cmaes_walk.py \
    --cmd-vel-x "${v}" \
    --maxiter "${MAXITER}" \
    --popsize "${POPSIZE}" \
    --workers "${WORKERS}" \
    --run-name "${run_name}" \
    --results-dir "${RESULTS_DIR}"
  status=$?

  if [ $status -ne 0 ]; then
      echo "  !! ${run_name} exited with status ${status}, continuing to next speed"
  fi

  # cmaes_log and best_params.json are overwritten on every run — archive
  # them under this speed's name before the next iteration clobbers them.
  if [ -d "$CMAES_LOG_DIR" ]; then
      archive_dir="$RESULTS_DIR/cmaes_log_${run_name}"
      rm -rf "$archive_dir"
      mv "$CMAES_LOG_DIR" "$archive_dir"
  fi
  if [ -f "$BEST_PARAMS_FILE" ]; then
      mv "$BEST_PARAMS_FILE" "$RESULTS_DIR/best_params_${run_name}.json"
  fi

  echo

  if [ -n "$STOP" ] && [ "$v" = "$STOP" ]; then
      echo "Reached configured stop point (${run_name}); ending sweep here."
      break
  fi
done

echo "cmd_vel sweep complete."
