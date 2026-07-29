#!/usr/bin/env bash
#
# Grid sweep over FVmax, independently per joint type (hip/thigh/calf),
# for cmaes_walk.py. Runs a full CMA-ES optimization once per combination,
# naming each wandb run "h<hip>_t<thigh>_c<calf>".
#
# 4 values/joint x 3 joints = 64 combinations. Each run is popsize=100 x
# maxiter=10 (1000 evals), taking on the order of ~90-100 min with workers=8,
# so this is roughly ~100 hours total — run it unattended (e.g. under
# nohup/disown) and check back periodically.
#
# Usage: ./sweep_fvmax.sh
#
# START_H/T/C and STOP_H/T/C below bound the slice of the h/t/c nested-loop
# order this run covers (both inclusive). Edit them directly to resume after
# a crash or to run two instances in parallel across separate tmux panes —
# just make sure each pane also gets its own RESULTS_SUFFIX so they don't
# clobber each other's cmaes_log files. Leave STOP_* blank to run through the
# last combination.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

MAXITER=10
POPSIZE=100
WORKERS=6

# 1.1 -> 1.4 in steps of 0.1, independently per joint type.
FVMAX_VALUES=(1.1 1.2 1.3 1.4)

# Start point: skip every combination before this one. Edit directly, or
# override via env var. Leave all three blank to run from the very start.
START_H="${START_H-1.2}"
START_T="${START_T-1.3}"
START_C="${START_C-1.4}"
started=false
[ -z "$START_H" ] && started=true

# Stop point: end the sweep right after this combination (inclusive). Leave
# all three blank (the default) to run through the last combination.
STOP_H="${STOP_H-}"
STOP_T="${STOP_T-}"
STOP_C="${STOP_C-}"

RESULTS_DIR="$HERE/results${RESULTS_SUFFIX:-}"
CMAES_LOG_DIR="$RESULTS_DIR/cmaes_log"
mkdir -p "$RESULTS_DIR"

for h in "${FVMAX_VALUES[@]}"; do
  for t in "${FVMAX_VALUES[@]}"; do
    for c in "${FVMAX_VALUES[@]}"; do
      if [ "$started" = false ]; then
        if [ "$h" = "$START_H" ] && [ "$t" = "$START_T" ] && [ "$c" = "$START_C" ]; then
          started=true
        else
          continue
        fi
      fi

      run_name="h${h}_t${t}_c${c}"
      echo "═══════════════════════════════════════════════════════"
      echo "  Running ${run_name}  (FVMAX_HIP=${h} FVMAX_THIGH=${t} FVMAX_CALF=${c})"
      echo "═══════════════════════════════════════════════════════"

      FVMAX_HIP="${h}" FVMAX_THIGH="${t}" FVMAX_CALF="${c}" \
        python cmaes_walk.py \
          --maxiter "${MAXITER}" \
          --popsize "${POPSIZE}" \
          --workers "${WORKERS}" \
          --run-name "${run_name}" \
          --results-dir "${RESULTS_DIR}"
      status=$?

      if [ $status -ne 0 ]; then
          echo "  !! ${run_name} exited with status ${status}, continuing to next combination"
      fi

      # cmaes_log is overwritten on every run — archive it under this
      # combination's name before the next iteration clobbers it.
      if [ -d "$CMAES_LOG_DIR" ]; then
          archive_dir="$RESULTS_DIR/cmaes_log_${run_name}"
          rm -rf "$archive_dir"
          mv "$CMAES_LOG_DIR" "$archive_dir"
      fi

      echo

      if [ -n "$STOP_H" ] && [ "$h" = "$STOP_H" ] && [ "$t" = "$STOP_T" ] && [ "$c" = "$STOP_C" ]; then
          echo "Reached configured stop point (${run_name}); ending sweep here."
          break 3
      fi
    done
  done
done

echo "Joint-grid sweep complete."
