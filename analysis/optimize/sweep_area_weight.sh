#!/usr/bin/env bash
#
# Grid sweep over --area-weight (the closed-form active/passive FL-curve
# area term added to locomotion cost — see objective.py::evaluate() and
# curve_area_mean()). Runs a full CMA-ES optimization once per weight,
# naming each wandb run "area_<weight>".
#
# 0 is included by default as the baseline/control run (no curve-shape
# pressure at all, i.e. today's plain locomotion-cost behavior), so you
# have something to compare every other weight against.
#
# mppi_sim's own locomotion cost has run-to-run noise on the order of
# ±150-300 for an identical candidate (see cmaes_walk.py discussion), so
# very small weights may show little to no visible effect on the resulting
# curves — that's expected, not a bug; the point of this sweep is to find
# empirically where the weight starts actually reshaping things.
#
# Usage: ./sweep_area_weight.sh
#
# START/STOP below bound the slice of AREA_WEIGHTS this run covers (both
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
WORKERS=6

# Edit this list freely. A doubling sequence covers a couple orders of
# magnitude with few runs; 0 is the no-pressure control.
AREA_WEIGHTS=(0 400)

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

for w in "${AREA_WEIGHTS[@]}"; do
  if [ "$started" = false ]; then
    if [ "$w" = "$START" ]; then
      started=true
    else
      continue
    fi
  fi

  run_name="area_${w}"
  echo "═══════════════════════════════════════════════════════"
  echo "  Running ${run_name}  (--area-weight ${w})"
  echo "═══════════════════════════════════════════════════════"

  python cmaes_walk.py \
    --area-weight "${w}" \
    --maxiter "${MAXITER}" \
    --popsize "${POPSIZE}" \
    --workers "${WORKERS}" \
    --run-name "${run_name}" \
    --results-dir "${RESULTS_DIR}"
  status=$?

  if [ $status -ne 0 ]; then
      echo "  !! ${run_name} exited with status ${status}, continuing to next weight"
  fi

  # cmaes_log and best_params.json are overwritten on every run — archive
  # them under this weight's name before the next iteration clobbers them.
  if [ -d "$CMAES_LOG_DIR" ]; then
      archive_dir="$RESULTS_DIR/cmaes_log_${run_name}"
      rm -rf "$archive_dir"
      mv "$CMAES_LOG_DIR" "$archive_dir"
  fi
  if [ -f "$BEST_PARAMS_FILE" ]; then
      mv "$BEST_PARAMS_FILE" "$RESULTS_DIR/best_params_${run_name}.json"
  fi

  echo

  if [ -n "$STOP" ] && [ "$w" = "$STOP" ]; then
      echo "Reached configured stop point (${run_name}); ending sweep here."
      break
  fi
done

echo "Area-weight sweep complete."
