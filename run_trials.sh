#!/usr/bin/env bash
#
# Batch trial runner for mppi_sim (muscle-actuated) and pd_mppi_sim (PD-actuated).
#
# Both sims already accumulate their CSVs under analysis/log/trials/<name>/trial_NNN/
# via their own --save flag (see README "Saving Trials"). What they do NOT keep is
# the console output — the per-solve [cost] diagnostics, the stand-up height, the
# convergence solve time, whether the robot fell and when. This script runs each
# binary N times back-to-back and drops that console output into the same
# trial_NNN/ directory as console.log, so every run's numbers stay with its data.
#
# Runs are strictly sequential: one sim process at a time, muscle batch first,
# then PD. Both binaries use every core for their rollout loop (num_threads: 0),
# so overlapping them would make the solve-time numbers meaningless.
#
# Usage (from anywhere):
#   ./run_trials.sh                          # 100 muscle + 100 pd, task "guinea_fowl", under workshop/
#   ./run_trials.sh -n 20                    # 20 of each
#   ./run_trials.sh -t walk -N flat          # different task, saved under flat/
#   ./run_trials.sh --muscle-only            # skip the pd batch
#   ./run_trials.sh --tee                    # also mirror each run's output to the terminal
#
# Output layout:
#   analysis/log/trials/<name>/muscle/trial_NNN/{mppi_sim.csv,mppi_sim_qpos.csv,console.log}
#   analysis/log/trials/<name>/pd/trial_NNN/{pd_mppi_sim.csv,pd_mppi_sim_qpos.csv,console.log}
#   analysis/log/trials/<name>/batch_<timestamp>/{batch.log,summary.csv,failed/}
#
# Trial indices continue from whatever is already there, so re-running this adds
# to the existing set rather than overwriting it.

set -uo pipefail

# ── locate the repo from this script, not the CWD ────────────────────────────
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$REPO_ROOT/controllers/build"
TRIALS_DIR="$REPO_ROOT/analysis/log/trials"

# ── defaults ─────────────────────────────────────────────────────────────────
N_RUNS=100
TASK="guinea_fowl"
BATCH_NAME="workshop"
RUN_MUSCLE=1
RUN_PD=1
TEE=0
MUSCLE_YAML="../muscle/utils/tasks.yaml"
PD_YAML="../pd/utils/tasks_pd.yaml"

usage() {
    sed -n '2,/^$/s/^# \?//p' "${BASH_SOURCE[0]}"
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--runs)        N_RUNS="$2"; shift 2 ;;
        -t|--task)        TASK="$2"; shift 2 ;;
        -N|--name)        BATCH_NAME="$2"; shift 2 ;;
        --muscle-yaml)    MUSCLE_YAML="$2"; shift 2 ;;
        --pd-yaml)        PD_YAML="$2"; shift 2 ;;
        --muscle-only)    RUN_PD=0; shift ;;
        --pd-only)        RUN_MUSCLE=0; shift ;;
        --tee)            TEE=1; shift ;;
        -h|--help)        usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

[[ "$N_RUNS" =~ ^[0-9]+$ && "$N_RUNS" -gt 0 ]] || { echo "-n needs a positive integer" >&2; exit 1; }
[[ "$BATCH_NAME" == *".."* ]] && { echo "--name must not contain '..'" >&2; exit 1; }

for bin in mppi_sim pd_mppi_sim; do
    [[ -x "$BUILD_DIR/$bin" ]] || { echo "Missing $BUILD_DIR/$bin — build first (see README)." >&2; exit 1; }
done

# Line-buffer the sims so a --tee'd run and a killed run both keep their output.
STDBUF=(); command -v stdbuf >/dev/null && STDBUF=(stdbuf -oL -eL)

# ── one batch at a time, machine-wide ────────────────────────────────────────
# The whole point is that no two sims overlap; a second invocation of this
# script would break that just as surely as backgrounding one run would.
LOCK="$TRIALS_DIR/.run_trials.lock"
mkdir -p "$TRIALS_DIR"
exec {lock_fd}>"$LOCK"
if ! flock -n "$lock_fd"; then
    echo "Another run_trials.sh is already running (lock: $LOCK). Refusing to run two batches at once." >&2
    exit 1
fi

# ── batch bookkeeping ────────────────────────────────────────────────────────
TS="$(date +%Y%m%d_%H%M%S)"
BATCH_DIR="$TRIALS_DIR/$BATCH_NAME/batch_$TS"
mkdir -p "$BATCH_DIR/failed"
BATCH_LOG="$BATCH_DIR/batch.log"
SUMMARY="$BATCH_DIR/summary.csv"

say() { echo "$*" | tee -a "$BATCH_LOG"; }

GIT_REV="$(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY=""
(cd "$REPO_ROOT" && git diff --quiet HEAD -- 2>/dev/null) || GIT_DIRTY=" (dirty)"

{
    echo "batch          : $BATCH_NAME/batch_$TS"
    echo "started        : $(date -Is)"
    echo "host           : $(hostname)"
    echo "task           : $TASK"
    echo "runs each      : $N_RUNS"
    echo "muscle batch   : $RUN_MUSCLE   (yaml $MUSCLE_YAML)"
    echo "pd batch       : $RUN_PD   (yaml $PD_YAML)"
    echo "git            : $GIT_REV$GIT_DIRTY"
    echo "mppi_sim       : $(date -Is -r "$BUILD_DIR/mppi_sim")  sha256 $(sha256sum "$BUILD_DIR/mppi_sim" | cut -c1-16)"
    echo "pd_mppi_sim    : $(date -Is -r "$BUILD_DIR/pd_mppi_sim")  sha256 $(sha256sum "$BUILD_DIR/pd_mppi_sim" | cut -c1-16)"
    echo
} > "$BATCH_LOG"
cat "$BATCH_LOG"

echo "controller,run_index,trial,trial_dir,exit_code,wall_s,stand_height_m,converged_solve_ms,avg_solve_ms,solves,fell,fall_t,reached_goal,phase_advances,early_stop_t,logged_rows,logged_t_end" > "$SUMMARY"

# Ctrl-C reaches the sim too (same foreground process group), so the run in
# flight dies and — since the sims only copy into trial_NNN/ once they finish —
# is lost; its console output still lands in batch_*/failed/. The flag just
# stops the batch from starting anything further.
ABORT=0
trap 'ABORT=1; echo; echo "Interrupted — stopping after the current run."' INT TERM

# ── run one sim once, capture its console output into its trial directory ────
# $1 label (muscle|pd)  $2 binary  $3 yaml  $4 --save name  $5 run index  $6 csv basename
run_one() {
    local label="$1" bin="$2" yaml="$3" save="$4" idx="$5" csv_base="$6"
    local tmp; tmp="$(mktemp)"
    local cmd="./$bin $TASK $yaml --save $save"

    local t0 t1 wall_ms rc
    t0="$(date +%s%N)"
    if (( TEE )); then
        ( cd "$BUILD_DIR" && "${STDBUF[@]}" "./$bin" "$TASK" "$yaml" --save "$save" ) 2>&1 | tee "$tmp"
        rc="${PIPESTATUS[0]}"
    else
        ( cd "$BUILD_DIR" && "${STDBUF[@]}" "./$bin" "$TASK" "$yaml" --save "$save" ) >"$tmp" 2>&1
        rc=$?
    fi
    t1="$(date +%s%N)"
    wall_ms=$(( (t1 - t0) / 1000000 ))
    local wall_s; wall_s="$(printf '%d.%03d' $((wall_ms / 1000)) $((wall_ms % 1000)))"

    # ── pull the run's own numbers back out of what it printed ───────────────
    local trial_dir stand_h conv_ms avg_ms solves fell fall_t done_t rows t_end
    trial_dir="$(sed -n 's/^Trial saved to //p' "$tmp" | tail -1)"
    stand_h="$(sed -n 's/^Stand-up complete\. Body height: \([0-9.-]*\) m.*/\1/p' "$tmp" | tail -1)"
    conv_ms="$(sed -n 's/^Converged (avg solve \([0-9.]*\) ms).*/\1/p' "$tmp" | tail -1)"
    avg_ms="$(sed -n 's/^Avg MPPI solve: \([0-9.]*\) ms over .*/\1/p' "$tmp" | tail -1)"
    solves="$(sed -n 's/^Avg MPPI solve: [0-9.]* ms over \([0-9]*\) solves.*/\1/p' "$tmp" | tail -1)"
    fall_t="$(sed -n 's/^Robot fell at t=\([0-9.]*\).*/\1/p' "$tmp" | tail -1)"
    fell=0; [[ -n "$fall_t" ]] && fell=1

    # Two different "done" signals — related, but not the same thing:
    #  * "[phase] task complete." comes from the CONTROLLER (advance_phase, in
    #    both mppi_locomotion.cpp and mppi_locomotion_pd.cpp) the moment the
    #    final phase's dwell gate passes.
    #  * "Task complete at t=..." comes from the SIM's early-stop, and carries
    #    the sim time. Both sims have it now; mppi_sim.cpp did not until
    #    MPPILocomotion gained a task_success() accessor, so trials logged
    #    before that show reached_goal=1 with an empty early_stop_t. Scoring
    #    success off the timestamp alone silently reported every one of those
    #    muscle runs as a failure — reached_goal is the portable column.
    local reached advances
    reached=0; grep -q '^\[phase\] task complete\.' "$tmp" && reached=1
    advances="$(grep -c '^\[phase\] -> ' "$tmp")"
    done_t="$(sed -n 's/^Task complete at t=\([0-9.]*\).*/\1/p' "$tmp" | tail -1)"

    # ── file the console output next to the data it describes ────────────────
    local console_dest trial_id
    if [[ -n "$trial_dir" && -d "$trial_dir" ]]; then
        console_dest="$trial_dir/console.log"
        trial_id="$(basename "$trial_dir")"
        rows="$(( $(wc -l < "$trial_dir/$csv_base.csv" 2>/dev/null || echo 1) - 1 ))"
        t_end="$(tail -1 "$trial_dir/$csv_base.csv" 2>/dev/null | cut -d, -f1)"
    else
        # No trial directory: the sim died before its --save copy, or --save
        # itself failed. Keep the output anyway — that log is the only record.
        console_dest="$BATCH_DIR/failed/${label}_run$(printf '%03d' "$idx").log"
        trial_id="none"; trial_dir=""; rows=""; t_end=""
    fi

    {
        echo "# ── run metadata (added by run_trials.sh) ──────────────────────"
        echo "# batch      : $BATCH_NAME/batch_$TS"
        echo "# controller : $label"
        echo "# run index  : $idx of $N_RUNS"
        echo "# trial      : $trial_id"
        echo "# command    : (cd controllers/build && $cmd)"
        echo "# started    : $(date -Is -d "@$((t0 / 1000000000))")"
        echo "# wall time  : ${wall_s}s"
        echo "# exit code  : $rc"
        echo "# git        : $GIT_REV$GIT_DIRTY"
        echo "# host       : $(hostname)"
        echo "# ──────────────────────────────────────────────────────────────"
        echo
        cat "$tmp"
    } > "$console_dest"
    rm -f "$tmp"

    echo "$label,$idx,$trial_id,${trial_dir#"$REPO_ROOT"/},$rc,$wall_s,$stand_h,$conv_ms,$avg_ms,$solves,$fell,$fall_t,$reached,$advances,$done_t,$rows,$t_end" >> "$SUMMARY"

    local note=""
    (( fell )) && note=" FELL@${fall_t}s"
    (( reached )) && note="$note GOAL"
    [[ -n "$done_t" ]] && note="$note stop@${done_t}s"
    (( rc != 0 )) && note="$note EXIT=$rc"
    say "$(printf '[%-6s %3d/%d] %-9s wall %6ss  avg solve %sms  t_end %s%s' \
        "$label" "$idx" "$N_RUNS" "$trial_id" "$wall_s" "${avg_ms:-?}" "${t_end:-?}" "$note")"
}

run_batch() {
    local label="$1" bin="$2" yaml="$3" csv_base="$4"
    local save="$BATCH_NAME/$label"
    say ""
    say "── $label: $N_RUNS × ./$bin $TASK --save $save ──"
    local i
    for (( i = 1; i <= N_RUNS; i++ )); do
        (( ABORT )) && { say "Aborted before ${label} run $i."; return; }
        run_one "$label" "$bin" "$yaml" "$save" "$i" "$csv_base"
    done
}

BATCH_START="$(date +%s)"
if (( RUN_MUSCLE && !ABORT )); then run_batch muscle mppi_sim    "$MUSCLE_YAML" mppi_sim;    fi
if (( RUN_PD     && !ABORT )); then run_batch pd     pd_mppi_sim "$PD_YAML"     pd_mppi_sim; fi
BATCH_END="$(date +%s)"

say ""
say "Finished $(date -Is) after $(( (BATCH_END - BATCH_START) / 60 ))m $(( (BATCH_END - BATCH_START) % 60 ))s."
say "Summary : $SUMMARY"
say "Log     : $BATCH_LOG"
say "Trials  : $TRIALS_DIR/$BATCH_NAME/{muscle,pd}/trial_NNN/console.log"
if compgen -G "$BATCH_DIR/failed/*.log" > /dev/null; then
    say "WARNING : $(ls "$BATCH_DIR/failed" | wc -l) run(s) produced no trial directory — see $BATCH_DIR/failed/"
else
    rmdir "$BATCH_DIR/failed" 2>/dev/null
fi
