"""
Pick one representative rollout per (condition, outcome) for the ablation studies.

Each condition folder holds muscle/trial_*/ directories written by run_trials.sh
(mppi_sim.csv, mppi_sim_qpos.csv, console.log) and the tasks.yaml it ran with.

Outcome, from console.log — every trial is exactly one of:
  success   "[phase] task complete."
  fall      "Robot fell at t=..."
  timeout   neither: still upright when the sim_duration clock ran out

Trials are grouped by (condition, outcome) before anything is chosen. Within one
condition the outcome classes differ in torque — no_passive's successes average
~33% more |tau| than its timeouts — so a single pick per condition would let the
dominant class speak for the rest.

Selection. For each class, mean |tau| is taken per joint type (hip, thigh, calf),
pooling the 4 legs, giving each trial a 3-vector. The class target is the mean of
those vectors over the trials in the class, and the representative is the trial
whose vector is nearest the target in Euclidean distance (N.m):

    target_p = mean_i( tau_p(i) )                     p in {hip, thigh, calf}
    pick     = argmin_i sqrt( sum_p ( tau_p(i) - target_p )^2 )

Distances are in raw N.m with no per-feature scaling, so the calf — whose mean
|tau| runs 2-3x the hip's — carries the most weight in the choice.

Other torque metrics are reported for the pick but take no part in choosing it:
  tau_mean        mean |tau| over every logged step and joint (N.m)
  tau_peak        largest single |tau| (N.m)
  tau_rate_mean   mean |d tau / dt| over steps and joints (N.m/s)
  tau_rate_p95    95th percentile of |d tau / dt| (N.m/s)
  over_limit      % of (step, joint) commands outside the actuator ctrlrange

Torque is the commanded torque logged by mppi_sim (tau_j*), read through
torque_stats.trial_torque() — not a Hill reconstruction, which would be wrong
for the ablated models.

Writes to --out (default figures/representatives/):
  per_trial.csv         one row per trial: condition, outcome, metrics
  representatives.csv   one row per (condition, outcome): the pick, the class
                        target it was matched against, and its distance

Usage:
  python3 select_representatives.py [trials_dir ...] [--out DIR] [--min-n N]

trials_dir defaults to analysis/log/trials/ablation_walk and
analysis/log/trials/ablation_hill; each is scanned for condition subfolders.
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import yaml

from plot_leg_muscles import (
    DEFAULT_PD_YAML,
    DEFAULT_YAML,
    JOINT_NAMES,
    LEG_OFFSET,
    _MODEL_BASE,
    model_joint_info,
)
from torque_stats import find_trials, trial_torque

_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_JOINTS = 12
TASK = "walk"

OUTCOMES = ["success", "fall", "timeout"]

# Controller subfolder -> (the YAML the batch ran with, the task's params key).
# The ablation studies hold muscle/ only; the control-rate sweep holds both.
CONTROLLERS = {"muscle": ("tasks.yaml", "muscle", DEFAULT_YAML),
               "pd":     ("tasks_pd.yaml", "pd", DEFAULT_PD_YAML)}

# The 3 values the representative is matched on: mean |tau| per joint type.
SELECT_ON = [f"{part}_mean" for part in JOINT_NAMES]

# Reported alongside, but not used to choose.
REPORTED = {
    "tau_mean":      "mean |τ|",
    "tau_peak":      "peak |τ|",
    "tau_rate_mean": "mean |dτ/dt|",
    "tau_rate_p95":  "p95 |dτ/dt|",
    "over_limit":    "over limit",
}


def load_condition_task(cond_dir, controller="muscle"):
    """The walk task a condition ran with: its own YAML if the batch kept one."""
    fname, _, fallback = CONTROLLERS[controller]
    path = os.path.join(cond_dir, fname)
    if not os.path.isfile(path):
        path = fallback
    with open(path) as f:
        return yaml.safe_load(f)[TASK]


def parse_outcome(path):
    """(outcome, event_time) from a run's console.log.

    A run stops on the first of: falling, reaching the final phase, or the
    sim_duration clock. The three are mutually exclusive in the log.
    """
    with open(path) as f:
        text = f.read()
    if re.search(r"^\[phase\] task complete\.", text, re.M):
        m = re.search(r"^Task complete at t=([0-9.]+)", text, re.M)
        return "success", float(m.group(1)) if m else np.nan
    if re.search(r"^Robot fell at t=", text, re.M):
        m = re.search(r"^Robot fell at t=([0-9.]+)", text, re.M)
        return "fall", float(m.group(1)) if m else np.nan
    return "timeout", np.nan


def trial_metrics(csv_path, task, qadr, ctrlrange, controller="muscle"):
    params_key = CONTROLLERS[controller][1]
    tau = trial_torque(csv_path, controller, task.get(params_key), qadr)
    lo = np.array([ctrlrange[j][0] for j in range(NUM_JOINTS)])
    hi = np.array([ctrlrange[j][1] for j in range(NUM_JOINTS)])

    mag = np.abs(tau)
    rate = np.abs(np.diff(tau, axis=0)) / task["dt"]
    joint_mean = mag.mean(axis=0)   # (12,)

    out = {
        "tau_mean":      mag.mean(),
        "tau_peak":      mag.max(),
        "tau_rate_mean": rate.mean() if len(rate) else np.nan,
        "tau_rate_p95":  np.percentile(rate, 95) if len(rate) else np.nan,
        "over_limit":    100.0 * ((tau < lo) | (tau > hi)).mean(),
    }
    # Pool the 4 legs per joint type: every trial logs the same number of steps
    # for all 12 joints, so averaging the 4 per-joint means pools their samples.
    for k, part in enumerate(JOINT_NAMES):
        cols = [k + off for off in sorted(LEG_OFFSET.values())]
        out[f"{part}_mean"] = joint_mean[cols].mean()
    out["steps"] = len(tau)
    return out


def collect(trials_dir, conditions):
    rows = []
    for cond in conditions:
        cond_dir = os.path.join(trials_dir, cond)
        for controller in CONTROLLERS:
            ctrl_dir = os.path.join(cond_dir, controller)
            if not os.path.isdir(ctrl_dir):
                continue
            task = load_condition_task(cond_dir, controller)
            qadr, ctrlrange = model_joint_info(os.path.normpath(
                os.path.join(_MODEL_BASE, task["model_path"])))

            trials = find_trials(ctrl_dir)
            skipped = 0
            for name, csv_path in trials:
                console = os.path.join(os.path.dirname(csv_path), "console.log")
                if not os.path.isfile(console):
                    skipped += 1
                    continue
                outcome, t_event = parse_outcome(console)
                df = pd.read_csv(csv_path)
                row = {
                    "study":      os.path.basename(trials_dir),
                    "condition":  cond,
                    "controller": controller,
                    "trial":      name,
                    "outcome":    outcome,
                    "t_event":    t_event,
                    "t_end":      df["t"].iloc[-1],
                    # px means whole-robot CoM for muscle and trunk origin for
                    # pd -- each matches its own cost. Kept only as context; any
                    # cross-controller position comparison must use the qpos log.
                    "final_px":   df["px"].iloc[-1],
                    "csv":        os.path.relpath(csv_path, _MODEL_BASE),
                }
                row.update(trial_metrics(csv_path, task, qadr, ctrlrange, controller))
                rows.append(row)
            counts = pd.Series([r["outcome"] for r in rows
                                if r["condition"] == cond
                                and r["controller"] == controller]).value_counts()
            detail = "  ".join(f"{o}={counts.get(o, 0)}" for o in OUTCOMES)
            label = f"{cond}/{controller}" if len(CONTROLLERS) > 1 else cond
            print(f"  {label:<24} {len(trials) - skipped:>4} trials   {detail}"
                  + (f"   ({skipped} without console.log skipped)" if skipped else ""))
    return pd.DataFrame(rows)


def pick_representatives(df, min_n):
    """The trial nearest its class's mean per-joint-type torque vector."""
    rows = []
    keys = ["study", "condition", "controller", "outcome"]
    for (study, cond, controller, outcome), g in df.groupby(keys, sort=False):
        g = g.reset_index(drop=True)
        vecs = g[SELECT_ON].to_numpy(dtype=float)        # (n_trials, 3)
        target = vecs.mean(axis=0)
        dist = np.linalg.norm(vecs - target, axis=1)
        order = np.argsort(dist)
        i = int(order[0])
        pick = g.loc[i]

        # runner_up shows how much the choice was worth: near-equal distances
        # mean either trial would serve.
        j = int(order[1]) if len(order) > 1 else i
        row = {
            "study": study, "condition": cond, "controller": controller,
            "outcome": outcome,
            "n": len(g), "share_pct": np.nan,     # filled in below
            "thin": len(g) < min_n,
            "trial": pick["trial"], "csv": pick["csv"],
            "dist": dist[i],
            "runner_up": g.loc[j, "trial"], "runner_up_dist": dist[j],
            "t_event": pick["t_event"], "t_end": pick["t_end"],
            "final_px": pick["final_px"], "steps": pick["steps"],
        }
        for k, col in enumerate(SELECT_ON):
            row[col] = pick[col]
            row[f"{col}_target"] = target[k]
        for col in REPORTED:
            row[col] = pick[col]
        rows.append(row)

    out = pd.DataFrame(rows)
    # share_pct is the class's share of its own (condition, controller) cell.
    totals = df.groupby(["study", "condition", "controller"]).size()
    out["share_pct"] = [100.0 * n / totals[(s, c, k)] for s, c, k, n
                        in zip(out.study, out.condition, out.controller, out.n)]
    # Conditions in the order they were collected, outcomes in a fixed order.
    order = {c: i for i, c in enumerate(df.condition.unique())}
    rank = {o: i for i, o in enumerate(OUTCOMES)}
    return out.sort_values(
        ["condition", "controller", "outcome"],
        key=lambda s: (s.map(order) if s.name == "condition"
                       else s.map(rank) if s.name == "outcome" else s)
    ).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trials_dirs", nargs="*", default=None,
                    help="study folders to scan (default: ablation_walk and ablation_hill)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-n", type=int, default=10,
                    help="classes with fewer trials than this are flagged 'thin' "
                         "in the output; they are still picked (default: 10)")
    args = ap.parse_args()

    base = os.path.join(_DIR, "..", "log", "trials")
    dirs = args.trials_dirs or [os.path.join(base, "ablation_walk"),
                                os.path.join(base, "ablation_hill")]
    out = args.out or os.path.join(_DIR, "figures", "representatives")
    os.makedirs(out, exist_ok=True)

    frames = []
    for d in dirs:
        d = os.path.normpath(d)
        if not os.path.isdir(d):
            raise SystemExit(f"no such trials folder: {d}")
        conditions = sorted(c for c in os.listdir(d)
                            if any(os.path.isdir(os.path.join(d, c, k))
                                   for k in CONTROLLERS))
        if not conditions:
            raise SystemExit(f"no condition folders with muscle/ or pd/ under {d}")
        print(f"{os.path.basename(d)}:")
        frames.append(collect(d, conditions))
    df = pd.concat(frames, ignore_index=True)

    reps = pick_representatives(df, args.min_n)
    df.to_csv(os.path.join(out, "per_trial.csv"), index=False)
    reps.to_csv(os.path.join(out, "representatives.csv"), index=False)

    pd.set_option("display.width", 220)
    print("\n=== REPRESENTATIVES (nearest the class mean of per-joint-type mean |τ|) ===")
    show = pd.DataFrame({
        "condition": reps.condition,
        "controller": reps.controller,
        "outcome":   reps.outcome,
        "n":         reps.n,
        "share":     [f"{v:.0f}%" for v in reps.share_pct],
        "trial":     reps.trial,
        "t_end":     reps.t_end.round(2),
        "dist":      reps.dist.round(3),
    })
    for part in JOINT_NAMES:
        col = f"{part}_mean"
        show[f"{part} |τ| (target)"] = [f"{v:.2f} ({t:.2f})"
                                        for v, t in zip(reps[col], reps[f"{col}_target"])]
    for col, label in REPORTED.items():
        show[label] = reps[col].round(3 if col == "over_limit" else 1)
    show["thin"] = np.where(reps.thin, "*", "")
    print(show.to_string(index=False))

    thin = reps[reps.thin]
    if len(thin):
        print(f"\n* {len(thin)} class(es) under {args.min_n} trials — a pick from so few "
              "is a sample, not a typical case:")
        for _, r in thin.iterrows():
            print(f"    {r.condition:<20} {r.outcome:<8} n={r.n}")

    print(f"\nTables: {out}/{{per_trial,representatives}}.csv")


if __name__ == "__main__":
    main()
