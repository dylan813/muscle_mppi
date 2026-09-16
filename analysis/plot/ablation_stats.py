"""
Metrics and statistics for the walk component ablation (run_ablation_walk.sh).

Each condition folder holds muscle/trial_*/ directories written by
run_trials.sh (mppi_sim.csv, mppi_sim_qpos.csv, console.log) and, if it was run
by run_ablation_walk.sh, the tasks.yaml it ran with.

Per trial:
  success         the controller reported "[phase] task complete."
  fell            the sim reported "Robot fell at t=..."
  time_to_goal    sim time of "Task complete at t=..." (s)
  tau_mean        mean |tau| over every logged step and joint (N.m)
  tau_peak        largest single |tau| (N.m)
  over_limit      % of (step, joint) commands outside the actuator ctrlrange
  tau_rate_mean   mean |d tau / dt| over steps and joints (N.m/s)
  tau_rate_p95    95th percentile of |d tau / dt| (N.m/s)
  cocon_std       std over time of a1 + a2, averaged over the 12 joints

Torque is the commanded torque logged by mppi_sim (tau_j*), read through
torque_stats.trial_torque(). The continuous metrics are compared on successful
trials only; how many that is per condition is reported alongside.

Statistics, for every pair of conditions:
  success, fell          Fisher's exact test
  continuous metrics     Mann-Whitney U (two-sided)
with a Holm correction across the pairs within each metric.

Writes to --out (default figures/<trials folder name>/):
  per_trial.csv     one row per trial
  summary.csv       one row per condition (rates, medians and IQRs)
  pairwise.csv      one row per (metric, pair) with raw and Holm-corrected p
  interaction.png   median of each metric across the 2x2 (only for the four
                    run_ablation_walk.sh conditions)

Usage:
  python3 ablation_stats.py [trials_dir] [--conditions full no_actdyn ...] [--out DIR]

trials_dir defaults to analysis/log/trials/ablation_walk.
"""

import argparse
import glob
import itertools
import os
import re

import numpy as np
import pandas as pd
import yaml
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_box_height import FS_AXIS, FS_LEG, FS_TICK, frame, style_legend
from plot_leg_muscles import DEFAULT_YAML, _MODEL_BASE, load_run, model_joint_info
from torque_stats import find_trials, trial_torque

_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_JOINTS = 12
TASK = "walk"

ABLATION_CONDITIONS = ["full", "no_actdyn", "no_hill", "no_actdyn_no_hill"]

# metric -> (label, unit)
CONTINUOUS = {
    "time_to_goal":  ("Time to goal", "s"),
    "tau_mean":      ("Mean |τ|", "N·m"),
    "tau_peak":      ("Peak |τ|", "N·m"),
    "over_limit":    ("Over limit", "%"),
    "tau_rate_mean": ("Mean |dτ/dt|", "N·m/s"),
    "tau_rate_p95":  ("95th pct |dτ/dt|", "N·m/s"),
    "cocon_std":     ("std(a1 + a2)", ""),
}
BINARY = {"success": ("Success rate", "%"), "fell": ("Fall rate", "%")}

# Two series (activation dynamics on / off), validated categorical slots 1 and 2.
C_ACT_ON, C_ACT_OFF = "#2a78d6", "#eb6834"


def load_condition_task(cond_dir):
    """The walk task a condition ran with: its own tasks.yaml if present."""
    path = os.path.join(cond_dir, "tasks.yaml")
    if not os.path.isfile(path):
        path = DEFAULT_YAML
    with open(path) as f:
        return yaml.safe_load(f)[TASK]


def parse_console(path):
    """(success, fell, time_to_goal) from a run's console.log."""
    with open(path) as f:
        text = f.read()
    success = bool(re.search(r"^\[phase\] task complete\.", text, re.M))
    fell = bool(re.search(r"^Robot fell at t=", text, re.M))
    m = re.search(r"^Task complete at t=([0-9.]+)", text, re.M)
    return success, fell, float(m.group(1)) if m else np.nan


def trial_metrics(csv_path, task, qadr, ctrlrange):
    tau = trial_torque(csv_path, "muscle", task["muscle"], qadr)
    lo = np.array([ctrlrange[j][0] for j in range(NUM_JOINTS)])
    hi = np.array([ctrlrange[j][1] for j in range(NUM_JOINTS)])
    mag = np.abs(tau)
    rate = np.abs(np.diff(tau, axis=0)) / task["dt"]

    df, _ = load_run(csv_path)
    cocon = [(df[f"act_m{2 * j}"] + df[f"act_m{2 * j + 1}"]).std(ddof=1)
             for j in range(NUM_JOINTS)]

    return {
        "tau_mean":      mag.mean(),
        "tau_peak":      mag.max(),
        "over_limit":    100.0 * ((tau < lo) | (tau > hi)).mean(),
        "tau_rate_mean": rate.mean() if len(rate) else np.nan,
        "tau_rate_p95":  np.percentile(rate, 95) if len(rate) else np.nan,
        "cocon_std":     float(np.mean(cocon)),
    }


def collect(trials_dir, conditions):
    rows = []
    for cond in conditions:
        cond_dir = os.path.join(trials_dir, cond)
        task = load_condition_task(cond_dir)
        qadr, ctrlrange = model_joint_info(os.path.normpath(
            os.path.join(_MODEL_BASE, task["model_path"])))

        trials = find_trials(os.path.join(cond_dir, "muscle"))
        skipped = 0
        for name, csv_path in trials:
            console = os.path.join(os.path.dirname(csv_path), "console.log")
            if not os.path.isfile(console):
                skipped += 1
                continue
            success, fell, t_goal = parse_console(console)
            row = {"condition": cond, "trial": name,
                   "success": success, "fell": fell, "time_to_goal": t_goal}
            row.update(trial_metrics(csv_path, task, qadr, ctrlrange))
            rows.append(row)
        print(f"{cond:>20}: {len(trials) - skipped} trials"
              + (f" ({skipped} without console.log skipped)" if skipped else ""))
    return pd.DataFrame(rows)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (NaNs are left out and passed through)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    idx = np.where(~np.isnan(p))[0]
    order = idx[np.argsort(p[idx])]
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def pairwise(df, conditions):
    rows = []
    pairs = list(itertools.combinations(conditions, 2))
    for metric in list(BINARY) + list(CONTINUOUS):
        block = []
        for a, b in pairs:
            A, B = df[df.condition == a], df[df.condition == b]
            if metric in BINARY:
                ka, kb = int(A[metric].sum()), int(B[metric].sum())
                _, p = stats.fisher_exact([[ka, len(A) - ka], [kb, len(B) - kb]])
                test, stat = "fisher", np.nan
            else:
                xa = A.loc[A.success, metric].dropna()
                xb = B.loc[B.success, metric].dropna()
                if len(xa) and len(xb):
                    stat, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
                else:
                    stat, p = np.nan, np.nan
                test = "mannwhitneyu"
            block.append({"metric": metric, "a": a, "b": b, "test": test,
                          "stat": stat, "p": p})
        for r, ph in zip(block, holm([r["p"] for r in block])):
            r["p_holm"] = ph
        rows += block
    return pd.DataFrame(rows)


def summary(df, conditions):
    rows = []
    for cond in conditions:
        d = df[df.condition == cond]
        ok = d[d.success]
        row = {"condition": cond, "n": len(d), "n_success": len(ok),
               "success_pct": 100.0 * d.success.mean() if len(d) else np.nan,
               "fell_pct": 100.0 * d.fell.mean() if len(d) else np.nan}
        for metric in CONTINUOUS:
            x = ok[metric].dropna()
            row[f"{metric}_median"] = x.median() if len(x) else np.nan
            row[f"{metric}_q1"] = x.quantile(0.25) if len(x) else np.nan
            row[f"{metric}_q3"] = x.quantile(0.75) if len(x) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def interaction_plot(summ, path):
    """One panel per metric: x = Hill model on/off, one line per activation
    dynamics setting, median with IQR bars (rates for success/fall)."""
    s = summ.set_index("condition")
    cells = {(True, True): "full", (False, True): "no_actdyn",
             (True, False): "no_hill", (False, False): "no_actdyn_no_hill"}
    metrics = list(BINARY) + list(CONTINUOUS)
    ncol = 3
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow))
    x = np.array([0, 1])   # Hill on, Hill off

    for ax, metric in zip(axes.flat, metrics):
        label, unit = {**BINARY, **CONTINUOUS}[metric]
        # dx nudges the two series apart so their IQR bars don't overlap.
        for act, color, marker, ls, name, dx in ((True, C_ACT_ON, "o", "-", "act. dynamics on", -0.04),
                                                 (False, C_ACT_OFF, "s", "--", "act. dynamics off", 0.04)):
            conds = [cells[(act, True)], cells[(act, False)]]
            if metric in BINARY:
                y = np.array([s.loc[c, f"{metric}_pct"] for c in conds])
                err = None
            else:
                y = np.array([s.loc[c, f"{metric}_median"] for c in conds])
                err = np.array([y - [s.loc[c, f"{metric}_q1"] for c in conds],
                                [s.loc[c, f"{metric}_q3"] for c in conds] - y])
            ax.errorbar(x + dx, y, yerr=err, color=color, marker=marker, markersize=8,
                        linewidth=2, linestyle=ls, capsize=4, label=name)
        frame(ax)
        if metric in BINARY:
            ax.set_ylim(-5, 105)   # full 0-100 % scale, so small differences aren't magnified
        ax.set_xticks(x, ["Hill on", "Hill off"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(label, fontsize=FS_LEG + 2, color="0.1")
        if unit:
            ax.set_ylabel(unit, fontsize=FS_LEG + 1, color="0.3")
        ax.tick_params(labelsize=FS_LEG)

    for ax in axes.flat[len(metrics):]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    style_legend(fig, fig.legend(handles, labels, loc="upper center", ncol=2,
                                 fontsize=FS_LEG + 1, frameon=True))
    fig.text(0.5, 0.005, "Continuous metrics: median and IQR over successful trials",
             ha="center", fontsize=FS_LEG - 1, color="0.35")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trials_dir", nargs="?",
                    default=os.path.join(_DIR, "..", "log", "trials", "ablation_walk"))
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="condition subfolders, in order (default: the four ablation conditions "
                         "that exist under trials_dir)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    trials_dir = os.path.normpath(args.trials_dir)
    conditions = args.conditions or [c for c in ABLATION_CONDITIONS
                                     if os.path.isdir(os.path.join(trials_dir, c))]
    if not conditions:
        raise SystemExit(f"no condition folders under {trials_dir}")
    out = args.out or os.path.join(_DIR, "figures", os.path.basename(trials_dir))
    os.makedirs(out, exist_ok=True)

    print(f"Trials: {trials_dir}")
    df = collect(trials_dir, conditions)
    if df.empty:
        raise SystemExit("no trials found")

    summ = summary(df, conditions)
    pw = pairwise(df, conditions)
    df.to_csv(os.path.join(out, "per_trial.csv"), index=False)
    summ.to_csv(os.path.join(out, "summary.csv"), index=False)
    pw.to_csv(os.path.join(out, "pairwise.csv"), index=False)

    pd.set_option("display.width", 200)
    print("\n=== PER CONDITION (continuous: median [IQR] over successful trials) ===")
    table = pd.DataFrame({"n": summ.n.values, "success": summ.n_success.values,
                          "success %": summ.success_pct.round(1).values,
                          "fell %": summ.fell_pct.round(1).values},
                         index=summ.condition)
    for metric in CONTINUOUS:
        table[metric] = [f"{m:.3g} [{a:.3g}, {b:.3g}]" if not np.isnan(m) else "-"
                         for m, a, b in zip(summ[f"{metric}_median"],
                                            summ[f"{metric}_q1"], summ[f"{metric}_q3"])]
    print(table.T.to_string())

    print("\n=== PAIRWISE (Holm-corrected within each metric) ===")
    shown = pw.assign(sig=np.where(pw.p_holm < 0.05, "*", ""))
    print(shown[["metric", "a", "b", "test", "p", "p_holm", "sig"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    if sorted(conditions) == sorted(ABLATION_CONDITIONS):
        interaction_plot(summ, os.path.join(out, "interaction.png"))
        print(f"\nFigure: {os.path.join(out, 'interaction.png')}")
    print(f"Tables: {out}/{{per_trial,summary,pairwise}}.csv")


if __name__ == "__main__":
    main()
