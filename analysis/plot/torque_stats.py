"""
Commanded-torque statistics across a set of saved trials.

Aggregates what plot_leg_muscles.py plots. Muscle runs that log tau_j* (added
for the walk ablation study) use that logged torque directly; older muscle runs
and PD runs don't log torque, so theirs is reconstructed from the logged state. All of the
reconstruction — the Hill model, the PD law, the model's joint->qpos mapping,
and the one-row log offset — is imported from plot_leg_muscles.py rather than
reimplemented here, so the numbers below and that script's traces can't drift
apart.

Reported per controller, over all 12 joints:
  mean            mean |tau| over every logged step and joint
  peak            largest single |tau| commanded in the trial
  mean joint peak each joint's own peak |tau|, averaged over the 12 joints
  over limit      % of (step, joint) commands outside the actuator ctrlrange

"Commanded" is the raw controller output. MuJoCo clamps d->ctrl to the actuator
ctrlrange before applying it, so where "over limit" is non-zero the delivered
torque is smaller than the command; --clip reports the delivered torque instead.

Usage:
  python3 torque_stats.py [trials_dir] [--task guinea_fowl] [--clip]

trials_dir defaults to analysis/log/trials/workshop and must hold one subdirectory per
controller ("muscle", "pd"), each holding trial_*/ directories as written by the
sims' --save flag.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from plot_leg_muscles import (
    DEFAULT_PD_YAML,
    DEFAULT_YAML,
    JOINT_NAMES,
    LEG_OFFSET,
    _MODEL_BASE,
    load_run,
    load_task,
    model_joint_info,
    muscle_torques,
    pd_torque,
)

_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_JOINTS = 12

JOINT_LABELS = [f"{leg} {part}"
                for leg in sorted(LEG_OFFSET, key=LEG_OFFSET.get)
                for part in JOINT_NAMES]


def trial_torque(csv_path, kind, params, qadr):
    """Commanded torque for one trial as a (steps, 12) array.

    The row offset matches plot_leg_muscles.py: row k's logged command acted on
    row k-1's state, so state is taken from [:-1] and the command from [1:], and
    the first logged row is dropped (its predecessor was never written).
    """
    df, qpos = load_run(csv_path)

    # Runs logged since the ablation study carry the torque the sim actually
    # commanded (tau_j*, same row convention as act_m*/qdes_j*) -- muscle since
    # then, and pd since the control-rate sweep. Use it whenever it is there:
    # the Hill reconstruction below is wrong for ablated muscle models, and a
    # logged command needs no reconstruction for either controller. Older logs
    # without the column still fall through to the reconstruction.
    if "tau_j0" in df.columns:
        return np.column_stack([df[f"tau_j{j}"].to_numpy()[1:] for j in range(NUM_JOINTS)])

    tau = np.zeros((len(df) - 1, NUM_JOINTS))

    for j in range(NUM_JOINTS):
        q  = qpos[:-1, qadr[j]]
        dq = df[f"dq_j{j}"].to_numpy()[:-1]

        if kind == "muscle":
            a1 = df[f"act_m{2 * j}"].to_numpy()[1:]
            a2 = df[f"act_m{2 * j + 1}"].to_numpy()[1:]
            t1, t2 = muscle_torques(q, dq, a1, a2, params, j)
            tau[:, j] = t1 + t2
        else:
            qdes = df[f"qdes_j{j}"].to_numpy()[1:]
            tau[:, j] = pd_torque(q, dq, qdes, params, j)

    return tau


def find_trials(controller_dir):
    """(trial_name, csv_path) for each trial_*/ holding a sim CSV."""
    out = []
    for trial in sorted(glob.glob(os.path.join(controller_dir, "trial_*"))):
        csvs = [p for p in glob.glob(os.path.join(trial, "*_sim.csv"))
                if not p.endswith("_qpos.csv")]
        if csvs:
            out.append((os.path.basename(trial), csvs[0]))
    return out


def summarize(controller_dir, kind, params, qadr, ctrlrange, clip):
    """Per-trial totals plus the per-trial, per-joint arrays the joint-type
    breakdown is derived from — all shaped (n_trials, 12)."""
    rows, joint_means, joint_peaks, joint_over = [], [], [], []

    for name, csv_path in find_trials(controller_dir):
        tau = trial_torque(csv_path, kind, params, qadr)

        lo = np.array([ctrlrange[j][0] for j in range(NUM_JOINTS)])
        hi = np.array([ctrlrange[j][1] for j in range(NUM_JOINTS)])
        outside = (tau < lo) | (tau > hi)
        if clip:
            tau = np.clip(tau, lo, hi)

        mag = np.abs(tau)
        joint_means.append(mag.mean(axis=0))
        joint_peaks.append(mag.max(axis=0))
        joint_over.append(100.0 * outside.mean(axis=0))
        rows.append({
            "trial":      name,
            "steps":      len(tau),
            "mean":       mag.mean(),
            "peak":       mag.max(),
            "joint_peak": mag.max(axis=0).mean(),
            "over_limit": 100.0 * outside.mean(),
        })

    return (pd.DataFrame(rows), np.array(joint_means),
            np.array(joint_peaks), np.array(joint_over))


def by_joint_type(jmeans, jpeaks, jover):
    """Collapse the 4 legs into one row per joint type.

    Each leg contributes equally and every trial logs the same number of steps
    for all 12 joints, so averaging the 4 per-joint means is the same as
    pooling their samples. Statistics are formed per trial first, then reduced
    across trials, so the ± is spread between trials rather than between legs.
    """
    out = []
    for k, part in enumerate(JOINT_NAMES):
        cols = [k + off for off in sorted(LEG_OFFSET.values())]
        per_trial_mean = jmeans[:, cols].mean(axis=1)   # pooled over the 4 legs
        per_trial_peak = jpeaks[:, cols].max(axis=1)    # worst leg in that trial
        per_trial_jpk  = jpeaks[:, cols].mean(axis=1)   # per-joint peak, over legs
        out.append({
            "joint":      part,
            "mean":       per_trial_mean.mean(),
            "mean_sd":    per_trial_mean.std(ddof=1),
            "peak":       per_trial_peak.mean(),
            "peak_sd":    per_trial_peak.std(ddof=1),
            "joint_peak": per_trial_jpk.mean(),
            "jpk_sd":     per_trial_jpk.std(ddof=1),
            "over_limit": jover[:, cols].mean(),
        })
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trials_dir", nargs="?",
                    default=os.path.join(_DIR, "..", "log", "trials", "workshop"))
    ap.add_argument("--task", default="guinea_fowl")
    ap.add_argument("--clip", action="store_true",
                    help="clamp to the actuator ctrlrange, as MuJoCo does, "
                         "reporting delivered rather than commanded torque")
    args = ap.parse_args()

    # A task only has to exist in the YAML of a controller that has trials here
    # (e.g. the walk ablation tasks exist only in the muscle YAML).
    have = {kind: os.path.isdir(os.path.join(args.trials_dir, kind))
            for kind in ("muscle", "pd")}
    tasks = {"muscle": load_task(DEFAULT_YAML, args.task) if have["muscle"] else None,
             "pd":     load_task(DEFAULT_PD_YAML, args.task) if have["pd"] else None}
    if not any(have.values()):
        raise SystemExit(f"no muscle/ or pd/ directory under {args.trials_dir}")

    model_task = tasks["muscle"] or tasks["pd"]
    qadr, ctrlrange = model_joint_info(os.path.normpath(
        os.path.join(_MODEL_BASE, model_task["model_path"])))

    label = "delivered (clipped)" if args.clip else "commanded"
    print(f"Task: {args.task}   torque: {label}")
    print(f"Trials: {args.trials_dir}")
    print(f"Actuator limit: ±{ctrlrange[0][1]:.4g} N·m\n")

    stats = {}
    for kind, section in (("muscle", "muscle"), ("pd", "pd")):
        cdir = os.path.join(args.trials_dir, kind)
        if not have[kind]:
            print(f"(no {kind}/ directory — skipping)\n")
            continue

        df, jmeans, jpeaks, jover = summarize(
            cdir, kind, tasks[kind][section], qadr, ctrlrange, args.clip)
        types = by_joint_type(jmeans, jpeaks, jover)
        stats[kind] = (df, jmeans, jpeaks, types)

        n = len(df)
        print(f"=== {kind.upper()}  ({n} trials) ===")
        print(df.to_string(index=False, float_format=lambda v: f"{v:9.3f}"))
        print(f"\n  all joints — mean {df['mean'].mean():7.3f} "
              f"± {df['mean'].std(ddof=1):.3f}   "
              f"peak {df['peak'].mean():7.3f} ± {df['peak'].std(ddof=1):.3f} N·m")

        print(f"\n  by joint type (mean ± SD over {n} trials, N·m):")
        print(f"    {'joint':>6} | {'mean':>17} | {'peak':>17} | "
              f"{'per-joint peak':>17} | {'>lim':>6}")
        print("    " + "-" * 74)
        for _, r in types.iterrows():
            print(f"    {r['joint']:>6} | {r['mean']:8.3f} ± {r['mean_sd']:6.3f} | "
                  f"{r['peak']:8.3f} ± {r['peak_sd']:6.3f} | "
                  f"{r['joint_peak']:8.3f} ± {r['jpk_sd']:6.3f} | "
                  f"{r['over_limit']:5.2f}%")
        print()

    if len(stats) == 2:
        m, p = stats["muscle"][3], stats["pd"][3]
        md, pdf = stats["muscle"][0], stats["pd"][0]

        # How much higher PD is than muscle, with muscle as the baseline.
        def pct(pd_val, muscle_val):
            return 100.0 * (pd_val - muscle_val) / muscle_val

        print("=== PD RELATIVE TO MUSCLE (N·m, and % higher than muscle) ===")
        print(f"{'joint':>10} | {'muscle':>9} {'pd':>9} {'mean %':>9} "
              f"| {'muscle':>9} {'pd':>9} {'peak %':>9}")
        print("-" * 76)
        for k in range(len(JOINT_NAMES)):
            mr, pr = m.iloc[k], p.iloc[k]
            print(f"{mr['joint']:>10} | {mr['mean']:9.3f} {pr['mean']:9.3f} "
                  f"{pct(pr['mean'], mr['mean']):+8.1f}% | "
                  f"{mr['peak']:9.3f} {pr['peak']:9.3f} "
                  f"{pct(pr['peak'], mr['peak']):+8.1f}%")
        print("-" * 76)
        print(f"{'all joints':>10} | {md['mean'].mean():9.3f} {pdf['mean'].mean():9.3f} "
              f"{pct(pdf['mean'].mean(), md['mean'].mean()):+8.1f}% | "
              f"{md['peak'].mean():9.3f} {pdf['peak'].mean():9.3f} "
              f"{pct(pdf['peak'].mean(), md['peak'].mean()):+8.1f}%")

        print("\n=== PER-JOINT: PD RELATIVE TO MUSCLE (N·m, % higher than muscle) ===")
        print(f"{'joint':>9} | {'muscle':>9} {'pd':>9} {'mean %':>9} "
              f"| {'muscle':>9} {'pd':>9} {'peak %':>9}")
        print("-" * 75)
        mm, mp = stats["muscle"][1].mean(axis=0), stats["muscle"][2].mean(axis=0)
        pm, pp = stats["pd"][1].mean(axis=0),     stats["pd"][2].mean(axis=0)
        for j in range(NUM_JOINTS):
            print(f"{JOINT_LABELS[j]:>9} | {mm[j]:9.3f} {pm[j]:9.3f} "
                  f"{pct(pm[j], mm[j]):+8.1f}% | "
                  f"{mp[j]:9.3f} {pp[j]:9.3f} {pct(pp[j], mp[j]):+8.1f}%")


if __name__ == "__main__":
    main()
