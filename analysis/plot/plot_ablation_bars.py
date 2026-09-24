"""
Ablation summary figures: task success and commanded torque per condition.

The same three figures plot_box_height.py draws for the box-height sweep, with
the ablation condition on the x axis in place of platform height (written to
figures/):

    fig_ablation_success.png        task success rate per condition
    fig_ablation_torque_mean.png    mean |commanded torque| per condition
    fig_ablation_torque_peak.png    peak |commanded torque| per condition

Outcome is binary here: a trial succeeds if it prints "[phase] task complete.",
and everything else is a failure -- both falling and running the sim_duration
clock out while still upright. Timeouts are failures to walk, not a third
category, so fl_only (which never fell and never arrived in 100 trials) reads
0%, the same as a condition that falls every time.

The box-height figures carry two factors, joint type and controller, because
they compare muscle against PD. These studies ran muscle only, so identity
rests on joint type alone and the bars use plot_box_height's muscle ramp.

Torque is pooled over every trial of a condition regardless of outcome -- the
same as plot_box_height, and deliberate: conditioning on success would compare
conditions on very different subsets (fl_only has no successes at all, so it
would vanish from the torque figures entirely). Each trial contributes one mean
and one peak, so a 2 s fall and a 20 s timeout count equally despite logging
very different numbers of steps.

Statistics come from torque_stats.summarize(), which plot_box_height also uses,
so these figures and that script's tables cannot drift apart. Torque is the
commanded torque logged by mppi_sim (tau_j*), read through
torque_stats.trial_torque() -- not a Hill reconstruction, which would be wrong
for the ablated models.

Usage:
  python3 plot_ablation_bars.py [trials_root] [--clip]
"""

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_box_height import (
    BAR_COLOR,
    FS_AXIS,
    FS_LEG,
    JOINT_LABEL,
    frame,
    joint_cols,
    style_legend,
)
from plot_condition_torque import DISPLAY_NAME
from plot_leg_muscles import C_MUSCLE, JOINT_NAMES, _MODEL_BASE, model_joint_info
from select_representatives import load_condition_task, parse_outcome
from torque_stats import summarize

_DIR = os.path.dirname(os.path.abspath(__file__))

# x order: the 2x2 activation-dynamics study, then the 2^3 Hill sub-component
# study. A divider is drawn between them rather than colouring by study, which
# would add a second encoding to a figure that only has one factor.
ORDER = [
    ("ablation_walk", "full"),
    ("ablation_walk", "no_actdyn"),
    ("ablation_walk", "no_hill"),
    ("ablation_walk", "no_actdyn_no_hill"),
    ("ablation_hill", "no_fl"),
    ("ablation_hill", "no_fv"),
    ("ablation_hill", "no_passive"),
    ("ablation_hill", "fl_only"),
    ("ablation_hill", "fv_only"),
    ("ablation_hill", "passive_only"),
]
STUDY_LABEL = {"ablation_walk": "activation dynamics × Hill",
               "ablation_hill": "Hill sub-components"}

FIGSIZE = (10.4, 4.4)


def outcomes(muscle_dir):
    """(n_success, n_fall, n_timeout, n) over one condition's trials."""
    counts = {"success": 0, "fall": 0, "timeout": 0}
    logs = sorted(glob.glob(os.path.join(muscle_dir, "trial_*", "console.log")))
    for log in logs:
        counts[parse_outcome(log)[0]] += 1
    return counts["success"], counts["fall"], counts["timeout"], len(logs)


def collect(trials_root, clip):
    """{(study, condition): per-trial stat arrays and outcome counts}."""
    out, ctrlrange = {}, None
    for study, cond in ORDER:
        cond_dir = os.path.join(trials_root, study, cond)
        muscle_dir = os.path.join(cond_dir, "muscle")
        if not os.path.isdir(muscle_dir):
            raise SystemExit(f"missing directory: {muscle_dir}")

        task = load_condition_task(cond_dir)
        qadr, ctrlrange = model_joint_info(os.path.normpath(
            os.path.join(_MODEL_BASE, task["model_path"])))
        df, jmeans, jpeaks, _ = summarize(
            muscle_dir, "muscle", task["muscle"], qadr, ctrlrange, clip)
        k, fell, timed, n = outcomes(muscle_dir)

        out[(study, cond)] = dict(jmeans=jmeans, jpeaks=jpeaks, reached=k, n=n)
        print(f"  {cond:<20} {n:3} trials   success {k:3}/{n}"
              f"   (fail: {fell} fell, {timed} timed out)")
    return out, ctrlrange


def series(data, key, stat):
    """(mean, sd) across trials for one joint type, per condition in ORDER."""
    mu, sd = [], []
    for spec in ORDER:
        arr = data[spec]["jmeans"] if stat == "mean" else data[spec]["jpeaks"]
        cols = joint_cols(JOINT_NAMES.index(key))
        # Per trial first, then across trials, so the SD is between-trial spread
        # rather than between-leg spread (matches torque_stats.py).
        v = arr[:, cols].mean(axis=1) if stat == "mean" else arr[:, cols].max(axis=1)
        mu.append(v.mean())
        sd.append(v.std(ddof=1))
    return np.array(mu), np.array(sd)


def xaxis(ax, pos):
    """Condition ticks, plus the divider and headings between the two studies."""
    labels = [DISPLAY_NAME.get(c, c) for _, c in ORDER]
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=FS_LEG)
    ax.set_xlim(pos[0] - 0.5, pos[-1] + 0.5)

    studies = [s for s, _ in ORDER]
    cut = studies.index("ablation_hill")
    ax.axvline(pos[cut] - 0.5, color="0.55", linewidth=0.9,
               linestyle=(0, (4, 3)), zorder=1)
    for study, lo, hi in (("ablation_walk", 0, cut), ("ablation_hill", cut, len(ORDER))):
        ax.annotate(STUDY_LABEL[study],
                    xy=(0.5 * (pos[lo] + pos[hi - 1]), 1.0), xytext=(0, 5),
                    xycoords=("data", "axes fraction"), textcoords="offset points",
                    ha="center", va="bottom", fontsize=FS_LEG, color="0.4")


def plot_success(data, out_stem):
    """One bar per condition: a single series, so no legend -- the y label names it."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    pos = np.arange(len(ORDER), dtype=float)
    pct = np.array([100.0 * data[s]["reached"] / data[s]["n"] for s in ORDER])

    ax.bar(pos, pct, width=0.62, color=C_MUSCLE, edgecolor="0.2",
           linewidth=0.6, zorder=2)
    # A bar at 0 is indistinguishable from a missing bar, and four conditions
    # sit there, so the value is written above every bar.
    for x, v in zip(pos, pct):
        ax.annotate(f"{v:.0f}", xy=(x, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=FS_LEG, color="0.25")

    frame(ax)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Task success rate (%)", fontsize=FS_AXIS)
    xaxis(ax, pos)
    fig.tight_layout()
    save(fig, out_stem)


def plot_torque(data, stat, out_stem, clip):
    """Three bars per condition, one per joint type."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    pos = np.arange(len(ORDER), dtype=float)

    bw, gap = 0.22, 0.02
    span = 3 * bw + 2 * gap
    top = 0.0
    for ji, key in enumerate(JOINT_NAMES):
        mu, sd = series(data, key, stat)
        top = max(top, (mu + sd).max())
        off = -span / 2 + ji * (bw + gap) + bw / 2
        ax.bar(pos + off, mu, width=bw, color=BAR_COLOR["muscle"][key],
               edgecolor="0.2", linewidth=0.6, zorder=2,
               yerr=sd, error_kw=dict(ecolor="0.15", elinewidth=1.0,
                                      capsize=2.6, capthick=1.0, zorder=3))

    frame(ax)
    # Headroom for the in-plot legend rather than letting it sit on the data.
    ax.set_ylim(0, top * 1.34)
    word = "Delivered" if clip else "Commanded"
    sym = (r"\overline{\tau}" if stat == "mean"
           else r"\overline{\tau}_{\mathrm{peak}}")
    ax.set_ylabel(f"{word} ${sym}$ (N·m)", fontsize=FS_AXIS)
    xaxis(ax, pos)

    handles = [Patch(facecolor=BAR_COLOR["muscle"][k], edgecolor="0.2",
                     linewidth=0.6, label=JOINT_LABEL[k]) for k in JOINT_NAMES]
    style_legend(ax, ax.legend(handles=handles, loc="upper left", ncol=3,
                               fontsize=FS_LEG, frameon=True, handlelength=1.5,
                               handleheight=1.0, columnspacing=1.4))
    fig.tight_layout()
    save(fig, out_stem)


def save(fig, stem):
    os.makedirs(os.path.join(_DIR, "figures"), exist_ok=True)
    fig.savefig(os.path.join(_DIR, "figures", f"{stem}.png"), dpi=200)
    plt.close(fig)
    print(f"  wrote figures/{stem}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trials_root", nargs="?",
                    default=os.path.join(_DIR, "..", "log", "trials"))
    ap.add_argument("--clip", action="store_true",
                    help="clamp to the actuator ctrlrange, as MuJoCo does, "
                         "reporting delivered rather than commanded torque")
    args = ap.parse_args()

    data, _ = collect(os.path.normpath(args.trials_root), args.clip)
    suffix = "_clipped" if args.clip else ""
    print()
    plot_success(data, "fig_ablation_success")
    plot_torque(data, "mean", f"fig_ablation_torque_mean{suffix}", args.clip)
    plot_torque(data, "peak", f"fig_ablation_torque_peak{suffix}", args.clip)


if __name__ == "__main__":
    main()
