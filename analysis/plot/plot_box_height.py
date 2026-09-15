"""
Box-height sweep figures: task success and commanded torque vs platform height.

The guinea_fowl task is a gap crossing between two platforms. Raising the
platforms (0.05 -> 0.08 m, matched by spawn_height_offset) makes it harder, and
100 trials of each controller were run at each height. This draws the three
figures from that sweep (written to figures/):

    fig_success.png             task success rate vs box height
    fig_torque_mean.png         mean |commanded torque| vs box height
    fig_torque_peak.png         peak |commanded torque| vs box height

Success comes from each run's own "[phase] task complete." line -- the
controller's signal, which both variants emit -- rather than either sim's
early-stop print, which mppi_sim.cpp only gained later (see run_trials.sh).

Torque is reconstructed from logged state by torque_stats.py, which in turn
imports the Hill model and PD law from plot_leg_muscles.py, so these figures,
that script's tables, and plot_leg_muscles.py's traces cannot drift apart.

Expects one directory per (height, controller) under analysis/log/trials/workshop, named
<height><controller> -- e.g. 5muscle, 5pd, 6muscle, ... -- each holding the
trial_*/ directories written by the sims' --save flag.

Usage:
  python3 plot_box_height.py [trials_dir] [--heights 5 6 7 8] [--clip]
"""

import argparse
import glob
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from plot_leg_muscles import (
    C_MUSCLE,
    C_PD,
    DEFAULT_PD_YAML,
    DEFAULT_YAML,
    JOINT_NAMES,
    LEG_OFFSET,
    _MODEL_BASE,
    load_task,
    model_joint_info,
)
from torque_stats import summarize

_DIR = os.path.dirname(os.path.abspath(__file__))

# Secondary encoding, required because the muscle/PD pair sits in the 6-8 dE
# band under protanopia (7.8 -- see plot_leg_muscles.py): identity must not rest
# on hue alone, so each series also carries its own marker and dash pattern.
STYLE = {
    "muscle": dict(color=C_MUSCLE, marker="o", linestyle="-",  label="Muscle"),
    "pd":     dict(color=C_PD,     marker="s", linestyle="--", label="PD"),
}

# Grouped bars carry two factors: hue family = controller, lightness within the
# family = joint, proximal to distal. The middle step of each ramp is the
# canonical C_MUSCLE / C_PD, so these bars still sit in the same green and
# orange as fig_success and torque_comparison.png.
#
# Each ramp is validated as an ordinal scale: monotone lightness, adjacent
# dL >= 0.06, light end clear of the 2:1 contrast floor, single hue (spread
# <= 10 deg). The two mid tones are at CVD dE 10.3, well clear of the floor.
BAR_COLOR = {
    "muscle": {"hip": "#4fbb92", "thigh": "#0f8a5f", "calf": "#075138"},
    "pd":     {"hip": "#e8974b", "thigh": "#d1620a", "calf": "#7d3a06"},
}

# Display names. plot_leg_muscles.py's JOINT_NAMES follow the URDF link names,
# which are not what the joints anatomically are -- go2.xml gives each its
# class: hip_joint is class="abduction", thigh_joint is class="front_hip",
# calf_joint is class="knee". Relabelled here to match the model, so the bars
# read ab-ad, hip, knee proximal to distal.
JOINT_LABEL = {"hip": "hip ab-ad", "thigh": "hip flex-ext", "calf": "knee"}

# Bar order within a group. PD first, then muscle.
KINDS = ("pd", "muscle")
# One type scale and one canvas size shared by all three figures. The size
# is fixed rather than cropped to content (savefig without bbox_inches=
# "tight"), so every PNG comes out the same pixel dimensions regardless of
# how wide its labels or legend happen to be.
FS_AXIS, FS_TICK, FS_LEG = 17, 15, 11
FIGSIZE = (5.6, 4.1)
GRID = dict(color="0.9", linewidth=0.6)
FIG_W = 6.8


def wilson(k, n, z=1.96):
    """Wilson score interval -- behaves at the 90%/2% ends where normal-
    approximation bars would run past 100% or below 0.

    Not drawn on fig_success any more (the bars are plain), but kept: it is the
    interval quoted in the text, and putting the error bars back is a one-line
    change at the ax.bar call below."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def success_rate(d):
    """(reached, n) over the trials in one directory."""
    logs = sorted(glob.glob(os.path.join(d, "trial_*", "console.log")))
    reached = sum(
        any(line.startswith("[phase] task complete.") for line in open(L, errors="replace"))
        for L in logs
    )
    return reached, len(logs)


def collect(trials_dir, heights, clip):
    """{(height, controller): dict of per-trial stat arrays}, plus success."""
    muscle_task = load_task(DEFAULT_YAML, "guinea_fowl")
    pd_task = load_task(DEFAULT_PD_YAML, "guinea_fowl")
    qadr, ctrlrange = model_joint_info(os.path.normpath(
        os.path.join(_MODEL_BASE, muscle_task["model_path"])))

    params = {"muscle": muscle_task["muscle"], "pd": pd_task["pd"]}
    out = {}
    for h in heights:
        for kind in ("muscle", "pd"):
            d = os.path.join(trials_dir, f"{h}{kind}")
            if not os.path.isdir(d):
                raise SystemExit(f"missing directory: {d}")
            df, jmeans, jpeaks, _ = summarize(
                d, kind, params[kind], qadr, ctrlrange, clip)
            k, n = success_rate(d)
            out[(h, kind)] = dict(jmeans=jmeans, jpeaks=jpeaks,
                                  all_mean=df["mean"].to_numpy(),
                                  all_peak=df["peak"].to_numpy(),
                                  reached=k, n=n)
            print(f"  {h}{kind:6}  {n:3} trials  success {k:3}/{n}")
    return out, ctrlrange


def joint_cols(k):
    """Column indices of one joint type across the 4 legs."""
    return [k + off for off in sorted(LEG_OFFSET.values())]


def series(data, heights, kind, key, stat):
    """(mean, sd) over trials at each height, for one joint type or all."""
    mu, sd = [], []
    for h in heights:
        d = data[(h, kind)]
        if key == "all":
            v = d["all_mean"] if stat == "mean" else d["all_peak"]
        else:
            cols = joint_cols(JOINT_NAMES.index(key))
            arr = d["jmeans"] if stat == "mean" else d["jpeaks"]
            # Per trial first, then across trials, so the SD is between-trial
            # spread rather than between-leg spread (matches torque_stats.py).
            v = arr[:, cols].mean(axis=1) if stat == "mean" else arr[:, cols].max(axis=1)
        mu.append(v.mean())
        sd.append(v.std(ddof=1))
    return np.array(mu), np.array(sd)


def frame(ax):
    """Full thin box with light grid, matching torque_comparison.png."""
    for s in ax.spines.values():
        s.set_linewidth(0.8)
        s.set_color("0.2")
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=FS_TICK, length=3, width=0.8, color="0.2")


def style_legend(ax, leg):
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_facecolor("white")
    return leg


def two_factor_legend(ax):
    """Six swatches, a column per controller and a row per joint, so the hue
    family and the lightness step are each read off one axis of the grid."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles, labels = [], []
    for kind in KINDS:
        # A headed column instead of repeating the controller on every row:
        # "Muscle hip flex-ext" x6 is what forces a wide canvas, and a wide
        # canvas is what makes the type look small.
        handles.append(Line2D([], [], linestyle="none"))
        labels.append(STYLE[kind]["label"])
        for j in JOINT_NAMES:
            handles.append(Patch(facecolor=BAR_COLOR[kind][j],
                                 edgecolor="0.2", linewidth=0.6))
            labels.append(JOINT_LABEL[j])
    leg = style_legend(ax, ax.legend(
        handles, labels, loc="upper right", ncol=2, fontsize=FS_LEG,
        frameon=True, handlelength=1.5, handleheight=1.1, columnspacing=1.4,
        borderpad=0.5, labelspacing=0.4, handletextpad=0.6))
    return leg


def plot_success(data, heights, xs, out_stem):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    pos = np.arange(len(xs), dtype=float)   # categorical, matching the torque figures
    for kind in KINDS:
        p = np.array([100 * data[(h, kind)]["reached"] / data[(h, kind)]["n"]
                      for h in heights])
        bw = 0.34
        off = (-0.5 if kind == KINDS[0] else 0.5) * bw
        ax.bar(pos + off, p, width=bw, color=STYLE[kind]["color"],
               edgecolor="0.2", linewidth=0.6, zorder=2,
               label=STYLE[kind]["label"])

    frame(ax)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{v:.2f}" for v in xs])
    ax.set_xlabel("Box height (m)", fontsize=FS_AXIS)
    ax.set_ylabel("Task success rate (%)", fontsize=FS_AXIS)
    ax.set_ylim(0, 100)
    ax.set_xlim(pos[0] - 0.5, pos[-1] + 0.5)
    style_legend(ax, ax.legend(loc="upper right", fontsize=FS_LEG, frameon=True,
                               handlelength=1.5, handleheight=1.0))
    fig.tight_layout()
    save(fig, out_stem)


def plot_torque(data, heights, xs, stat, ctrlrange, out_stem, clip):
    """All three joint types and both controllers on one axes.

    Six series, so identity rides on two encodings at once: hue = joint,
    line style + marker fill = controller.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Two devices for the crowded bands, in place of transparency (which would
    # cost contrast and blend crossing lines into a third colour):
    #  * dodge -- the x values are 4 discrete heights, not a continuum, so each
    #    joint gets a small offset. Stacked error bars stop colliding and the
    #    muscle/PD pair for one joint stays vertically comparable.
    #  * halo -- a white stroke under each line, so wherever two cross, one
    #    reads as passing in front of the other instead of merging.
    # Positions are categorical (4 discrete heights), not metric, so a dodge is
    # a within-group offset rather than a claim about the x value -- the same
    # convention a grouped bar chart uses.
    pos = np.arange(len(xs), dtype=float)
    # Six bars per height: the muscle/PD pair for a joint sits adjacent, with a
    # wider gap between joints, so the comparison the figure is about (same
    # joint, two controllers) is the one physically side by side.
    # PD and muscle touch within a joint (their shared edge is the only divider
    # they need), joints are separated by a small gap, and the group as a whole
    # is kept well inside its slot so the platform heights read as distinct
    # blocks rather than one continuous run of bars.
    bw, joint_gap = 0.105, 0.045
    pair_w = 2 * bw
    span = 3 * pair_w + 2 * joint_gap
    top = 0.0
    for ji, key in enumerate(JOINT_NAMES):
        for ci, kind in enumerate(KINDS):
            mu, sd = series(data, heights, kind, key, stat)
            top = max(top, (mu + sd).max())
            off = -span / 2 + ji * (pair_w + joint_gap) + ci * bw + bw / 2
            ax.bar(pos + off, mu, width=bw, color=BAR_COLOR[kind][key],
                   edgecolor="0.2", linewidth=0.6, zorder=2,
                   yerr=sd, error_kw=dict(ecolor="0.15", elinewidth=1.0,
                                          capsize=2.6, capthick=1.0, zorder=3))

    frame(ax)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{v:.2f}" for v in xs])
    ax.set_xlim(pos[0] - 0.5, pos[-1] + 0.5)
    # Headroom for the in-plot legend rather than letting it sit on the data.
    ax.set_ylim(0, top * 1.60)
    ax.set_xlabel("Box height (m)", fontsize=FS_AXIS)
    # Overbar is the conventional mark for a mean, which collapses both labels
    # to one line -- the spelled-out versions were tall enough to clip on a
    # short canvas. "Commanded" stays because the commanded/delivered
    # distinction is load-bearing; the rest is carried by the caption.
    word = "Delivered" if clip else "Commanded"
    sym = (r"\overline{\tau}" if stat == "mean"
           else r"\overline{\tau}_{\mathrm{peak}}")
    ax.set_ylabel(f"{word} ${sym}$ (N·m)", fontsize=FS_AXIS)
    two_factor_legend(ax)
    fig.tight_layout()
    save(fig, out_stem)


def save(fig, stem):
    os.makedirs(os.path.join(_DIR, "figures"), exist_ok=True)
    fig.savefig(os.path.join(_DIR, "figures", f"{stem}.png"), dpi=200)
    plt.close(fig)
    print(f"  wrote figures/{stem}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trials_dir", nargs="?",
                    default=os.path.join(_DIR, "..", "log", "trials", "workshop"))
    ap.add_argument("--heights", nargs="+", default=["5", "6", "7", "8"])
    ap.add_argument("--clip", action="store_true",
                    help="clamp to the actuator ctrlrange, as MuJoCo does")
    args = ap.parse_args()

    heights = args.heights
    xs = np.array([int(h) / 100 for h in heights])

    print(f"Reading {args.trials_dir}")
    data, ctrlrange = collect(args.trials_dir, heights, args.clip)

    suffix = "_clipped" if args.clip else ""
    plot_success(data, heights, xs, "fig_success")
    plot_torque(data, heights, xs, "mean", ctrlrange, f"fig_torque_mean{suffix}", args.clip)
    plot_torque(data, heights, xs, "peak", ctrlrange, f"fig_torque_peak{suffix}", args.clip)


if __name__ == "__main__":
    main()
