"""
Study summary figures: trial outcomes and commanded peak torque per condition.

Two figures per study, written to figures/:

    fig_<study>_outcome.png        % of trials per outcome: success, fall, timeout
    fig_<study>_torque_peak.png    peak |commanded torque| per joint type

Studies, each on an x axis split into headed sections:

    breakdown   run_breakdown.sh: activation dynamics x Hill | Hill sub-components
    bandwidth   run_bandwidth.sh: motor bandwidth inf | 20 | 8 | 3.2 | 1 Hz,
                with muscle, pd_act and pd in each section. Each heading
                gives the cutoff f and the motor's time constant 1/(2*pi*f).

Outcome comes from each trial's console.log through
select_representatives.parse_outcome(): success, fall (including falls
relabelled by hand after review, appended to the log), or timeout (still
upright when the sim_duration clock ran out). Three bars per condition, each
the % of its trials, so they sum to 100. A bar at 0 is indistinguishable from
a missing bar, so every bar carries its value.

Torque is pooled over every trial of a condition regardless of outcome -- the
same as plot_box_height, and deliberate: conditioning on success would compare
conditions on very different subsets (fl_only has no successes at all, so it
would vanish from the torque figure entirely). Each trial contributes one peak,
so a 2 s fall and a 20 s timeout count equally despite logging very different
numbers of steps. A fall counts only up to its fall time: the sim stops
logging at a fall it detects, and a relabelled fall, whose run kept going, is
cut to match.

Bar colours for torque: hue family = controller, lightness within the family =
joint, proximal to distal (plot_box_height's BAR_COLOR for muscle and pd).
pd_act gets its own blue ramp, validated as an ordinal scale (monotone
lightness, adjacent dL >= 0.06, light end 2.5:1 on white, hue spread 3 deg);
its mid tone sits at normal-vision dE >= 20.8 from the other two. Controller
identity also rests on the tick labels, which the muscle/pd pair needs (protan
dE 7.8, see plot_box_height).

Outcome colours are three of plot_condition_torque's validated slots (all-pairs
CVD dE >= 9.2). Aqua is 2.8:1 on white, so the value labels are required, not
decoration.

Statistics come from torque_stats.summarize(), which plot_box_height also uses,
so these figures and that script's tables cannot drift apart. Torque is the
commanded torque logged by the sims (tau_j*), read through
torque_stats.trial_torque() -- not a Hill reconstruction, which would be wrong
for the ablated models.

Usage:
  python3 plot_ablation_bars.py [breakdown] [bandwidth] [--trials-root DIR] [--clip]
"""

import argparse
import glob
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
from plot_condition_torque import BW_BANDWIDTHS, BW_CONTROLLERS, DISPLAY_NAME
from plot_leg_muscles import JOINT_NAMES, _MODEL_BASE, model_joint_info
from select_representatives import CONTROLLERS, OUTCOMES, load_condition_task, parse_outcome
from torque_stats import summarize

_DIR = os.path.dirname(os.path.abspath(__file__))


def bandwidth_heading(band):
    """'20 Hz (7.96 ms)': motor cutoff f and its time constant 1/(2*pi*f)."""
    f = math.inf if band == "inf" else float(band.removesuffix("Hz"))
    cutoff = "∞" if math.isinf(f) else f"{f:g}"
    return f"{cutoff} Hz ({1000.0 / (2 * math.pi * f):.3g} ms)"


# Each study: headed sections of conditions, in x order. A condition names its
# folder, its bar colour family and its tick label. Sections are separated by a
# divider rather than coloured, which would add a second encoding.
STUDIES = {
    "breakdown": [
        ("activation dynamics × Hill",
         [dict(cond=c, family="muscle", label=DISPLAY_NAME.get(c, c))
          for c in ["full", "no_actdyn", "no_hill", "no_actdyn_no_hill"]]),
        ("Hill sub-components",
         [dict(cond=c, family="muscle", label=DISPLAY_NAME.get(c, c))
          for c in ["no_fl", "no_fv", "no_passive", "fl_only", "fv_only", "passive_only"]]),
    ],
    "bandwidth": [
        (bandwidth_heading(b),
         [dict(cond=f"{c}_{b}", family=c, label=c) for c in BW_CONTROLLERS])
        for b in BW_BANDWIDTHS
    ],
}

# Torque y-axis headroom above the tallest bar (mean + SD). The breakdown's
# legend needs room above its bars; the bandwidth's tallest bars are at 1 Hz,
# on the right, clear of its legend at upper left.
TORQUE_HEADROOM = {"breakdown": 1.34, "bandwidth": 1.05}

# Torque bars: BAR_COLOR's muscle and pd ramps, plus pd_act's (see docstring).
FAMILY_COLOR = {**BAR_COLOR,
                "pd_act": {"hip": "#6fa7ea", "thigh": "#2a78d6", "calf": "#16488a"}}

OUTCOME_COLOR = {"success": "#1baf7a", "fall": "#eb6834", "timeout": "#2a78d6"}

FIG_H = 4.4
W_PER_COND = 1.04   # inches per condition: the breakdown's 10 conditions at 10.4 in
BAR_W, BAR_GAP = 0.22, 0.02


def conditions(study):
    return [c for _, conds in STUDIES[study] for c in conds]


def outcomes(ctrl_dir):
    """Outcome counts over one condition's trials, and each fall's time."""
    counts = dict.fromkeys(OUTCOMES, 0)
    fall_t = {}
    for log in sorted(glob.glob(os.path.join(ctrl_dir, "trial_*", "console.log"))):
        outcome, t = parse_outcome(log)
        counts[outcome] += 1
        if outcome == "fall" and np.isfinite(t):
            fall_t[os.path.basename(os.path.dirname(log))] = t
    return counts, fall_t


def collect(study, trials_root, clip):
    """{condition: per-trial peak arrays and outcome counts}."""
    out = {}
    for c in conditions(study):
        cond_dir = os.path.join(trials_root, study, c["cond"])
        kinds = [k for k in CONTROLLERS if os.path.isdir(os.path.join(cond_dir, k))]
        if len(kinds) != 1:
            raise SystemExit(f"expected one controller folder (muscle/ or pd/) in {cond_dir}")
        kind = kinds[0]
        ctrl_dir = os.path.join(cond_dir, kind)

        task = load_condition_task(cond_dir, kind)
        qadr, ctrlrange = model_joint_info(os.path.normpath(
            os.path.join(_MODEL_BASE, task["model_path"])))
        counts, fall_t = outcomes(ctrl_dir)
        df, _, jpeaks, _ = summarize(ctrl_dir, kind, task[CONTROLLERS[kind][1]],
                                     qadr, ctrlrange, clip, t_end=fall_t)
        n = sum(counts.values())
        if len(df) != n:
            raise SystemExit(f"{ctrl_dir}: {n} console logs but {len(df)} trial CSVs")

        out[c["cond"]] = dict(jpeaks=jpeaks, counts=counts, n=n)
        print(f"  {c['cond']:<20} {n:3} trials   "
              + "  ".join(f"{o} {counts[o]:3}" for o in OUTCOMES))
    return out


def peak_series(study, data, key):
    """(mean, sd) across trials of one joint type's peak, per condition."""
    mu, sd = [], []
    for c in conditions(study):
        # Per trial first (max over the 4 legs), then across trials, so the SD
        # is between-trial spread rather than between-leg spread.
        v = data[c["cond"]]["jpeaks"][:, joint_cols(JOINT_NAMES.index(key))].max(axis=1)
        mu.append(v.mean())
        sd.append(v.std(ddof=1))
    return np.array(mu), np.array(sd)


def bar_offset(i):
    """x offset of the i-th of three bars in a condition's group."""
    span = 3 * BAR_W + 2 * BAR_GAP
    return -span / 2 + i * (BAR_W + BAR_GAP) + BAR_W / 2


def new_axes(study):
    n = len(conditions(study))
    fig, ax = plt.subplots(figsize=(W_PER_COND * n, FIG_H))
    return fig, ax, np.arange(n, dtype=float)


def xaxis(ax, pos, study):
    """Condition ticks, plus a divider and a heading per section."""
    ax.set_xticks(pos)
    ax.set_xticklabels([c["label"] for c in conditions(study)],
                       rotation=30, ha="right", fontsize=FS_LEG)
    ax.set_xlim(pos[0] - 0.5, pos[-1] + 0.5)

    lo = 0
    for k, (heading, conds) in enumerate(STUDIES[study]):
        hi = lo + len(conds)
        if k > 0:
            ax.axvline(pos[lo] - 0.5, color="0.55", linewidth=0.9,
                       linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(heading, xy=(0.5 * (pos[lo] + pos[hi - 1]), 1.0), xytext=(0, 5),
                    xycoords=("data", "axes fraction"), textcoords="offset points",
                    ha="center", va="bottom", fontsize=FS_LEG, color="0.4")
        lo = hi


def plot_outcome(study, data, out_stem):
    """Three bars per condition: % success, fall, timeout."""
    fig, ax, pos = new_axes(study)

    for i, outcome in enumerate(OUTCOMES):
        pct = np.array([100.0 * data[c["cond"]]["counts"][outcome] / data[c["cond"]]["n"]
                        for c in conditions(study)])
        x = pos + bar_offset(i)
        ax.bar(x, pct, width=BAR_W, color=OUTCOME_COLOR[outcome],
               edgecolor="0.2", linewidth=0.6, zorder=2)
        for xi, v in zip(x, pct):
            ax.annotate(f"{v:.0f}", xy=(xi, v), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=FS_LEG - 3, color="0.25")

    frame(ax)
    # Headroom for the in-plot legend rather than letting it sit on the data.
    ax.set_ylim(0, 134)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Trials (%)", fontsize=FS_AXIS)
    xaxis(ax, pos, study)

    handles = [Patch(facecolor=OUTCOME_COLOR[o], edgecolor="0.2", linewidth=0.6, label=o)
               for o in OUTCOMES]
    style_legend(ax, ax.legend(handles=handles, loc="upper left", ncol=3,
                               fontsize=FS_LEG, frameon=True, handlelength=1.5,
                               handleheight=1.0, columnspacing=1.4))
    fig.tight_layout()
    save(fig, out_stem)


def plot_torque_peak(study, data, out_stem, clip):
    """Three bars per condition, one per joint type, in its controller's ramp."""
    fig, ax, pos = new_axes(study)
    families = [c["family"] for c in conditions(study)]

    top = 0.0
    for ji, key in enumerate(JOINT_NAMES):
        mu, sd = peak_series(study, data, key)
        top = max(top, (mu + sd).max())
        ax.bar(pos + bar_offset(ji), mu, width=BAR_W,
               color=[FAMILY_COLOR[f][key] for f in families],
               edgecolor="0.2", linewidth=0.6, zorder=2,
               yerr=sd, error_kw=dict(ecolor="0.15", elinewidth=1.0,
                                      capsize=2.6, capthick=1.0, zorder=3))

    frame(ax)
    word = "Delivered" if clip else "Commanded"
    ax.set_ylabel(f"{word} " + r"$\overline{\tau}_{\mathrm{peak}}$ (N·m)", fontsize=FS_AXIS)
    xaxis(ax, pos, study)

    # One family: a swatch per joint. Several: a headed column per controller,
    # a row per joint, so hue and lightness are each read off one axis.
    handles, labels = [], []
    order = list(dict.fromkeys(families))
    for f in order:
        if len(order) > 1:
            handles.append(Line2D([], [], linestyle="none"))
            labels.append(f)
        for key in JOINT_NAMES:
            handles.append(Patch(facecolor=FAMILY_COLOR[f][key], edgecolor="0.2", linewidth=0.6))
            labels.append(JOINT_LABEL[key])
    one_row = len(order) == 1
    ax.set_ylim(0, top * TORQUE_HEADROOM[study])
    style_legend(ax, ax.legend(handles, labels, loc="upper left",
                               ncol=len(JOINT_NAMES) if one_row else len(order),
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
    ap.add_argument("studies", nargs="*", help=f"any of {', '.join(STUDIES)} (default: all)")
    ap.add_argument("--trials-root", default=os.path.join(_DIR, "..", "log", "trials"))
    ap.add_argument("--clip", action="store_true",
                    help="clamp to the actuator ctrlrange, reporting delivered "
                         "rather than commanded torque")
    args = ap.parse_args()
    unknown = set(args.studies) - set(STUDIES)
    if unknown:
        ap.error(f"unknown study {', '.join(sorted(unknown))}; expected {', '.join(STUDIES)}")

    suffix = "_clipped" if args.clip else ""
    for study in args.studies or list(STUDIES):
        print(f"{study}:")
        data = collect(study, os.path.normpath(args.trials_root), args.clip)
        plot_outcome(study, data, f"fig_{study}_outcome")
        plot_torque_peak(study, data, f"fig_{study}_torque_peak{suffix}", args.clip)


if __name__ == "__main__":
    main()
