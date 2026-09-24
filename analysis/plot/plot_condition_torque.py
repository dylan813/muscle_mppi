"""
Commanded torque over the rollout, one trace per ablation condition.

Reads figures/representatives/representatives.csv (written by
select_representatives.py) and draws each pick's torque through
plot_leg_muscles.plot_joints() -- the same stacked 12-panel figure that script
already produces for a single run, with one series per condition in each panel
instead of the muscle/PD pair.

Five figures:

  torque_success_actdyn.png    success: full vs no_actdyn
  torque_success_removed.png   success: no_active_fl, no_fv, no_passive_fl
  torque_fail_hill.png         fail:    no_hill, no_actdyn_no_hill
  torque_fail_removed.png      fail:    no_active_fl, no_fv, no_passive_fl
  torque_fail_only.png         fail:    active_fl_only, fv_only, passive_fl_only

The success/removed and fail/removed pair carry the same three conditions in
each outcome, so they are listed in the same order and keep the same colours,
and the two figures can be read directly against each other.

Legend names spell out which force-length curve a condition touches: the run
scripts' "fl" is the active force-length and "passive" is the parallel elastic
force-length, so no_fl reads as no_active_fl and passive_only as
passive_fl_only. Folder names on disk are unchanged.

Colours are the validated categorical slots (see the dataviz palette): each
figure takes the first N slots in their fixed order, never cycled.

Torque is the commanded torque logged by mppi_sim (tau_j*), read through
torque_stats.trial_torque() -- not a Hill reconstruction, which would be wrong
for the ablated models. MuJoCo clamps d->ctrl to the actuator ctrlrange, so
where a trace runs past the dashed limit line the delivered torque was smaller.

Usage:
  python3 plot_condition_torque.py [--reps CSV] [--out DIR] [--include-thin]
                                   [--leg FR] [--per-panel-scale]
"""

import argparse
import os

import numpy as np
import pandas as pd

from plot_leg_muscles import (
    JOINT_NAMES,
    LEG_OFFSET,
    _MODEL_BASE,
    model_joint_info,
    plot_joints,
)
from select_representatives import load_condition_task
from torque_stats import trial_torque

_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_JOINTS = 12

# Validated categorical slots, in their fixed order. Each figure takes the
# first N; never cycled, never re-ordered.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
         "#008300", "#4a3aa7", "#e34948"]

# Legend names. The run scripts' "fl" and "passive" both refer to a
# force-length curve -- the active one and the parallel elastic one -- so the
# labels spell out which, leaving the folder names on disk untouched.
DISPLAY_NAME = {
    "no_fl":        "no_active_fl",
    "no_passive":   "no_passive_fl",
    "fl_only":      "active_fl_only",
    "passive_only": "passive_fl_only",
}

# Figure -> the conditions it shows, in slot order. Figures 2 and 4 are the same
# three conditions in success and in failure, so their conditions are listed in
# the same order and keep the same colours across the pair.
FIGURES = {
    "torque_success_actdyn": dict(
        outcome="success",
        conditions=["full", "no_actdyn"],
    ),
    "torque_success_removed": dict(
        outcome="success",
        conditions=["no_fl", "no_fv", "no_passive"],
    ),
    "torque_fail_hill": dict(
        outcome="fail",
        conditions=["no_hill", "no_actdyn_no_hill"],
    ),
    "torque_fail_removed": dict(
        outcome="fail",
        conditions=["no_fl", "no_fv", "no_passive"],
    ),
    "torque_fail_only": dict(
        outcome="fail",
        conditions=["fl_only", "fv_only", "passive_only"],
    ),
}


# Control-rate sweep: one figure per rate, the two controllers overlaid. The
# series order is fixed, so muscle is slot 1 and pd slot 2 in every rate.
FREQ_RATES = ["hz_100", "hz_50", "hz_25", "hz_12p5"]
FREQ_CONTROLLERS = ["muscle", "pd"]


def dominant_row(reps, cond, controller):
    """The representative of that cell's largest outcome class.

    Below 100 Hz most cells are overwhelmingly one outcome, and the sweep's own
    handoff warns the console labels mean little there (sub-50 Hz muscle runs
    log as "timeout" but are sagging crawls), so the largest class is taken as
    the cell's typical behaviour rather than any particular label.
    """
    g = reps[(reps.condition == cond) & (reps.controller == controller)]
    return g.loc[g.n.idxmax()] if len(g) else None


def freq_figures(reps):
    """[(stem, rows)] for the control-rate sweep."""
    out = []
    for rate in FREQ_RATES:
        rows = [r for r in (dominant_row(reps, rate, k) for k in FREQ_CONTROLLERS)
                if r is not None]
        if rows:
            out.append((f"torque_{rate}", rows))
    return out


def fail_row(reps, cond):
    """The condition's fail representative: its fall pick, or -- for fl_only,
    which never fell and never arrived -- its timeout pick."""
    rows = reps[(reps.condition == cond) & (reps.outcome == "fall")]
    if rows.empty:
        rows = reps[(reps.condition == cond) & (reps.outcome == "timeout")]
    return rows.iloc[0] if len(rows) else None


def trial_series(row, trials_root):
    """(t, tau, ctrlrange) for one representative: tau is (steps, 12), signed."""
    cond_dir = os.path.join(trials_root, row.study, row.condition)
    controller = getattr(row, "controller", "muscle")
    task = load_condition_task(cond_dir, controller)
    qadr, ctrlrange = model_joint_info(os.path.normpath(
        os.path.join(_MODEL_BASE, task["model_path"])))
    tau = trial_torque(os.path.join(_MODEL_BASE, row.csv), controller,
                       task.get(controller), qadr)
    # trial_torque drops the first logged row (its predecessor was never
    # written), so the trace starts one step in.
    t = np.arange(1, len(tau) + 1) * task["dt"]
    return t, tau, ctrlrange


def build_panels(rows, trials_root, legs):
    """plot_joints panels: one per (leg, joint), one series per condition."""
    loaded = []
    for colour, row in zip(SLOTS, rows):
        t, tau, ctrlrange = trial_series(row, trials_root)
        # The outcome is the figure's subject, so the label only carries the
        # condition and how long its rollout lasted.
        # In the ablation figures the condition is what varies; in the rate
        # sweep the rate is the figure and the controller is what varies.
        if "controller" in row and row.study == "freq_sweep":
            name = f"{row.controller}  [{row.outcome}]"
        else:
            name = DISPLAY_NAME.get(row.condition, row.condition)
        label = f"{name}  ({row.t_end:.1f}s)"
        loaded.append((label, colour, t, tau, ctrlrange))

    panels = []
    for leg in legs:
        for k, jname in enumerate(JOINT_NAMES):
            j = LEG_OFFSET[leg] + k
            series = [(t, tau[:, j], label, colour)
                      for label, colour, t, tau, _ in loaded]
            panels.append({
                "name": jname,
                "leg": leg,
                "limit": loaded[0][4][j][1],   # same model for every condition
                "series": series,
            })
    return panels


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", default=os.path.join(_DIR, "figures", "representatives",
                                                   "representatives.csv"))
    ap.add_argument("--out", default=os.path.join(_DIR, "figures", "representatives"))
    ap.add_argument("--leg", default=None, choices=sorted(LEG_OFFSET),
                    help="restrict to one leg's 3 joints (default: all 12 panels)")
    ap.add_argument("--panel-height", type=float, default=1.8,
                    help="inches of figure height per joint panel; the default "
                         "of 1.0 in plot_leg_muscles suits one or two series, "
                         "not several overlaid ones (default: 1.8)")
    ap.add_argument("--per-panel-scale", action="store_true",
                    help="scale each panel to its own peak instead of sharing one "
                         "y-scale per joint type")
    args = ap.parse_args()

    reps = pd.read_csv(args.reps)
    trials_root = os.path.join(_MODEL_BASE, "analysis", "log", "trials")
    os.makedirs(args.out, exist_ok=True)
    legs = [args.leg] if args.leg else sorted(LEG_OFFSET, key=LEG_OFFSET.get)

    if (reps.study == "freq_sweep").any():
        figures = freq_figures(reps)
    else:
        figures = []
        for stem, spec in FIGURES.items():
            rows, missing = [], []
            for cond in spec["conditions"]:
                if spec["outcome"] == "success":
                    r = reps[(reps.condition == cond) & (reps.outcome == "success")]
                    r = r.iloc[0] if len(r) else None
                else:
                    r = fail_row(reps, cond)
                (rows if r is not None else missing).append(
                    r if r is not None else cond)
            if missing:
                print(f"{stem}: no representative for {', '.join(missing)} — skipped")
            if rows:
                figures.append((stem, rows))

    for stem, rows in figures:
        print(f"{stem}: " + ", ".join(
            f"{r.controller if r.study == 'freq_sweep' else r.condition}"
            f"({r.outcome}, n={r.n})" for r in rows))
        panels = build_panels(rows, trials_root, legs)
        plot_joints(panels, os.path.join(args.out, f"{stem}.png"),
                    shared_scale=not args.per_panel_scale,
                    x_scale=1.0, x_label="Time (s)", linewidth=1.3,
                    panel_h=args.panel_height)


if __name__ == "__main__":
    main()
