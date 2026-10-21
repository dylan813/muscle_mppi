"""
Commanded FR knee torque over the rollout: three summary figures from each
study's representative rollouts (select_representatives.py).

  torque_breakdown_act_dyn_hill_FR_calf.png
      full, no_actdyn, no_hill, no_actdyn_no_hill
  torque_breakdown_hill_sub_FR_calf.png
      full (the all-on reference), no_fl, no_fv, no_passive, fl_only, fv_only,
      passive_only
  torque_bandwidth_FR_calf.png
      a panel per motor bandwidth (inf to 1 Hz), muscle, pd_act and pd overlaid
  torque_bandwidth_zoh_FR_calf.png
      the same for the zero-order-hold sweep: a panel per motor update rate
      (500 to 25 Hz)

Joint: the knee is where the breakdown conditions differ most (range of
condition-mean |tau| 4.0-4.3 N.m, against at most 1.7 for any thigh), and the
bandwidth figure uses it too so both studies show the same joint. The
bandwidth effect is larger at the thigh (11-12 N.m for pd and pd_act, against
~6 at the knee) but still clear at the knee for PD.

Leg: FR. The front legs change most between conditions in both studies, and
FR leads FL in 3 of the 5 groups.

Each condition shows the representative of its most common outcome, named in
its panel. Where the top two outcomes are
within NEAR_TIE trials of each other the script says so, since the outcome
shown is then close to a coin flip.

Panels are stacked on a shared time axis, each scaled to its own peak so the
low-torque conditions keep their detail; compare magnitudes on the tick labels,
not the trace heights. Breakdown: one trace per
panel, in the muscle colour. Bandwidth: the three controllers overlaid in the
peak-torque bars' colours (plot_ablation_bars' FAMILY_COLOR mid tones); muscle
and pd sit at protan dE 7.8, so pd also takes plot_box_height's dashed line.

Time is the sim clock (the CSV's t). A fall stops at its fall time: the sim
stops logging at a fall it detects, and a fall relabelled after review, whose
run kept going, is cut to match. Torque is the commanded torque the sims log
(tau_j*), read through torque_stats.trial_torque(); the dashed grey lines mark
the actuator limit where a trace passes it.

Writes figures/torque_profiles/.

Usage:
  python3 plot_joint_torque.py
"""

import os

import pandas as pd

from plot_ablation_bars import FAMILY_COLOR, ZOH_RATES, bandwidth_heading, hold_heading
from plot_box_height import JOINT_LABEL, STYLE
from plot_condition_torque import BW_BANDWIDTHS, BW_CONTROLLERS, DISPLAY_NAME
from plot_leg_muscles import JOINT_NAMES, LEG_OFFSET, _MODEL_BASE, model_joint_info, plot_joints
from select_representatives import SELECT_JOINT, SELECT_LEG, load_condition_task
from torque_stats import trial_torque

_DIR = os.path.dirname(os.path.abspath(__file__))
TRIALS = os.path.join(_MODEL_BASE, "analysis", "log", "trials")
OUT_DIR = os.path.join(_DIR, "figures", "torque_profiles")

LEG, JOINT = SELECT_LEG, SELECT_JOINT   # the joint the representatives are picked on
J = LEG_OFFSET[LEG] + JOINT_NAMES.index(JOINT)

FIG_W = 16.0     # inches: ~0.8 in per second of a 20 s rollout
PANEL_H = {"breakdown": 1.5, "bandwidth": 2.0, "bandwidth_zoh": 2.0}   # one trace, or three overlaid
NEAR_TIE = 10    # trials

# Bandwidth sweeps: each one's motor rates (folder suffixes, fastest first) and
# the panel heading for a rate.
SWEEPS = {
    "bandwidth":     (BW_BANDWIDTHS, bandwidth_heading),
    "bandwidth_zoh": (ZOH_RATES, hold_heading),
}

BREAKDOWN_GROUPS = {
    "act_dyn_hill": ["full", "no_actdyn", "no_hill", "no_actdyn_no_hill"],
    "hill_sub":     ["full", "no_fl", "no_fv", "no_passive", "fl_only", "fv_only", "passive_only"],
}

# Trace colour per controller: the mid (hip flex-ext) step of its bar ramp.
CONTROLLER_COLOUR = {family: ramp["thigh"] for family, ramp in FAMILY_COLOR.items()}
CONTROLLER_LINESTYLE = {"muscle": STYLE["muscle"]["linestyle"],
                        "pd": STYLE["pd"]["linestyle"],
                        "pd_act": "-"}


def most_common(reps, cond):
    """The representative of cond's most common outcome."""
    g = reps[reps.condition == cond].sort_values("n", ascending=False)
    top = g.iloc[0]
    if len(g) > 1 and top.n - g.iloc[1].n <= NEAR_TIE:
        print(f"  near tie: {cond} {top.outcome} {top.n} vs {g.iloc[1].outcome} {g.iloc[1].n}")
    return top


def knee_trace(study, row):
    """(t, tau, limit) of the FR knee for one representative, t the sim clock."""
    cond_dir = os.path.join(TRIALS, study, row.condition)
    task = load_condition_task(cond_dir, row.controller)
    qadr, ctrlrange = model_joint_info(os.path.normpath(
        os.path.join(_MODEL_BASE, task["model_path"])))
    csv = os.path.join(_MODEL_BASE, row.csv)
    tau = trial_torque(csv, row.controller, task.get(row.controller), qadr)[:, J]
    # tau's rows are the CSV's rows from the second on (see trial_torque()).
    t = pd.read_csv(csv, usecols=["t"])["t"].to_numpy()[1:]
    if row.outcome == "fall":
        keep = t <= row.t_event + 1e-6
        t, tau = t[keep], tau[keep]
    return t, tau, ctrlrange[J][1]


def save(panels, study, stem):
    plot_joints(panels, os.path.join(OUT_DIR, f"{stem}.png"), shared_scale=False,
                x_scale=1.0, x_label="Time (s)", linewidth=1.3,
                panel_h=PANEL_H[study], fig_w=FIG_W)


def breakdown_figures():
    reps = pd.read_csv(os.path.join(TRIALS, "breakdown", "videos", "representatives.csv"))
    for group, conds in BREAKDOWN_GROUPS.items():
        print(f"breakdown/{group}:")
        panels = []
        for cond in conds:
            row = most_common(reps, cond)
            t, tau, limit = knee_trace("breakdown", row)
            title = DISPLAY_NAME.get(cond, cond)
            panels.append({"name": JOINT_LABEL[JOINT], "leg": LEG, "limit": limit,
                           "note": f"{title}\n{row.outcome}",
                           "series": [(t, tau, title, CONTROLLER_COLOUR["muscle"])]})
        save(panels, "breakdown", f"torque_breakdown_{group}_{LEG}_{JOINT}")


def bandwidth_figure(study):
    rates, heading = SWEEPS[study]
    reps = pd.read_csv(os.path.join(TRIALS, study, "videos", "representatives.csv"))
    print(f"{study}:")
    panels = []
    for band in rates:
        series, outcomes = [], []
        for controller in BW_CONTROLLERS:
            row = most_common(reps, f"{controller}_{band}")
            t, tau, limit = knee_trace(study, row)
            # Labelled by controller alone, the same in every panel, so the
            # legend (taken from the first panel) names all of them.
            series.append((t, tau, controller, CONTROLLER_COLOUR[controller],
                           CONTROLLER_LINESTYLE[controller]))
            outcomes.append(f"{controller}: {row.outcome}")
        panels.append({"name": JOINT_LABEL[JOINT], "leg": LEG, "limit": limit,
                       "note": "\n".join([heading(band)] + outcomes),
                       "series": series})
    save(panels, study, f"torque_{study}_{LEG}_{JOINT}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    breakdown_figures()
    for study in SWEEPS:
        bandwidth_figure(study)


if __name__ == "__main__":
    main()
