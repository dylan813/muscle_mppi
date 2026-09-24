"""
Plot joint torque output from mppi_sim's CSV pair (<name>.csv with t/dq_j*/act_m*,
and <name>_qpos.csv with the full qpos trajectory).

Produces one PNG with all 12 joints stacked vertically (FR, FL, RR, RL x hip, thigh,
calf), each panel showing the net torque that joint's antagonistic muscle pair
produces, overlaid with the same joint's torque from the PD baseline (pd_mppi_sim).
Each panel is scaled to its own peak; --shared-scale gives every panel of a joint type
one y-scale instead, so legs compare directly. Traces show the raw commanded torque,
with the actuator ctrlrange drawn as a dashed reference wherever a trace exceeds it;
--clip instead clamps to that range, as MuJoCo does, showing delivered torque. Pass
--leg to restrict the figure to a single leg's 3 joints, --no-pd to drop the overlay.

Neither sim logs torque, so both traces are reconstructed:
  muscle — re-runs the Hill model of controllers/muscle/control/muscle.h on the logged
           (q, dq, act) and sums the agonist and antagonist contributions
  PD     — kp*(q_des - q) - kd*dq from the logged (q, dq, q_des), matching
           controllers/pd/control/base_mppi_pd.h::unitree_pd_torque

Parameters come from controllers/muscle/utils/tasks.yaml and controllers/pd/utils/tasks_pd.yaml, so --task must
name the task the runs came from (same requirement as analysis/render_gif.py). The two
CSVs are separate runs, so the traces are compared as distributions over time, not
sample-by-sample.

Usage:
  python3 plot_leg_muscles.py [csv_path] [name] [--leg FR] [--task walk]
                              [--yaml PATH] [--model PATH] [--outdir DIR]

  csv_path  defaults to analysis/data/mppi_sim/mppi_sim.csv
  name      output filename prefix (default "<task>", or "<task>_<leg>" with --leg)
  --leg     FR | FL | RR | RL   (default: all four)
  --task    task key in tasks.yaml the run was generated from (default "walk")
  --outdir  where <name>_muscle_torque.png goes (default: figures/ beside this script;
            pass the trial directory to keep the plot with its data)
"""

import argparse
import os

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
import yaml

_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_DIR, "..", "..")

DEFAULT_CSV  = os.path.join(_REPO, "analysis", "data", "mppi_sim", "mppi_sim.csv")
DEFAULT_YAML = os.path.join(_REPO, "controllers", "muscle", "utils", "tasks.yaml")
DEFAULT_PD_CSV  = os.path.join(_REPO, "analysis", "data", "pd_mppi_sim", "pd_mppi_sim.csv")
DEFAULT_PD_YAML = os.path.join(_REPO, "controllers", "pd", "utils", "tasks_pd.yaml")
# model_path in both yamls is written relative to the repo root.
_MODEL_BASE  = _REPO

JOINT_NAMES = ["hip", "thigh", "calf"]
# Muscle-model joint index follows actuator order (FR, FL, RR, RL), not qpos order.
LEG_OFFSET  = {"FR": 0, "FL": 3, "RR": 6, "RL": 9}

# A flatter green (#4a7c1f, #2e7d32) is indistinguishable from this orange under
# protanopia (dE 0.5-2.2); this one clears the separation floor at 7.8.
C_MUSCLE = "#0f8a5f"
C_PD     = "#d1620a"

EPS = 1e-6


# ── Hill model (vectorized port of controllers/muscle/control/muscle.h) ──────

def active_force_length(length, A, mid, B):
    """MuJoCo's piecewise-quadratic force-length bump: 1.0 at mid, 0 outside [A, B]."""
    left  = 0.5 * (A + mid)
    right = 0.5 * (mid + B)
    out = np.zeros_like(length)

    m = (length > A) & (length < left)
    t = (length[m] - A) / (left - A)
    out[m] = 0.5 * t * t

    m = (length >= left) & (length < mid)
    t = (mid - length[m]) / (mid - left)
    out[m] = 1.0 - 0.5 * t * t

    m = (length >= mid) & (length < right)
    t = (length[m] - mid) / (right - mid)
    out[m] = 1.0 - 0.5 * t * t

    m = (length >= right) & (length < B)
    t = (B - length[m]) / (B - right)
    out[m] = 0.5 * t * t

    return out


def force_vel(velocity, c, vmax, FVmax):
    """Concentric quadratic rise 0->1; eccentric quadratic rise 1->FVmax."""
    eff = velocity / vmax
    out = np.full_like(eff, FVmax)

    out[eff < -1.0] = 0.0

    m = (eff >= -1.0) & (eff <= 0.0)
    out[m] = (eff[m] + 1.0) ** 2

    m = (eff > 0.0) & (eff <= c)
    out[m] = FVmax - (c - eff[m]) ** 2 / c

    return out


def passive_force_length(length, pmax, b):
    """Parallel elastic force: zero below optimal length, cubic then linear above."""
    out = np.zeros_like(length)

    m = (length > 1.0) & (length <= b)
    t = (length[m] - 1.0) / (b - 1.0)
    out[m] = 0.25 * pmax * t * t * t

    m = length > b
    t = (length[m] - b) / (b - 1.0)
    out[m] = 0.25 * pmax * (1.0 + 3.0 * t)

    return out


def muscle_torques(q, dq, act1, act2, p, j):
    """Per-muscle torque for joint j. Returns (tau_ag, tau_ant) in N.m, signed so
    their sum is the net joint torque."""
    lce_min, lce_max = p["lce_min"][j], p["lce_max"][j]
    phi_min, phi_max = p["phi_min"][j], p["phi_max"][j]

    r1 = (lce_max - lce_min + EPS) / (phi_max - phi_min + EPS)
    r2 = (lce_max - lce_min + EPS) / (phi_min - phi_max + EPS)

    lce1 = q * r1 + (lce_min - r1 * phi_min)
    lce2 = q * r2 + (lce_min - r2 * phi_max)

    shoulder_mid = 0.5 * (lce_min + 0.95)
    afl1 = (active_force_length(lce1, lce_min, 1.0, lce_max)
            + 0.15 * active_force_length(lce1, lce_min, shoulder_mid, 0.95))
    afl2 = (active_force_length(lce2, lce_min, 1.0, lce_max)
            + 0.15 * active_force_length(lce2, lce_min, shoulder_mid, 0.95))

    FVmax = p["FVmax"][j]
    c     = FVmax - 1.0
    fv1 = force_vel(r1 * dq, c, p["vmax"][j], FVmax)
    fv2 = force_vel(r2 * dq, c, p["vmax"][j], FVmax)

    b_passive = 0.5 * (lce_max + 1.0)
    pfl1 = passive_force_length(lce1, p["pFLmax"][j], b_passive)
    pfl2 = passive_force_length(lce2, p["pFLmax"][j], b_passive)

    peak = p["peak_force"][j]
    F1 = (afl1 * fv1 * act1 + pfl1) * peak
    F2 = (afl2 * fv2 * act2 + pfl2) * peak

    # Net joint torque in muscle.h is -(F1*r1 + F2*r2); split it per muscle.
    return -F1 * r1, -F2 * r2


# ── PD baseline (controllers/pd/control/base_mppi_pd.h::unitree_pd_torque) ───

def pd_torque(q, dq, q_des, pd, j):
    """Actuator-level PD torque for joint j, in N.m. Callers in pd_mppi pass
    dq_des=0 and tau_ff=0, so this is kp*(q_des - q) - kd*dq."""
    return pd["kp"][j] * (q_des - q) - pd["kd"][j] * dq


# ── loading ──────────────────────────────────────────────────────────────────

def load_run(csv_path):
    """Return (df, qpos) with the qpos rows aligned to df's rows."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    qpos_path = os.path.splitext(csv_path)[0] + "_qpos.csv"
    if not os.path.isfile(qpos_path):
        raise SystemExit(f"missing companion qpos log: {qpos_path}")
    qpos = np.loadtxt(qpos_path, delimiter=",")

    n = min(len(df), len(qpos))
    return df.iloc[:n].reset_index(drop=True), qpos[:n]


def load_task(yaml_path, task_name):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if task_name not in cfg:
        raise SystemExit(f"task '{task_name}' not in {yaml_path}; "
                         f"have: {', '.join(k for k in cfg if not k.startswith('default'))}")
    return cfg[task_name]


def model_joint_info(model_path):
    """Return (qadr, ctrlrange) indexed by actuator-order joint, matching how
    mppi_sim.cpp resolves qa[j]. ctrlrange is what MuJoCo clamps d->ctrl to, so
    it bounds the torque either controller can actually deliver."""
    model = mujoco.MjModel.from_xml_path(model_path)
    qadr = [int(model.jnt_qposadr[model.actuator_trnid[a, 0]])
            for a in range(model.nu)]
    limits = [tuple(model.actuator_ctrlrange[a]) if model.actuator_ctrllimited[a]
              else (-np.inf, np.inf) for a in range(model.nu)]
    return qadr, limits


# ── plotting ─────────────────────────────────────────────────────────────────

PANEL_H = 1.0   # inches of figure height per joint panel
FIG_W   = 11.0  # inches of figure width

# Left-gutter geometry, in inches from the figure's left edge. Ordered so the
# leg/joint label sits furthest out and the units label nearest the plots.
LABEL_X_IN    = 0.85   # right edge of the "FR Thigh" labels
UNITS_X_IN    = 1.15   # centre of the rotated "Torque (N·m)"
LEGEND_X_IN   = 1.25   # right edge of the stacked legend
AXES_LEFT_IN  = 1.70   # left edge of the axes (tick labels fill the gap)


def plot_joints(panels, out_path, shared_scale=False,
                x_scale=1e3, x_label="Time (ms)", linewidth=1.0, panel_h=PANEL_H):
    """Stacked torque panels, one per (leg, joint), each overlaying its series.

    x_scale/x_label default to milliseconds, which suits the short single-run
    traces this was written for; the ablation comparison passes seconds, whose
    rollouts run to 20 s. linewidth is raised there too, because its series
    colours are categorical rather than the two-series muscle/PD pair and the
    lighter slots need the extra weight to read against the surface. panel_h is
    the inches of height per panel: one or two series read fine in a 1 inch
    panel, but four or five overlaid traces need more vertical room to separate.
    """
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(FIG_W, panel_h * n), sharex=True)
    axes = np.atleast_1d(axes)
    # Header offset is in inches so it holds at any panel count.
    height = panel_h * n

    # Each panel is scaled to its own peak. A shared per-joint-type scale reads
    # better when the legs are similar, but one leg's saturation transient can
    # be 4x the others and flattens the rest; --shared-scale opts back into it.
    lim_by_panel = [1.05 * max(np.abs(tau).max() for _, tau, _, _ in p["series"])
                    for p in panels]
    if shared_scale:
        by_type = {}
        for panel, lim in zip(panels, lim_by_panel):
            by_type[panel["name"]] = max(by_type.get(panel["name"], 0.0), lim)
        lim_by_panel = [by_type[p["name"]] for p in panels]

    for ax, panel, ylim in zip(axes, panels, lim_by_panel):
        ax.axhline(0, color="0.6", linewidth=0.7, zorder=0)
        # Actuator saturation limit — only worth drawing when a trace reaches it.
        lim = panel.get("limit")
        if lim is not None and np.isfinite(lim) and lim < ylim:
            for sign in (1, -1):
                ax.axhline(sign * lim, color="0.45", linewidth=0.7,
                           linestyle=(0, (4, 3)), zorder=1)
        # Painted front-to-back in list order, so the first series sits on top
        # while the legend still reads in that same order.
        for i, (t, tau, label, color) in enumerate(panel["series"]):
            ax.plot(t * x_scale, tau, color=color, linewidth=linewidth, label=label,
                    zorder=len(panel["series"]) - i + 1)

        ax.set_ylim(-ylim, ylim)
        ax.grid(axis="x", color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_ylabel("")

    axes[-1].set_xlabel(x_label)

    # Explicit margins rather than tight_layout: the left gutter holds three
    # columns that must not collide — leg/joint labels outermost, the shared
    # units label nearer the axes, then the tick labels. Positions are in
    # inches converted to figure fractions, so they hold at any panel count.
    # The legend now sits in the left gutter rather than under the plots, so the
    # bottom margin only has to clear the x-label.
    n_series = len(panels[0]["series"])
    has_legend = n_series > 1
    # The legend is stacked one row per series in the bottom of the left gutter,
    # so past the muscle/PD pair it grows tall enough to reach the last panel's
    # label. Give it the room rather than letting it overlap.
    bottom_in = 0.55 if n_series <= 2 else 0.30 + 0.20 * n_series
    fig.subplots_adjust(left=AXES_LEFT_IN / FIG_W, right=0.985,
                        top=1 - 0.10 / height, bottom=bottom_in / height,
                        hspace=0.45)

    fig.text(UNITS_X_IN / FIG_W, (bottom_in / height + 1) / 2, "Torque (N·m)",
             rotation=90, ha="center", va="center", fontsize=10)
    for ax, panel in zip(axes, panels):
        box = ax.get_position()
        fig.text(LABEL_X_IN / FIG_W, 0.5 * (box.y0 + box.y1),
                 f"{panel['leg']} {panel['name'].capitalize()}",
                 ha="right", va="center", fontsize=10)

    if has_legend:
        handles, labels = axes[0].get_legend_handles_labels()
        # Stacked in the left gutter, nudged past the leg/joint label column so
        # it sits under the units label rather than flush with the labels.
        leg = fig.legend(handles, labels, loc="lower right", ncol=1,
                         fontsize=9, frameon=True, handlelength=1.6,
                         labelspacing=0.4, borderaxespad=0, borderpad=0.6,
                         bbox_to_anchor=(LEGEND_X_IN / FIG_W, 0.08 / height))
        leg.get_frame().set_linewidth(0.8)
        leg.get_frame().set_edgecolor("black")
        leg.get_frame().set_facecolor("white")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", nargs="?", default=DEFAULT_CSV)
    ap.add_argument("name", nargs="?", default=None)
    ap.add_argument("--leg", default=None, choices=sorted(LEG_OFFSET),
                    help="restrict to one leg's 3 joints (default: all 12)")
    ap.add_argument("--task", default="walk")
    ap.add_argument("--outdir", default=os.path.join(_DIR, "figures"),
                    help="output directory (default: figures/ beside this script)")
    ap.add_argument("--yaml", default=DEFAULT_YAML)
    ap.add_argument("--model", default=None,
                    help="override the model XML (default: task's model_path)")
    ap.add_argument("--pd-csv", default=DEFAULT_PD_CSV,
                    help="pd_mppi_sim CSV to overlay for comparison")
    ap.add_argument("--pd-task", default=None,
                    help="task key in tasks_pd.yaml (default: same as --task)")
    ap.add_argument("--pd-yaml", default=DEFAULT_PD_YAML)
    ap.add_argument("--no-pd", action="store_true",
                    help="plot the muscle torque alone, without the PD overlay")
    ap.add_argument("--shared-scale", action="store_true",
                    help="give every panel of the same joint type one y-scale, "
                         "so legs can be compared directly")
    ap.add_argument("--clip", action="store_true",
                    help="clip traces to the actuator ctrlrange, as MuJoCo does, "
                         "showing delivered torque instead of the raw command")
    args = ap.parse_args()

    name = args.name or (f"{args.task}_{args.leg}" if args.leg else args.task)
    csv_path = os.path.abspath(args.csv_path)
    out_dir = os.path.abspath(args.outdir)
    os.makedirs(out_dir, exist_ok=True)

    task = load_task(args.yaml, args.task)
    p = task["muscle"]
    model_path = args.model or os.path.normpath(
        os.path.join(_MODEL_BASE, task["model_path"]))

    df, qpos = load_run(csv_path)
    qadr, ctrlrange = model_joint_info(model_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"  task '{args.task}'  |  model {model_path}")

    # mppi_sim computes tau from the state read *before* mj_step, then logs the
    # post-step q/dq alongside the activation that produced tau. So the torque
    # applied over row k used row k-1's state; shift to reconstruct it, dropping
    # the first row (whose predecessor was never logged).
    t = df["t"].to_numpy()[1:]
    legs = [args.leg] if args.leg else sorted(LEG_OFFSET, key=LEG_OFFSET.get)

    # pd_mppi_sim has the same loop structure and the same one-row log offset,
    # but logs q_des instead of activation; its torque is the PD law, not Hill.
    pd_run = None
    pd_csv = os.path.abspath(args.pd_csv)
    if not args.no_pd:
        if not os.path.isfile(pd_csv):
            print(f"  no PD log at {pd_csv} — plotting muscle torque alone")
        else:
            pd_task_name = args.pd_task or args.task
            pd_task = load_task(args.pd_yaml, pd_task_name)
            pd_model = os.path.normpath(
                os.path.join(_MODEL_BASE, pd_task["model_path"]))
            pd_df, pd_qpos = load_run(pd_csv)
            pd_qadr, _ = model_joint_info(pd_model)
            pd_run = dict(pd=pd_task["pd"], df=pd_df, qpos=pd_qpos,
                          qadr=pd_qadr, t=pd_df["t"].to_numpy()[1:])
            print(f"  PD overlay: {len(pd_df)} rows from {pd_csv}")
            print(f"    task '{pd_task_name}'  |  model {pd_model}")

    panels = []
    for leg in legs:
        for k, jname in enumerate(JOINT_NAMES):
            j = LEG_OFFSET[leg] + k
            q  = qpos[:-1, qadr[j]]
            dq = df[f"dq_j{j}"].to_numpy()[:-1]
            a1 = df[f"act_m{2 * j}"].to_numpy()[1:]
            a2 = df[f"act_m{2 * j + 1}"].to_numpy()[1:]

            # MuJoCo clamps d->ctrl to the actuator range. --clip reproduces that
            # (the torque actually delivered); by default the raw command is
            # shown instead, with the limit drawn as a reference line so the
            # saturation is still visible rather than flattened out of sight.
            lo, hi = ctrlrange[j]
            clip = (lambda x: np.clip(x, lo, hi)) if args.clip else (lambda x: x)

            tau1, tau2 = muscle_torques(q, dq, a1, a2, p, j)
            series = [(t, clip(tau1 + tau2), "Muscle", C_MUSCLE)]

            if pd_run is not None:
                tau_pd = pd_torque(pd_run["qpos"][:-1, pd_run["qadr"][j]],
                                   pd_run["df"][f"dq_j{j}"].to_numpy()[:-1],
                                   pd_run["df"][f"qdes_j{j}"].to_numpy()[1:],
                                   pd_run["pd"], j)
                sat = 100.0 * np.mean((tau_pd < lo) | (tau_pd > hi))
                if sat > 0.5:
                    print(f"    joint {j:2d}: PD command exceeds the "
                          f"±{hi:.4g} N·m actuator limit on {sat:.1f}% of steps"
                          + (" (clipped)" if args.clip else ""))
                series.append((pd_run["t"], clip(tau_pd), "PD", C_PD))

            panels.append(dict(joint=j, name=jname, leg=leg, series=series,
                               limit=hi))

    out_path = os.path.join(out_dir, f"{name}_muscle_torque.png")
    plot_joints(panels, out_path, shared_scale=args.shared_scale)
    print("Done.")


if __name__ == "__main__":
    main()
