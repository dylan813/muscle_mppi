"""
Commanded torque for one leg, zoomed on the window where the PD controller's
demand exceeds the actuator limit.

At the 0.07 m box height the PD run stalls at the far lip of the gap and claws
its way off it, commanding ~46 N.m across three hip flex-ext joints in a single
~1 s burst -- roughly twice the +/-23.7 N.m those actuators can deliver. The
muscle run never exceeds any limit at any point in its trajectory.

Both traces are commanded torque, reconstructed from the logged state by
torque_stats.trial_torque(), which imports the Hill model and the PD law from
plot_leg_muscles.py. MuJoCo clamps d->ctrl to ctrlrange before applying it, so
the PD controller asked for these values and received the clipped ones -- the
claim is about demand, not delivery.

NOTE on the comparison: the two trials are different runs, so at a given wall
clock time the robots are at different points in the task. Through this window
the PD robot is recovering at the gap (x 1.05 -> 1.15, body height down to
0.251 m) while the muscle robot is already past it walking normally (x ~ 1.29).
The muscle trace here is therefore ordinary walking, not its own hardest
moment -- which is the point: its whole-run peak on hip flex-ext is 9.9 N.m.

Usage:
  python3 plot_torque_spike.py [--leg RL] [--t0 10.5] [--t1 13.5]
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_box_height import FS_AXIS, FS_LEG, FS_TICK, frame, style_legend
from plot_leg_muscles import (C_MUSCLE, C_PD, DEFAULT_PD_YAML, DEFAULT_YAML,
                              JOINT_NAMES, LEG_OFFSET, _MODEL_BASE,
                              load_task, model_joint_info)
from torque_stats import trial_torque

_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_DIR, "..", ".."))

TRIALS = os.path.join(_REPO, "analysis", "log", "trials", "workshop")
MUSCLE_TRIAL = os.path.join(TRIALS, "7muscle", "trial_017")
PD_TRIAL = os.path.join(TRIALS, "7pd", "trial_070")
OUT = os.path.join(_DIR, "figures", "torque_comparison.png")

JOINT_LABEL = {"hip": "hip ab-ad", "thigh": "hip flex-ext", "calf": "knee"}
FIGSIZE = (11.0, 5.0)
DT = 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", default="RL", choices=sorted(LEG_OFFSET))
    ap.add_argument("--t0", type=float, default=7.0)
    ap.add_argument("--t1", type=float, default=16.0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    mt = load_task(DEFAULT_YAML, "guinea_fowl")
    pt = load_task(DEFAULT_PD_YAML, "guinea_fowl")
    qadr, ctrlrange = model_joint_info(os.path.normpath(
        os.path.join(_MODEL_BASE, mt["model_path"])))

    tau = {
        "muscle": trial_torque(os.path.join(MUSCLE_TRIAL, "mppi_sim.csv"),
                               "muscle", mt["muscle"], qadr),
        "pd": trial_torque(os.path.join(PD_TRIAL, "pd_mppi_sim.csv"),
                           "pd", pt["pd"], qadr),
    }
    # trial_torque drops the first logged row (its predecessor was never
    # written), so row k is the command issued at t = (k+1)*DT.
    t = {k: (np.arange(v.shape[0]) + 1) * DT for k, v in tau.items()}

    off = LEG_OFFSET[args.leg]
    fig, axes = plt.subplots(3, 1, figsize=FIGSIZE, sharex=True)

    for ax, jn in zip(axes, JOINT_NAMES):
        col = off + JOINT_NAMES.index(jn)
        lim = ctrlrange[JOINT_NAMES.index(jn)][1]

        for sign in (+1, -1):
            ax.axhline(sign * lim, color="0.45", linewidth=1.0,
                       linestyle=(0, (5, 4)), zorder=1,
                       label="Actuator limit" if sign > 0 and jn == "hip" else None)
        ax.axhline(0, color="0.75", linewidth=0.8, zorder=0)

        for kind, colr, lbl in (("pd", C_PD, "PD"), ("muscle", C_MUSCLE, "Muscle")):
            sel = (t[kind] >= args.t0) & (t[kind] <= args.t1)
            ax.plot(t[kind][sel] * 1e3, tau[kind][sel, col], color=colr,
                    linewidth=1.6, zorder=3,
                    label=lbl if jn == "hip" else None)

        frame(ax)
        ax.set_xlim(args.t0 * 1e3, args.t1 * 1e3)
        ax.tick_params(labelsize=FS_TICK)
        # Symmetric about zero, with room for whichever is larger: the trace or
        # the limit rule. The top panel gets extra headroom on the positive side
        # only -- the legend lives in its upper-right corner and would otherwise
        # sit on the +limit rule, which is the one line that must stay readable.
        peak = max(np.abs(tau["pd"][:, col]).max(), lim) * 1.18
        ax.set_ylim(-peak, peak)

    axes[-1].set_xlabel("Time (ms)", fontsize=FS_AXIS)
    fig.supylabel("Commanded \u03c4 (N·m)", fontsize=FS_AXIS)
    leg = style_legend(axes[0], axes[0].legend(loc="upper right", ncol=3,
                                               fontsize=FS_LEG, frameon=True,
                                               handlelength=1.8, borderpad=0.5))
    fig.tight_layout()

    # Grow the top panel's positive headroom until the legend actually clears
    # the +limit rule. A fixed multiplier cannot do this: the legend is a fixed
    # number of points tall, so the fraction of the panel it covers depends on
    # the figure height, and a value tuned at one FIGSIZE overlaps at another.
    top_lim = ctrlrange[0][1]
    for _ in range(15):
        fig.canvas.draw()
        bb = leg.get_window_extent().transformed(axes[0].transData.inverted())
        if bb.y0 >= top_lim * 1.10:
            break
        lo, hi = axes[0].get_ylim()
        axes[0].set_ylim(lo, hi * 1.10)
        fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)

    print(f"leg    : {args.leg}   window {args.t0}-{args.t1}s")
    for jn in JOINT_NAMES:
        col = off + JOINT_NAMES.index(jn)
        lim = ctrlrange[JOINT_NAMES.index(jn)][1]
        sel = (t["pd"] >= args.t0) & (t["pd"] <= args.t1)
        pk = np.abs(tau["pd"][sel, col]).max()
        mk = np.abs(tau["muscle"][(t["muscle"] >= args.t0) & (t["muscle"] <= args.t1), col]).max()
        print(f"  {JOINT_LABEL[jn]:<14} PD peak {pk:6.1f}  muscle peak {mk:5.1f}"
              f"  limit {lim:5.1f}  {'OVER' if pk > lim else ''}")
    print(f"wrote  : {args.out}")


if __name__ == "__main__":
    main()
