"""
Agonist/antagonist activation for the FR hip ab-ad joint, from one saved trial.

Redraws muscle_FR_activations.png as a single-joint figure in the same visual
style as the box-height sweep set (fig_success / fig_torque_*): identical type
scale, canvas size, frame and legend treatment, all imported from
plot_box_height.py rather than duplicated, so the four figures cannot drift.

Differences from plot_walk_leg.py, which drew the original:
  * one joint, not three -- so the left-gutter row labels are gone
  * the rotated "Activation Signal" gutter text is replaced by a conventional
    y-axis label; a single-panel figure has no gutter to hang it in
  * data comes from a saved trial directory rather than the working CSV

Default trial is 7muscle/trial_017 -- box height 0.07, the batch with the
strongest muscle-vs-PD separation (Fisher p = 1.4e-4, vs 0.0016 at 0.06 and
non-significant at 0.05/0.08), and the trial whose time-to-goal is exactly the
median of that batch's 37 successes, so it is representative rather than picked.

Usage:
  python3 plot_hip_activation.py [trial_dir] [--joint 0|1|2] [--out <path.png>]
"""

import argparse
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_box_height import FS_AXIS, FS_LEG, FS_TICK, frame, style_legend

# Only the type scale is shared with the sweep figures -- the canvas is this
# figure's own. A time series wants the width plot_walk_leg.py gave it (11 in);
# borrowing the sweep figures' near-square canvas squeezed 24 s of activation
# into a third of the horizontal room.
FIGSIZE = (11.0, 4.2)

_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_DIR, "..", ".."))

DEFAULT_TRIAL = os.path.join(_REPO, "analysis", "log", "trials", "workshop", "7muscle", "trial_017")
DEFAULT_OUT = os.path.join(_DIR, "figures", "muscle_FR_activations.png")

LEG_OFFSET = 0            # FR is the first leg in actuator order
JOINT_LABEL = ["hip ab-ad", "hip flex-ext", "knee"]

# Agonist/antagonist, unchanged from plot_walk_leg.py so this figure still
# matches the rest of the activation set. The pair separates well (CVD dE 22.0
# protan, 33.8 normal-vision); both sit below the validator's lightness band,
# which costs a little legibility on screen but nothing on identity.
C_AGONIST = "navy"
C_ANTAGONIST = "darkred"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trial_dir", nargs="?", default=DEFAULT_TRIAL)
    ap.add_argument("--joint", type=int, default=0, choices=(0, 1, 2),
                    help="0 = hip ab-ad (default), 1 = hip flex-ext, 2 = knee")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    csv_path = os.path.join(args.trial_dir, "mppi_sim.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"no mppi_sim.csv in {args.trial_dir}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    joint = LEG_OFFSET + args.joint
    ag, ant = 2 * joint, 2 * joint + 1
    t = df["t"].to_numpy() * 1e3

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(t, df[f"act_m{ag}"].to_numpy(), color=C_AGONIST,
            linewidth=1.2, label="Agonist")
    ax.plot(t, df[f"act_m{ant}"].to_numpy(), color=C_ANTAGONIST,
            linewidth=1.2, label="Antagonist")

    frame(ax)
    # Activation is bounded [0, 1], so the ticks stop at 1.0 and the extra
    # headroom is purely somewhere for the legend to sit -- both traces peak
    # near 0.95 late in the run and were passing behind the legend box.
    ax.set_ylim(-0.04, 1.30)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # Padding at both ends rather than clamping to the first and last sample,
    # matching torque_comparison.png -- a trace butted against the spine reads
    # as clipped.
    ax.margins(x=0.025)
    ax.set_xlabel("Time (ms)", fontsize=FS_AXIS)
    ax.set_ylabel("Activation signal", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    style_legend(ax, ax.legend(loc="upper right", fontsize=FS_LEG,
                               frameon=True, handlelength=1.8))
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)

    print(f"trial : {os.path.relpath(args.trial_dir, _REPO)}  ({len(df)} rows)")
    print(f"joint : FR {JOINT_LABEL[args.joint]}  (act_m{ag} / act_m{ant})")
    print(f"wrote : {args.out}")


if __name__ == "__main__":
    main()
