"""
Plot per-joint muscle activation and joint velocity for the FR leg during
the "walk" task, from mppi_sim's output CSV (t, px..qw, roll_deg, dq_j*, act_m*).

Produces two PNGs in figures/ beside this script (or --outdir):
  <name>_activations.png  — agonist/antagonist activation per joint
  <name>_velocities.png   — joint velocity per joint

Usage:
  python plot_walk_leg.py [csv_path] [name] [--outdir DIR]
  python plot_walk_leg.py [name]           (csv_path defaults; name-only shorthand,
                                             only when the single arg isn't an existing file)
    name     output filename prefix (default "walk_FR")
    --outdir output directory (pass the trial directory to keep the plots with its data)
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

JOINT_NAMES = ["hip", "thigh", "calf"]
LEG    = "FR"
OFFSET = 0  # first joint index of the FR leg

FIG_W = 11.0   # inches of figure width
FIG_H = 7.2    # inches of figure height

# Left-gutter geometry, in inches from the figure's left edge, matching
# plot_leg_muscles.py. The joint label sits furthest out, the units label
# nearest the plots, and the tick labels fill the gap to the axes.
LABEL_X_IN   = 0.85
UNITS_X_IN   = 1.15
LEGEND_X_IN  = 1.25
AXES_LEFT_IN = 1.70


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def layout(fig, axes, units_label: str, legend: bool = False):
    """Explicit margins plus left-gutter labels, in place of per-panel titles."""
    fig.subplots_adjust(left=AXES_LEFT_IN / FIG_W, right=0.985,
                        top=1 - 0.15 / FIG_H, bottom=0.55 / FIG_H, hspace=0.10)

    fig.text(UNITS_X_IN / FIG_W, 0.5 * (0.55 / FIG_H + 1 - 0.15 / FIG_H),
             units_label, rotation=90, ha="center", va="center", fontsize=11)
    for ax, jname in zip(axes, JOINT_NAMES):
        box = ax.get_position()
        fig.text(LABEL_X_IN / FIG_W, 0.5 * (box.y0 + box.y1), jname.capitalize(),
                 ha="right", va="center", fontsize=11)

    if legend:
        handles, labels = axes[0].get_legend_handles_labels()
        leg = fig.legend(handles, labels, loc="lower right", ncol=1,
                         fontsize=9, frameon=True, handlelength=1.6,
                         labelspacing=0.4, borderaxespad=0, borderpad=0.6,
                         bbox_to_anchor=(LEGEND_X_IN / FIG_W, 0.08 / FIG_H))
        leg.get_frame().set_linewidth(0.8)
        leg.get_frame().set_edgecolor("black")
        leg.get_frame().set_facecolor("white")


def plot_activations(df: pd.DataFrame, out_dir: str, name: str):
    t = df["t"].to_numpy() * 1e3
    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, FIG_H), sharex=True)

    for j in range(3):
        joint = OFFSET + j
        ag, ant = 2 * joint, 2 * joint + 1
        ax = axes[j]
        ax.plot(t, df[f"act_m{ag}"].to_numpy(),  color="navy",     linewidth=1.2, label="Agonist")
        ax.plot(t, df[f"act_m{ant}"].to_numpy(), color="darkred",  linewidth=1.2, label="Antagonist")
        ax.set_ylim(-0.05, 1.05)

    axes[-1].set_xlabel("Time (ms)")
    layout(fig, axes, "Activation Signal", legend=True)
    out_path = os.path.join(out_dir, f"{name}_activations.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_velocities(df: pd.DataFrame, out_dir: str, name: str):
    t = df["t"].to_numpy() * 1e3
    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, FIG_H), sharex=True)

    for j in range(3):
        joint = OFFSET + j
        ax = axes[j]
        ax.plot(t, df[f"dq_j{joint}"].to_numpy(), color="seagreen", linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

    axes[-1].set_xlabel("Time (ms)")
    layout(fig, axes, "Velocity (rad/s)")
    out_path = os.path.join(out_dir, f"{name}_velocities.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    default_csv = os.path.join(os.path.dirname(__file__), "..",
                                "data", "mppi_sim", "mppi_sim.csv")
    default_name = f"walk_{LEG}"

    args = sys.argv[1:]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    if "--outdir" in args:
        i = args.index("--outdir")
        if i + 1 >= len(args):
            sys.exit("--outdir needs a directory")
        out_dir = args[i + 1]
        del args[i:i + 2]

    if len(args) >= 2:
        csv_path, name = args[0], args[1]
    elif len(args) == 1:
        # Single arg: treat as csv_path if it's an existing file, else as name.
        if os.path.isfile(args[0]):
            csv_path, name = args[0], default_name
        else:
            csv_path, name = default_csv, args[0]
    else:
        csv_path, name = default_csv, default_name

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    df = load(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    plot_activations(df, out_dir, name)
    plot_velocities(df, out_dir, name)
    print("Done.")


if __name__ == "__main__":
    main()
