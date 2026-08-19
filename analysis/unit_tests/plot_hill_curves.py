"""
Plot both Hill-muscle characteristic curves in one figure: force-length on the top
row, force-velocity on the bottom, one column per joint type (hip, thigh, calf).
Parameters come from tasks.yaml (default_muscle_quad block) — values are identical
across the 4 legs, so one set covers all of them.

Top row    X: lce / l_opt          (fiber length normalized by optimal length)
Bottom row X: lce_dot / vmax       (normalized fiber velocity)
Both rows  Y: F / F_max            (force normalized by peak isometric force)

This is the combined version of plot_force_length.py and plot_force_velocity.py,
which remain as the standalone single-figure scripts. The muscle.h helpers are
duplicated here because those two run their plotting at module level and so can't
be imported.

Usage:
  python3 plot_hill_curves.py        ->  hill_curves.png beside this file
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

JOINT_NAMES = ["Hip", "Thigh", "Calf"]   # column order; tagged (a)/(b)/(c) in the figure
SUBFIG_TAGS = ["(a)", "(b)", "(c)"]

# The primary curve of each row — force-length Total and the force-velocity
# curve — shares one style so the two rows read as the same kind of object.
C_CURVE  = "#64B5F6"
LW_CURVE = 3

# Dashed annotation lines share one weight across both rows.
LW_DASH = 1.5

# One legend size for both rows.
LEGEND_FS = 10

FIG_W = 15.0   # inches of figure width
FIG_H = 9.0    # inches of figure height

# Left margin, in inches, holding the leftmost panel's ylabel and tick labels.
AXES_LEFT_IN = 0.75


# ---------------------------------------------------------------------------
# Reimplementation of the muscle.h helpers
# ---------------------------------------------------------------------------
def active_force_length(length, A, mid, B):
    left  = 0.5 * (A + mid)
    right = 0.5 * (mid + B)
    if length <= A or length >= B:
        return 0.0
    if length < left:
        temp = (length - A) / (left - A)
        return 0.5 * temp * temp
    elif length < mid:
        temp = (mid - length) / (mid - left)
        return 1.0 - 0.5 * temp * temp
    elif length < right:
        temp = (length - mid) / (right - mid)
        return 1.0 - 0.5 * temp * temp
    else:
        temp = (B - length) / (B - right)
        return 0.5 * temp * temp


def active_fl(lce, lmin, lmax):
    return (active_force_length(lce, lmin, 1.0, lmax)
            + 0.15 * active_force_length(lce, lmin, 0.5 * (lmin + 0.95), 0.95))


def passive_force_length(lce, fpmax, b):
    if lce <= 1.0:
        return 0.0
    elif lce <= b:
        temp = (lce - 1.0) / (b - 1.0)
        return 0.25 * fpmax * temp ** 3
    else:
        temp = (lce - b) / (b - 1.0)
        return 0.25 * fpmax * (1.0 + 3.0 * temp)


def force_vel(eff_vel, c, FVmax):
    if eff_vel < -1.0:
        return 0.0
    elif eff_vel <= 0.0:
        return (eff_vel + 1.0) ** 2
    elif eff_vel <= c:
        return FVmax - (c - eff_vel) ** 2 / c
    else:
        return FVmax


# ---------------------------------------------------------------------------
# Per-row plotting
# ---------------------------------------------------------------------------
def plot_force_length_row(axes, lce_min, lce_max, fpmax):
    lce = np.linspace(0.5, 1.4, 500)

    for j, ax in enumerate(axes):
        lmin, lmax, fpm = lce_min[j], lce_max[j], fpmax[j]
        b_passive = 0.5 * (lmax + 1.0)   # matches muscle.h

        fl_active  = np.array([active_fl(l, lmin, lmax) for l in lce])
        fl_passive = np.array([passive_force_length(l, fpm, b_passive) for l in lce])

        # Total sits underneath so the two dashed components stay visible
        # where they coincide with it (passive is zero below l_opt).
        ax.plot(lce, fl_active,              label="Active",  color="#E8890C", linewidth=LW_DASH, linestyle="--", zorder=3)
        ax.plot(lce, fl_passive,             label="Passive", color="#00875a", linewidth=LW_DASH, linestyle="--", zorder=3)
        ax.plot(lce, fl_active + fl_passive, label="Total",   color=C_CURVE, linewidth=LW_CURVE, zorder=2)

        # Optimal fiber length only; the lce_min/lce_max rules are left off.
        ax.axvline(1.0, color="gray", linewidth=0.8, linestyle="-", alpha=0.6)

        ax.grid(True, alpha=0.3)
        ax.set_xlabel("$l_{ce} / l_{opt}$", fontsize=11)

    axes[0].set_ylabel("$F / F_{max}$", fontsize=11)
    axes[0].legend(fontsize=LEGEND_FS, loc="upper right")
    axes[0].set_xlim(0.6, 1.4)
    axes[0].set_ylim(-0.05, 2.4)


def plot_force_velocity_row(axes, FVmax_list):
    eff_vel = np.linspace(-1.5, 2.0, 1000)

    for j, ax in enumerate(axes):
        FVmax = FVmax_list[j]
        c = FVmax - 1.0

        fv = np.array([force_vel(v, c, FVmax) for v in eff_vel])
        ax.plot(eff_vel, fv, color=C_CURVE, linewidth=LW_CURVE, zorder=3)

        # Isometric and unit-force references, both drawn like the l_opt rule
        # on the force-length row.
        ax.axvline(0.0, color="gray", linewidth=0.8, linestyle="-", alpha=0.6)
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="-", alpha=0.6)
        ax.axhline(FVmax, color="#E53935", linewidth=LW_DASH, linestyle="--", label="$F_{v,max}$")

        ax.grid(True, alpha=0.3)
        ax.set_xlabel("$\\dot{l}_{ce} / v_{max}$", fontsize=11)

    axes[0].set_ylabel("$F / F_{max}$", fontsize=11)
    axes[0].legend(fontsize=LEGEND_FS, loc="upper right")
    axes[0].set_xlim(-1.5, 2.0)
    axes[0].set_ylim(-0.05, max(FVmax_list) * 1.15)


def main():
    yaml_path = os.path.join(os.path.dirname(__file__),
                             "../../muscle_mppi/utils/tasks.yaml")
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    muscle = cfg["default_muscle_quad"]

    fig, axes = plt.subplots(2, 3, figsize=(FIG_W, FIG_H))
    # Rows share a y-scale within themselves but not with each other, and each
    # row has its own x quantity, so sharing is set up per row rather than
    # through subplots(sharex=..., sharey=...).
    for row in axes:
        for ax in row[1:]:
            ax.sharex(row[0])
            ax.sharey(row[0])
            ax.tick_params(labelleft=False)

    plot_force_length_row(axes[0], muscle["lce_min"][:3], muscle["lce_max"][:3],
                          muscle["pFLmax"][:3])
    plot_force_velocity_row(axes[1], muscle["FVmax"][:3])

    fig.subplots_adjust(left=AXES_LEFT_IN / FIG_W, right=0.985,
                        top=1 - 0.40 / FIG_H, bottom=0.90 / FIG_H,
                        hspace=0.28, wspace=0.08)

    # One title centred over each row, rather than repeated per panel. The joint
    # each column belongs to is carried by the (a)/(b)/(c) tags along the bottom.
    for row, label in zip(axes, ("Force–Length Relationship",
                                 "Force–Velocity Relationship")):
        left, right = row[0].get_position(), row[-1].get_position()
        fig.text(0.5 * (left.x0 + right.x1), left.y1 + 0.10 / FIG_H, label,
                 ha="center", va="bottom", fontsize=14)

    for ax, tag in zip(axes[-1], SUBFIG_TAGS):
        box = ax.get_position()
        fig.text(0.5 * (box.x0 + box.x1), 0.20 / FIG_H, tag,
                 ha="center", va="center", fontsize=13)

    out_path = os.path.join(os.path.dirname(__file__), "hill_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
