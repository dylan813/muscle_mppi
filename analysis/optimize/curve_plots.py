"""
Render force-length and force-velocity curve plots for a 12D CMA-ES
candidate x, for wandb logging during cmaes_walk.py runs.

FL curve logic is recycled directly from objective.py's closed-form
active/passive functions (the same ones curve_area_mean() integrates).
FV curve logic is recycled from analysis/unit_tests/plot_force_velocity.py's
force_vel(), reimplemented here since that script hardcodes tasks.yaml as
its parameter source instead of taking x.

Both plotting functions mirror the subplot layout/style of
analysis/unit_tests/plot_force_length.py and plot_force_velocity.py
(one column per joint type), just parameterized by an arbitrary x instead
of tasks.yaml, and returning a Figure instead of saving a PNG.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from objective import _active_fl, _passive_force_length, _joint_curve_area

JOINT_NAMES = ["hip", "thigh", "calf"]

# Wide enough to show the full curve regardless of where lce_min/lce_max/
# pFLmax land within cmaes_walk.py's LO/HI bounds (same domain as
# objective._AREA_DOMAIN).
_FL_DOMAIN = np.linspace(0.3, 1.9, 400)
_FV_DOMAIN = np.linspace(-1.5, 2.0, 400)


def _force_vel(eff_vel, c, FVmax):
    """Recycled from analysis/unit_tests/plot_force_velocity.py::force_vel."""
    if eff_vel < -1.0:
        return 0.0
    elif eff_vel <= 0.0:
        return (eff_vel + 1.0) ** 2
    elif eff_vel <= c:
        return FVmax - (c - eff_vel) ** 2 / c
    else:
        return FVmax


def plot_fl_curves(x, title_suffix=""):
    """x's first 9 entries are [lce_min x3, lce_max x3, pFLmax x3]."""
    lce_min, lce_max, pFLmax = x[0:3], x[3:6], x[6:9]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for j in range(3):
        ax = axes[j]
        lmin, lmax, fpm = lce_min[j], lce_max[j], pFLmax[j]
        b = 0.5 * (lmax + 1.0)

        active   = np.array([_active_fl(l, lmin, lmax) for l in _FL_DOMAIN])
        passive  = np.array([_passive_force_length(l, fpm, b) for l in _FL_DOMAIN])
        combined = active + passive
        area     = _joint_curve_area(lmin, lmax, fpm)

        ax.plot(_FL_DOMAIN, active,   label="Active",   color="#2196F3", linewidth=2)
        ax.plot(_FL_DOMAIN, passive,  label="Passive",  color="#FF9800", linewidth=2, linestyle="--")
        ax.plot(_FL_DOMAIN, combined, label="Combined", color="#4CAF50", linewidth=2, linestyle=":")
        # Highlight the region being maximized: where active exceeds passive.
        ax.fill_between(_FL_DOMAIN, passive, active, where=active > passive,
                         color="#4CAF50", alpha=0.15, interpolate=True)

        ax.axvline(lmin, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axvline(1.0,  color="gray", linewidth=0.8, linestyle="-",  alpha=0.6)
        ax.axvline(lmax, color="gray", linewidth=0.8, linestyle="-.", alpha=0.6)

        ax.set_title(f"{JOINT_NAMES[j]}\nlce=[{lmin:.3f}, {lmax:.3f}]  "
                     f"pFLmax={fpm:.3f}  area={area:.3f}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("$l_{ce} / l_{opt}$", fontsize=11)
        if j == 0:
            ax.set_ylabel("$F / F_{max}$", fontsize=11)
            ax.legend(fontsize=9, loc="upper right")

    # Shared view window sized to this candidate's actual range, so the
    # curve is never clipped regardless of where lce_min/lce_max landed.
    x_lo = min(lce_min) - 0.15
    x_hi = max(lce_max) + 0.15
    # Fixed y-limit (matches plot_force_length.py) rather than autoscale:
    # when lce_max sits very close to 1.0, the passive curve's slope past
    # its crossover blows up over _FL_DOMAIN's wide range, which would
    # otherwise squash the interesting near-1.0 region into an unreadable
    # sliver at the bottom of the plot.
    for ax in axes:
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(-0.05, 2.5)

    fig.suptitle(f"Force–Length Curves{title_suffix}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return fig


def plot_fv_curves(x, title_suffix=""):
    """x's FVmax entries are x[9:12]."""
    FVmax_list = x[9:12]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for j in range(3):
        ax = axes[j]
        FVmax = FVmax_list[j]
        c = FVmax - 1.0

        fv = np.array([_force_vel(v, c, FVmax) for v in _FV_DOMAIN])
        ax.plot(_FV_DOMAIN, fv, color="#2196F3", linewidth=3, zorder=3)

        ax.axvline(0.0,  color="#4CAF50", linewidth=2.0, linestyle="--", alpha=0.9, label="Isometric")
        ax.axvline(-1.0, color="#9C27B0", linewidth=2.0, linestyle=":",  alpha=0.9, label="Max shortening")
        ax.axvline(c,    color="#E91E63", linewidth=2.0, linestyle="-.", alpha=0.9, label="Plateau onset")
        ax.axhline(1.0,   color="#607D8B", linewidth=1.5, linestyle="--", alpha=0.6)
        ax.axhline(FVmax, color="#FF9800", linewidth=2.0, linestyle="--", alpha=0.9, label="$F_{v,max}$")

        ax.set_title(f"{JOINT_NAMES[j]}  FVmax={FVmax:.3f}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("$\\dot{l}_{ce} / v_{max}$", fontsize=11)
        if j == 0:
            ax.set_ylabel("$F / F_{max}$", fontsize=11)
            ax.legend(fontsize=8, loc="upper left")

    axes[0].set_xlim(-1.5, 2.0)
    axes[0].set_ylim(-0.05, max(FVmax_list) * 1.15)

    fig.suptitle(f"Force–Velocity Curves{title_suffix}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return fig
