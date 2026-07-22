"""
Plot the active, passive, and combined force-length curves for the Hill muscle model
implemented in muscle.h. Parameters are read from tasks.yaml (default_muscle_quad block),
one subplot per joint type (hip, thigh, calf) — values are identical across the 4 legs.

X-axis: lce / l_opt  (fiber length normalized by optimal isometric length)
Y-axis: F / F_max    (force normalized by peak isometric force)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
import os

# ---------------------------------------------------------------------------
# Load parameters from tasks.yaml
# ---------------------------------------------------------------------------
yaml_path = os.path.join(os.path.dirname(__file__), "../../muscle_mppi/utils/tasks.yaml")
with open(yaml_path) as f:
    cfg = yaml.safe_load(f)

muscle = cfg["default_muscle_quad"]
lce_min = muscle["lce_min"][:3]   # identical across the 4 legs; only need one set
lce_max = muscle["lce_max"][:3]
fpmax   = muscle["pFLmax"][:3]

JOINT_NAMES = ["hip", "thigh", "calf"]

# ---------------------------------------------------------------------------
# Reimplementation of muscle.h helpers
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


# ---------------------------------------------------------------------------
# Evaluate + plot per joint
# ---------------------------------------------------------------------------
lce = np.linspace(0.5, 1.4, 500)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

for joint_j in range(3):
    ax = axes[joint_j]

    lmin = lce_min[joint_j]
    lmax = lce_max[joint_j]
    fpm  = fpmax[joint_j]
    b_passive = 0.5 * (lmax + 1.0)  # matches muscle.h

    fl_active  = np.array([active_fl(l, lmin, lmax) for l in lce])
    fl_passive = np.array([passive_force_length(l, fpm, b_passive) for l in lce])
    fl_total   = fl_active + fl_passive

    ax.plot(lce, fl_active,  label="Active",   color="#2196F3", linewidth=2)
    ax.plot(lce, fl_passive, label="Passive",  color="#FF9800", linewidth=2, linestyle="--")
    ax.plot(lce, fl_total,   label="Combined", color="#4CAF50", linewidth=2, linestyle=":")

    ax.axvline(lmin, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(1.0,  color="gray", linewidth=0.8, linestyle="-",  alpha=0.6)
    ax.axvline(lmax, color="gray", linewidth=0.8, linestyle="-.", alpha=0.6)

    ax.set_title(f"{JOINT_NAMES[joint_j]}\nlce=[{lmin:.3f}, {lmax:.3f}]  pFLmax={fpm:.3f}", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("$l_{ce} / l_{opt}$", fontsize=11)
    if joint_j == 0:
        ax.set_ylabel("$F / F_{max}$", fontsize=11)

axes[0].legend(fontsize=9, loc="upper right")

half_range = 0.4
axes[0].set_xlim(1.0 - half_range, 1.0 + half_range)
axes[0].set_ylim(-0.05, 2.4)

fig.suptitle("Force–Length Curves (default_muscle_quad, per joint type)", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.93])

out_path = os.path.join(os.path.dirname(__file__), "force_length.png")
fig.savefig(out_path, dpi=150)
print(f"Saved → {out_path}")
