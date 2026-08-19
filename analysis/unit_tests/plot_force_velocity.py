"""
Plot the force-velocity curve for the Hill muscle model in muscle.h.
Parameters are read from tasks.yaml (default_muscle_quad block),
one subplot per joint type (hip, thigh, calf) — values are identical across the 4 legs.

X-axis: lce_dot / vmax  (normalized fiber velocity)
Y-axis: F / F_max       (force normalized by peak isometric force)
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
FVmax_list = muscle["FVmax"][:3]   # identical across the 4 legs; only need one set

JOINT_NAMES = ["hip", "thigh", "calf"]

# ---------------------------------------------------------------------------
# Reimplementation of force_vel() from muscle.h
# ---------------------------------------------------------------------------
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
# Evaluate + plot per joint
# ---------------------------------------------------------------------------
eff_vel = np.linspace(-1.5, 2.0, 1000)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

for joint_j in range(3):
    ax = axes[joint_j]

    FVmax = FVmax_list[joint_j]
    c = FVmax - 1.0

    fv = np.array([force_vel(v, c, FVmax) for v in eff_vel])

    ax.plot(eff_vel, fv, color="#2196F3", linewidth=3, zorder=3)

    ax.axvline(0.0,  color="#4CAF50", linewidth=2.0, linestyle="--", alpha=0.9, label="Isometric")
    ax.axvline(-1.0, color="#9C27B0", linewidth=2.0, linestyle=":",  alpha=0.9, label="Max shortening")
    ax.axvline(c,    color="#E91E63", linewidth=2.0, linestyle="-.", alpha=0.9, label="Plateau onset")
    ax.axhline(1.0,  color="#607D8B", linewidth=1.5, linestyle="--", alpha=0.6)
    ax.axhline(FVmax, color="#FF9800", linewidth=2.0, linestyle="--", alpha=0.9, label="$F_{v,max}$")

    ax.set_title(f"{JOINT_NAMES[joint_j]}  FVmax={FVmax:.3f}", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("$\\dot{l}_{ce} / v_{max}$", fontsize=11)
    if joint_j == 0:
        ax.set_ylabel("$F / F_{max}$", fontsize=11)

axes[0].legend(fontsize=8, loc="upper left")

axes[0].set_xlim(-1.5, 2.0)
axes[0].set_ylim(-0.05, max(FVmax_list) * 1.15)

fig.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "force_velocity.png")
fig.savefig(out_path, dpi=150)
print(f"Saved → {out_path}")
