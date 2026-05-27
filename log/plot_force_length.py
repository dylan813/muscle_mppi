"""
Plot the active, passive, and combined force-length curves for the Hill muscle model
implemented in muscle.h. Parameters are read from tasks.yaml (default_muscle block).

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
yaml_path = os.path.join(os.path.dirname(__file__), "../muscle_mppi/utils/tasks.yaml")
with open(yaml_path) as f:
    cfg = yaml.safe_load(f)

muscle = cfg["default_muscle"]
lce_min = muscle["lce_min"][0]   # same for all joints
lce_max = muscle["lce_max"][0]
fpmax   = muscle["pflmax"][0]

b_passive = 0.5 * (lce_max + 1.0)  # matches muscle.h

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
# Evaluate over a range that shows both curves fully
# ---------------------------------------------------------------------------
lce = np.linspace(0.5, 1.4, 500)

fl_active  = np.array([active_fl(l, lce_min, lce_max) for l in lce])
fl_passive = np.array([passive_force_length(l, fpmax, b_passive) for l in lce])
fl_total   = fl_active + fl_passive

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(lce, fl_active,  label="Active FL",           color="#2196F3", linewidth=2)
ax.plot(lce, fl_passive, label="Passive FL",           color="#FF9800", linewidth=2, linestyle="--")
ax.plot(lce, fl_total,   label="Combined (Active + Passive)", color="#4CAF50", linewidth=2, linestyle=":")

# Mark key landmarks
ax.axvline(lce_min,   color="gray", linewidth=0.8, linestyle="--", alpha=0.6, label=f"lce_min = {lce_min}")
ax.axvline(1.0,       color="gray", linewidth=0.8, linestyle="-",  alpha=0.6, label="l_opt = 1.0")
ax.axvline(lce_max,   color="gray", linewidth=0.8, linestyle="-.", alpha=0.6, label=f"lce_max = {lce_max}")

ax.set_xlabel("Muscle length  $l_{ce} / l_{opt}$", fontsize=13)
ax.set_ylabel("Force  $F / F_{max}$", fontsize=13)
ax.set_title("Force–Length Curves", fontsize=14)
ax.legend(fontsize=10)
half_range = 0.35
ax.set_xlim(1.0 - half_range, 1.0 + half_range)
ax.set_ylim(-0.05, 1.6)
ax.grid(True, alpha=0.3)

out_path = os.path.join(os.path.dirname(__file__), "force_length.png")
fig.tight_layout()
fig.savefig(out_path, dpi=150)
print(f"Saved → {out_path}")
