"""
Plot the force-velocity curve for the Hill muscle model in muscle.h.
Parameters are read from tasks.yaml (default_muscle block).

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
yaml_path = os.path.join(os.path.dirname(__file__), "../muscle_mppi/utils/tasks.yaml")
with open(yaml_path) as f:
    cfg = yaml.safe_load(f)

muscle = cfg["default_muscle"]
FVmax = muscle["FVmax"][0]   # same for all joints
c = FVmax - 1.0

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
# Evaluate over normalised velocity range
# ---------------------------------------------------------------------------
eff_vel = np.linspace(-1.5, 2.0, 1000)
fv = np.array([force_vel(v, c, FVmax) for v in eff_vel])

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(eff_vel, fv, color="#2196F3", linewidth=2)

ax.axvline(0.0,  color="gray", linewidth=0.8, linestyle="--", alpha=0.6, label="Isometric ($\\dot{l}_{ce}=0$)")
ax.axvline(-1.0, color="gray", linewidth=0.8, linestyle=":",  alpha=0.6, label="Max shortening ($\\dot{l}_{ce}/v_{max}=-1$)")
ax.axvline(c,    color="gray", linewidth=0.8, linestyle="-.", alpha=0.6, label=f"Eccentric plateau onset ($c={c:.2f}$)")
ax.axhline(1.0,  color="gray", linewidth=0.8, linestyle="--", alpha=0.4)
ax.axhline(FVmax, color="#FF9800", linewidth=1.0, linestyle="--", alpha=0.7, label=f"$F_{{v,max}}={FVmax}$")

ax.set_xlabel("Muscle velocity  $\\dot{l}_{ce} / v_{max}$", fontsize=13)
ax.set_ylabel("Force  $F / F_{max}$", fontsize=13)
ax.set_title("Force–Velocity Curve", fontsize=14)
ax.legend(fontsize=10)
ax.set_xlim(-1.5, 2.0)
ax.set_ylim(-0.05, FVmax * 1.15)
ax.grid(True, alpha=0.3)

out_path = os.path.join(os.path.dirname(__file__), "force_velocity.png")
fig.tight_layout()
fig.savefig(out_path, dpi=150)
print(f"Saved → {out_path}")
