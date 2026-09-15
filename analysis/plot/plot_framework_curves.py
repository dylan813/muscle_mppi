"""
Three schematic curves for the control-framework figure: activation dynamics,
active force-length, and force-velocity.

Deliberately unquantified — axis labels only, no tick numbers — so each panel
drops into a block diagram as a shape rather than as a data plot. The curves are
still the real ones: every function below mirrors controllers/muscle/control/muscle.h,
and the parameters come from tasks.yaml.

Writes three SVGs to figures/:
    curve_activation.svg
    curve_force_length.svg
    curve_force_velocity.svg

Run from anywhere; paths are resolved relative to this file.
"""

import os

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ─────────────────────────────────────────────────────────────────────
_DIR       = os.path.dirname(os.path.abspath(__file__))
YAML_PATH  = os.path.join(_DIR, "../../controllers/muscle/utils/tasks.yaml")

# Joint whose parameters shape the curves. 0 = hip, 1 = thigh, 2 = calf.
# Hip has the widest [lce_min, lce_max] span, so its force-length bell is the
# most legible; switch this to redraw the set for another joint.
JOINT = 0

ACCENT = "#c0392b"
CMD    = "0.45"

# ── config ────────────────────────────────────────────────────────────────────
with open(YAML_PATH) as f:
    cfg = yaml.safe_load(f)

m       = cfg["default_muscle_quad"]
F_ACT   = m["act_bandwidth"]
DT      = cfg["walk"]["dt"]
ALPHA   = F_ACT * DT
LCE_MIN = m["lce_min"][JOINT]
LCE_MAX = m["lce_max"][JOINT]
FV_MAX  = m["FVmax"][JOINT]
PFL_MAX = m["pFLmax"][JOINT]


# ── model functions (muscle.h) ────────────────────────────────────────────────
def afl_bump(lce, A, mid, B):
    """Piecewise-quadratic bump, muscle.h:13-31."""
    lce = np.asarray(lce, dtype=float)
    left, right = 0.5 * (A + mid), 0.5 * (mid + B)
    out = np.zeros_like(lce)
    s = (lce > A) & (lce < left)
    out[s] = 0.5 * ((lce[s] - A) / (left - A)) ** 2
    s = (lce >= left) & (lce < mid)
    out[s] = 1.0 - 0.5 * ((mid - lce[s]) / (mid - left)) ** 2
    s = (lce >= mid) & (lce < right)
    out[s] = 1.0 - 0.5 * ((lce[s] - mid) / (right - mid)) ** 2
    s = (lce >= right) & (lce < B)
    out[s] = 0.5 * ((B - lce[s]) / (B - right)) ** 2
    return out


def active_fl(lce):
    """Primary bell plus the 15% short-length shoulder, muscle.h:152-155."""
    return (afl_bump(lce, LCE_MIN, 1.0, LCE_MAX)
            + 0.15 * afl_bump(lce, LCE_MIN, 0.5 * (LCE_MIN + 0.95), 0.95))


def passive_fl(lce):
    """Passive limb, muscle.h -- zero below l_opt, cubic then linear above.
    b is the same 0.5*(lce_max + 1) breakpoint plot_hill_curves.py uses."""
    lce = np.asarray(lce, dtype=float)
    b = 0.5 * (LCE_MAX + 1.0)
    out = np.zeros_like(lce)
    s = (lce > 1.0) & (lce <= b)
    out[s] = 0.25 * PFL_MAX * ((lce[s] - 1.0) / (b - 1.0)) ** 3
    s = lce > b
    out[s] = 0.25 * PFL_MAX * (1.0 + 3.0 * (lce[s] - b) / (b - 1.0))
    return out


def total_fl(lce):
    """What the Hill model actually applies: active + passive."""
    return active_fl(lce) + passive_fl(lce)


def force_vel(vhat):
    """Force-velocity in normalized velocity vhat = lce_dot/vmax, muscle.h:36-42."""
    vhat = np.asarray(vhat, dtype=float)
    c = FV_MAX - 1.0
    out = np.full_like(vhat, FV_MAX)
    out[vhat < -1.0] = 0.0
    s = (vhat >= -1.0) & (vhat <= 0.0)
    out[s] = (vhat[s] + 1.0) ** 2
    s = (vhat > 0.0) & (vhat <= c)
    out[s] = FV_MAX - (c - vhat[s]) ** 2 / c
    return out


def act_filter(u, alpha=ALPHA, a0=0.0):
    """Activation filter, muscle.h:140-145."""
    a, out = a0, np.empty_like(u, dtype=float)
    for n, ui in enumerate(u):
        a = np.clip(a + alpha * (np.clip(ui, 0.0, 1.0) - a), 0.0, 1.0)
        out[n] = a
    return out


# ── shared styling ────────────────────────────────────────────────────────────
def new_axes(xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.2)
        ax.spines[side].set_color("0.2")
    return fig, ax


def save(fig, stem):
    fig.tight_layout()
    os.makedirs(os.path.join(_DIR, "figures"), exist_ok=True)
    fig.savefig(os.path.join(_DIR, "figures", f"{stem}.svg"), transparent=False)
    plt.close(fig)
    print(f"wrote figures/{stem}.svg")


# ── 1. activation dynamics ────────────────────────────────────────────────────
t = np.arange(220) * DT
u = np.where((t >= 0.2) & (t < 1.0), 1.0, 0.0)
fig, ax = new_axes("Time", "Activation")
ax.step(t, u, where="post", color=CMD, lw=1.8, ls="--")
ax.plot(t, act_filter(u), color=ACCENT, lw=2.6)
ax.text(0.62, 1.06, "$u$", color=CMD, fontsize=13, ha="center")
ax.text(0.62, 0.80, "$a$", color=ACCENT, fontsize=13, ha="center")
ax.set_xlim(0, t[-1])
ax.set_ylim(-0.06, 1.25)
save(fig, "curve_activation")

# ── 2. active force-length ────────────────────────────────────────────────────
pad = 0.12 * (LCE_MAX - LCE_MIN)
lce = np.linspace(LCE_MIN - pad, LCE_MAX + pad, 900)
fig, ax = new_axes("Muscle length", "Force")
ax.plot(lce, total_fl(lce), color=ACCENT, lw=2.6)
ax.set_xlim(lce[0], lce[-1])
ax.set_ylim(-0.06, max(1.22, total_fl(lce).max() * 1.12))
save(fig, "curve_force_length")

# ── 3. force-velocity ─────────────────────────────────────────────────────────
vhat = np.linspace(-1.35, 0.75, 900)
fig, ax = new_axes("Muscle velocity", "Force")
ax.plot(vhat, force_vel(vhat), color=ACCENT, lw=2.6)
ax.set_xlim(vhat[0], vhat[-1])
ax.set_ylim(-0.06, 1.25)
save(fig, "curve_force_velocity")

