"""
Generate muscle activation gait files from joint-space gait trajectories.

Extends plot_stiffness_cocontraction.py from a single static pose to the full
gait cycle. For each phase t in a source gait TSV (rows 0-11 = joint positions,
rows 12-23 = joint velocities), solves the torque-balance constraint line:

    FL1(t)·FV1(t)·a1  −  FL2(t)·FV2(t)·a2  =  C(t)

where:
    C(t) = tau_req(t) / (−r · F_peak)  −  (P1(t) − P2(t))
    tau_req(t) = qfrc_bias from MuJoCo (gravity + Coriolis + centrifugal)
    FL1/FL2 = active force-length at current joint angle
    FV1/FV2 = force-velocity at current joint velocity (key extension vs. static)
    P1/P2   = passive force-length

The midpoint of the feasible segment of this line on [0,1]² gives the nominal
activation pair (a1*(t), a2*(t)) for each joint at each gait phase.

Output: 24 × N TSV files saved to muscle_mppi/muscle_mppi/gaits/
    rows  0-11 = a1 (agonist)  per joint  [FR_hip, FR_thigh, FR_calf, FL..., RR..., RL...]
    rows 12-23 = a2 (antagonist) per joint (same order)

Source gait TSVs: the 4 files actually used by RTWholeBodyMPPI (in_place, walk,
walk_fast, trot). These carry joint positions/velocities for a Go1 robot but are
applied to the Go2 model here — treat as a gait pattern template, not an exact
kinematic match.

Run from anywhere; paths are resolved relative to this file.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
import os
import mujoco

# ── paths ─────────────────────────────────────────────────────────────────────
_DIR       = os.path.dirname(os.path.abspath(__file__))
YAML_PATH  = os.path.join(_DIR,  "../../muscle_mppi/utils/tasks.yaml")
MODEL_PATH = os.path.join(_DIR,  "../../unitree_mujoco/unitree_robots/go2/scene.xml")
OUT_DIR    = os.path.join(_DIR,  "../../muscle_mppi/gaits")
GAIT_DIR   = os.path.join(_DIR,  "../../../RTWholeBodyMPPI/legged_mppi/"
                                  "whole_body_mppi/control/gait_scheduler/gaits")

os.makedirs(OUT_DIR, exist_ok=True)

# ── full gait library: 3 tiers × 4 velocities × 3 heights = 36 files ─────────
# Only 100hz variants — 80hz files are skipped.
# TIER:   SLOW / MED / FAST — gait cadence pattern (step shape and crouch depth)
# vel:    forward velocity encoded in the trajectory (0.0, 0.1, 0.2, 0.5 m/s)
# height: foot clearance during swing (10cm, 15cm, 20cm)
#
# RTWholeBodyMPPI task → gait mapping (the 4 hardcoded in mppi_locomotion.py):
#   in_place  → FAST_0_0_10cm   (stand/wait, zero velocity)
#   walk      → MED_0_1_10cm    (careful/slow — stairs, obstacles)
#   walk_fast → FAST_0_1_10cm   (general flat-terrain locomotion)
#   trot      → MED_0_5_15cm    (aggressive — box climbing, fast terrain)
TIERS      = ["SLOW", "MED", "FAST"]
VELOCITIES = ["0_0", "0_1", "0_2", "0_5"]
HEIGHTS    = ["10cm", "15cm", "20cm"]

GAIT_FILES = {}   # key → (path, tier, vel, height)
for tier in TIERS:
    for vel in VELOCITIES:
        for height in HEIGHTS:
            fname = f"walking_gait_raibert_{tier}_{vel}_{height}_100hz.tsv"
            fpath = os.path.join(GAIT_DIR, tier, fname)
            if os.path.exists(fpath):
                GAIT_FILES[f"{tier}_{vel}_{height}"] = (fpath, tier, vel, height)

print(f"Found {len(GAIT_FILES)} gait files to process.")

# ── load muscle parameters from tasks.yaml ────────────────────────────────────
with open(YAML_PATH) as f:
    cfg = yaml.safe_load(f)

muscle   = cfg["default_muscle_quad"]
walk_cfg = cfg["walk"]

lce_min    = np.array(muscle["lce_min"])     # (12,)
lce_max    = np.array(muscle["lce_max"])     # (12,)
phi_min    = np.array(muscle["phi_min"])     # (12,)
phi_max    = np.array(muscle["phi_max"])     # (12,)
peak_force = np.array(muscle["peak_force"])  # (12,)
pFLmax     = np.array(muscle["pFLmax"])      # (12,)
vmax_arr   = np.array(muscle["vmax"])        # (12,)
FVmax_arr  = np.array(muscle["FVmax"])       # (12,)
HEIGHT     = walk_cfg["height_target"]       # base height for MuJoCo state

NUM_JOINTS  = 12
JOINT_NAMES = ["FR_hip", "FR_thigh", "FR_calf",
               "FL_hip", "FL_thigh", "FL_calf",
               "RR_hip", "RR_thigh", "RR_calf",
               "RL_hip", "RL_thigh", "RL_calf"]

# ── Hill model helpers — identical to plot_stiffness_cocontraction.py ──────────
def _afl_bump(lce, A, mid, B):
    left  = 0.5 * (A + mid)
    right = 0.5 * (mid + B)
    if lce <= A or lce >= B:
        return 0.0
    if lce < left:
        t = (lce - A) / (left - A);     return 0.5 * t * t
    elif lce < mid:
        t = (mid - lce) / (mid - left); return 1.0 - 0.5 * t * t
    elif lce < right:
        t = (lce - mid) / (right - mid); return 1.0 - 0.5 * t * t
    else:
        t = (B - lce) / (B - right);    return 0.5 * t * t

def active_fl(lce, j):
    return (_afl_bump(lce, lce_min[j], 1.0, lce_max[j])
            + 0.15 * _afl_bump(lce, lce_min[j], 0.5 * (lce_min[j] + 0.95), 0.95))

def passive_fl(lce, j):
    b = 0.5 * (lce_max[j] + 1.0)
    if lce <= 1.0: return 0.0
    if lce <= b:
        t = (lce - 1.0) / (b - 1.0); return 0.25 * pFLmax[j] * t ** 3
    t = (lce - b) / (b - 1.0); return 0.25 * pFLmax[j] * (1.0 + 3.0 * t)

def force_vel(velocity, j):
    c   = FVmax_arr[j] - 1.0
    eff = velocity / vmax_arr[j]
    if eff < -1.0: return 0.0
    if eff <= 0.0: return (eff + 1.0) ** 2
    if eff <= c:   return FVmax_arr[j] - (c - eff) ** 2 / c
    return FVmax_arr[j]

def moment_arm(j):
    eps = 1e-6
    return (lce_max[j] - lce_min[j] + eps) / (phi_max[j] - phi_min[j] + eps)

def lce_pair(q, j):
    r = moment_arm(j)
    lce1 = q *  r + (lce_min[j] - r * phi_min[j])
    lce2 = q * -r + (lce_min[j] + r * phi_max[j])
    return lce1, lce2

# Blend along the feasible constraint-line segment:
#   0.5 = midpoint (minimum co-contraction, original behaviour)
#   1.0 = high-activation end (maximum co-contraction on the constraint line)
# Higher values → stiffer joint, more robust to perturbations, less efficient.
STIFFNESS = 0.75

def constraint_midpoint(q, dq, tau_req, j):
    """
    Point on the feasible constraint-line segment for joint j at pose (q, dq)
    with required net torque tau_req, blended by STIFFNESS.

    Constraint (with FV): FL1·FV1·a1 − FL2·FV2·a2 = C
        C = tau_req / (−r · F_peak) − (P1 − P2)

    STIFFNESS=0.5 → midpoint (minimum co-contraction)
    STIFFNESS=1.0 → high-activation end (maximum co-contraction)

    Returns (a1, a2) clamped to [0,1]². Returns (0, 0) if FL1·FV1 ≈ 0.
    """
    r    = moment_arm(j)
    lce1, lce2 = lce_pair(q, j)

    FL1 = active_fl(lce1, j)
    FL2 = active_fl(lce2, j)
    P1  = passive_fl(lce1, j)
    P2  = passive_fl(lce2, j)
    FV1 = force_vel( r * dq, j)
    FV2 = force_vel(-r * dq, j)

    eff1 = FL1 * FV1   # effective agonist   scaling
    eff2 = FL2 * FV2   # effective antagonist scaling

    if eff1 < 1e-8:
        return 0.0, 0.0

    C = tau_req / (-r * peak_force[j]) - (P1 - P2)

    denom2 = eff2 if eff2 > 1e-8 else 1e-8
    a2_lo  = max(0.0, -C / denom2)
    a2_hi  = min(1.0, (eff1 - C) / denom2)

    if a2_lo > a2_hi:
        # Torque demand outside muscle capability — clamp to nearest feasible point
        a2_out = np.clip(0.5 * (a2_lo + a2_hi), 0.0, 1.0)
    else:
        a2_out = a2_lo + STIFFNESS * (a2_hi - a2_lo)

    a1_out = np.clip((C + eff2 * a2_out) / eff1, 0.0, 1.0)
    return float(a1_out), float(a2_out)

# ── MuJoCo setup ──────────────────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)

_jid     = [model.actuator_trnid[i, 0] for i in range(NUM_JOINTS)]
_qa_adr  = [model.jnt_qposadr[j]       for j in _jid]
_dof_adr = [model.jnt_dofadr[j]        for j in _jid]

def get_bias_torques(q_joints, dq_joints):
    """
    Set Go2 to (HEIGHT, level base, q_joints, dq_joints) and return qfrc_bias
    for the 12 actuated joints — gravity + Coriolis + centrifugal, no contact.
    Base velocity is zeroed (we only have joint velocities from the gait TSV).
    """
    mujoco.mj_resetData(model, data)
    data.qpos[2] = HEIGHT      # z height
    data.qpos[3] = 1.0         # quaternion w (level base)
    for i in range(NUM_JOINTS):
        data.qpos[_qa_adr[i]]  = q_joints[i]
        data.qvel[_dof_adr[i]] = dq_joints[i]
    mujoco.mj_forward(model, data)
    return np.array([data.qfrc_bias[_dof_adr[i]] for i in range(NUM_JOINTS)])

# ── process all gaits ─────────────────────────────────────────────────────────
# One plot per tier (SLOW/MED/FAST), showing FR leg joints only for clarity.
# Each line = one gait variant; solid=a1, dashed=a2.
VEL_COLORS  = {"0_0": "#2196F3", "0_1": "#4CAF50", "0_2": "#FF9800", "0_5": "#E91E63"}
HEIGHT_STYLES = {"10cm": "-", "15cm": "--", "20cm": ":"}
FR_JOINTS   = [0, 1, 2]   # FR_hip, FR_thigh, FR_calf

tier_figs = {}
for tier in TIERS:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Activation gait reference — {tier} cadence\n"
                 f"(FR leg only; solid=a1 agonist, dashed=a2 antagonist; "
                 f"colour=velocity, style=step height)", fontsize=10)
    tier_figs[tier] = (fig, axes)

infeasible_summary = []

for gait_key, (gait_path, tier, vel, height) in GAIT_FILES.items():

    gait    = np.loadtxt(gait_path, delimiter='\t')
    N       = gait.shape[1]
    q_traj  = gait[:12, :]
    dq_traj = gait[12:24, :]

    a1_out       = np.zeros((NUM_JOINTS, N))
    a2_out       = np.zeros((NUM_JOINTS, N))
    n_infeasible = np.zeros(NUM_JOINTS, dtype=int)

    for t in range(N):
        tau = get_bias_torques(q_traj[:, t], dq_traj[:, t])
        for j in range(NUM_JOINTS):
            a1_out[j, t], a2_out[j, t] = constraint_midpoint(
                q_traj[j, t], dq_traj[j, t], tau[j], j)
            r              = moment_arm(j)
            lce1, lce2     = lce_pair(q_traj[j, t], j)
            eff1           = active_fl(lce1, j) * force_vel( r * dq_traj[j, t], j)
            eff2           = active_fl(lce2, j) * force_vel(-r * dq_traj[j, t], j)
            P1, P2         = passive_fl(lce1, j), passive_fl(lce2, j)
            C              = tau[j] / (-r * peak_force[j]) - (P1 - P2)
            denom2         = eff2 if eff2 > 1e-8 else 1e-8
            if max(0.0, -C / denom2) > min(1.0, (eff1 - C) / denom2):
                n_infeasible[j] += 1

    # Save TSV — mirror source folder structure under OUT_DIR
    tier_out_dir = os.path.join(OUT_DIR, tier)
    os.makedirs(tier_out_dir, exist_ok=True)
    out_array = np.vstack([a1_out, a2_out])   # (24, N)
    out_fname = f"activation_gait_{gait_key}.tsv"
    out_path  = os.path.join(tier_out_dir, out_fname)
    np.savetxt(out_path, out_array, delimiter='\t', fmt='%.8f')

    total_inf = n_infeasible.sum()
    inf_str   = f"  ⚠ {total_inf} infeasible phases (clamped)" if total_inf else ""
    print(f"  {gait_key:<22}  ({N:>3} steps)  "
          f"a1=[{a1_out.min():.2f},{a1_out.max():.2f}]  "
          f"a2=[{a2_out.min():.2f},{a2_out.max():.2f}]{inf_str}")

    if total_inf:
        infeasible_summary.append((gait_key, n_infeasible, N))

    # Plot FR leg joints for this tier
    fig, axes = tier_figs[tier]
    phases = np.arange(N)
    color  = VEL_COLORS[vel]
    ls     = HEIGHT_STYLES[height]
    lbl    = f"v={vel.replace('_','.')} {height}"
    for ji, j in enumerate(FR_JOINTS):
        axes[ji].plot(phases, a1_out[j], color=color, linestyle=ls,
                      linewidth=1.2, label=lbl)
        axes[ji].plot(phases, a2_out[j], color=color, linestyle=ls,
                      linewidth=1.2, alpha=0.4)

# Finalise per-tier plots
for tier, (fig, axes) in tier_figs.items():
    for ji, j in enumerate(FR_JOINTS):
        axes[ji].set_title(JOINT_NAMES[j], fontsize=9)
        axes[ji].set_ylim(-0.05, 1.05)
        axes[ji].set_xlabel("Gait phase (step)", fontsize=8)
        axes[ji].set_ylabel("Activation", fontsize=8)
        axes[ji].grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[2].legend(by_label.values(), by_label.keys(), fontsize=7, loc="upper right")
    fig.tight_layout()
    out_plot = os.path.join(_DIR, f"activation_gaits_{tier.lower()}.png")
    fig.savefig(out_plot, dpi=150)
    print(f"\nSaved plot → {out_plot}")

# Infeasibility report
if infeasible_summary:
    print(f"\n{'='*60}")
    print("Infeasible phase report (constraint outside [0,1]² — values clamped):")
    for gait_key, n_inf, N in infeasible_summary:
        joints_inf = [f"{JOINT_NAMES[j]}:{n_inf[j]}/{N}" for j in range(NUM_JOINTS) if n_inf[j]]
        print(f"  {gait_key}: {', '.join(joints_inf)}")

print(f"\nAll activation gait files written to: {OUT_DIR}")
