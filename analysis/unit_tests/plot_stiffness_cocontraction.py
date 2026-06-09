"""
Stiffness vs. co-contraction at the nominal standing pose.

For each joint type (hip, thigh, calf), at the standing nominal_pose:
  1. Extracts gravitational torques via MuJoCo (qfrc_bias at qvel=0 = pure gravity).
  2. Solves the torque-balance constraint line in (a1, a2) activation space.
  3. Sweeps along the constraint line and computes muscle stiffness K = -dtau/dq.

Outputs:
  constraint_lines.png        — constraint line per joint coloured by K
  stiffness_cocontraction.png — K (N·m/rad) vs. total activation (a1+a2) per joint

Run from anywhere; paths are relative to this file.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
import os
import mujoco

# ── paths ──────────────────────────────────────────────────────────────────────
_DIR       = os.path.dirname(os.path.abspath(__file__))
YAML_PATH  = os.path.join(_DIR, "../../muscle_mppi/utils/tasks.yaml")
MODEL_PATH = os.path.join(_DIR, "../../unitree_mujoco/unitree_robots/go2/scene.xml")

# ── load parameters ────────────────────────────────────────────────────────────
with open(YAML_PATH) as f:
    cfg = yaml.safe_load(f)

muscle       = cfg["default_muscle_quad"]
stand        = cfg["stand"]
nominal_pose = stand["nominal_pose"]   # 12 values: FR(hip,thigh,calf) FL RR RL

lce_min    = muscle["lce_min"][:3]
lce_max    = muscle["lce_max"][:3]
phi_min    = muscle["phi_min"][:3]
phi_max    = muscle["phi_max"][:3]
peak_force = muscle["peak_force"][:3]
pFLmax     = muscle["pFLmax"][:3]
vmax       = muscle["vmax"][:3]
FVmax_arr  = muscle["FVmax"][:3]

JOINT_NAMES = ["Hip", "Thigh", "Calf"]
COLORS      = ["#2196F3", "#4CAF50", "#FF9800"]

# ── Hill model helpers (matches muscle.h exactly) ──────────────────────────────
def _afl_bump(lce, A, mid, B):
    left  = 0.5 * (A + mid)
    right = 0.5 * (mid + B)
    if lce <= A or lce >= B:
        return 0.0
    if lce < left:
        t = (lce - A) / (left - A);    return 0.5 * t * t
    elif lce < mid:
        t = (mid - lce) / (mid - left); return 1.0 - 0.5 * t * t
    elif lce < right:
        t = (lce - mid) / (right - mid); return 1.0 - 0.5 * t * t
    else:
        t = (B - lce) / (B - right);   return 0.5 * t * t

def active_fl(lce, lmin, lmax):
    return (_afl_bump(lce, lmin, 1.0, lmax)
            + 0.15 * _afl_bump(lce, lmin, 0.5 * (lmin + 0.95), 0.95))

def passive_fl(lce, fpmax, b):
    if lce <= 1.0: return 0.0
    if lce <= b:
        t = (lce - 1.0) / (b - 1.0); return 0.25 * fpmax * t ** 3
    t = (lce - b) / (b - 1.0); return 0.25 * fpmax * (1.0 + 3.0 * t)

def force_vel(velocity, c, vm, FVm):
    """Force-velocity curve — matches force_vel() in muscle.h exactly."""
    eff = velocity / vm
    if eff < -1.0: return 0.0
    if eff <= 0.0: return (eff + 1.0) ** 2
    if eff <= c:   return FVm - (c - eff) ** 2 / c
    return FVm

# ── geometry helpers ──────────────────────────────────────────────────────────
def _r1(j):
    eps = 1e-6
    return (lce_max[j] - lce_min[j] + eps) / (phi_max[j] - phi_min[j] + eps)

def lce_pair(q, j):
    r = _r1(j)
    lce1 = q *  r + (lce_min[j] - r  * phi_min[j])
    lce2 = q * -r + (lce_min[j] - (-r) * phi_max[j])
    return lce1, lce2

def net_torque(q, dq, a1, a2, j):
    """Net joint torque — matches hill_compute_torques in muscle.h (FV included)."""
    lce1, lce2 = lce_pair(q, j)
    r  = _r1(j)
    b  = 0.5 * (lce_max[j] + 1.0)
    FL1 = active_fl(lce1, lce_min[j], lce_max[j])
    FL2 = active_fl(lce2, lce_min[j], lce_max[j])
    P1  = passive_fl(lce1, pFLmax[j], b)
    P2  = passive_fl(lce2, pFLmax[j], b)
    c   = FVmax_arr[j] - 1.0
    FV1 = force_vel( r * dq, c, vmax[j], FVmax_arr[j])
    FV2 = force_vel(-r * dq, c, vmax[j], FVmax_arr[j])
    F1  = (FL1 * FV1 * a1 + P1) * peak_force[j]
    F2  = (FL2 * FV2 * a2 + P2) * peak_force[j]
    return -(F1 * r + F2 * -r)

def joint_stiffness(q, a1, a2, j, eps=1e-4):
    """Numerical muscle stiffness K = -dtau/dq at dq=0 (N·m/rad)."""
    return -(net_torque(q + eps, 0.0, a1, a2, j) - net_torque(q - eps, 0.0, a1, a2, j)) / (2.0 * eps)

# ── required joint torques — simulate stand_go2.cpp PD stand-up ──────────────
# A single mj_forward at nominal_pose gives wrong contact forces (no dynamic
# contact buildup). Instead, replicate the stand_go2.cpp stand-up procedure so
# contact forces develop naturally, then read equilibrium torques at steady state.
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)

# ctrl[] order is FR→FL→RR→RL (tasks.yaml / actuator order), but the model's
# joint body-tree order is FL→FR→RL→RR.  Use model.actuator_trnid to get the
# correct qpos/qvel address for each actuator rather than assuming 7+i.
_jid     = [model.actuator_trnid[i, 0] for i in range(12)]
_qa_adr  = [model.jnt_qposadr[j] for j in _jid]   # qpos index per actuator
_dof_adr = [model.jnt_dofadr[j]  for j in _jid]   # qvel/qfrc index per actuator

_yaml_nominal  = list(stand["nominal_pose"])        # save YAML values for comparison
stand_up_pos   = list(_yaml_nominal)
stand_down_pos = [ 0.0473455,  1.22187, -2.44375,
                  -0.0473455,  1.22187, -2.44375,
                   0.0473455,  1.22187, -2.44375,
                  -0.0473455,  1.22187, -2.44375]

# Place robot at stand_down_pos, feet flush with ground plane.
data.qpos[:] = 0.0
data.qpos[2] = 0.5
data.qpos[3] = 1.0
for i in range(12):
    data.qpos[_qa_adr[i]] = stand_down_pos[i]
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

_foot_bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
              for n in ('FR_foot', 'FL_foot', 'RR_foot', 'RL_foot')]
_foot_bids = [b for b in _foot_bids if b >= 0]
if _foot_bids:
    data.qpos[2] -= min(data.xpos[b, 2] for b in _foot_bids)
    mujoco.mj_forward(model, data)

# Phase 1 (0-3 s): tanh stand-up, kp 20→50, kd=3.5  — identical to stand_go2.cpp.
# Phase 2 (3-5 s): hold at stand_up_pos with kp=50 to settle to equilibrium.
dt_sim  = 0.002
kd_pd   = 3.5
n_steps = int(5.0 / dt_sim)
t       = 0.0

for _ in range(n_steps):
    t += dt_sim
    if t < 3.0:
        phase = np.tanh(t / 1.2)
        kp    = phase * 50.0 + (1.0 - phase) * 20.0
    else:
        phase = 1.0
        kp    = 50.0
    for i in range(12):
        q_des = phase * stand_up_pos[i] + (1.0 - phase) * stand_down_pos[i]
        data.ctrl[i] = kp * (q_des - data.qpos[_qa_adr[i]]) + kd_pd * (-data.qvel[_dof_adr[i]])
    mujoco.mj_step(model, data)

print(f"Steady-state body height: {data.qpos[2]:.4f} m  |  "
      f"Max |dq|: {np.max(np.abs(data.qvel[6:18])):.5f} rad/s  |  "
      f"Contacts: {data.ncon}")

# Equilibrium torques in tasks.yaml / ctrl order (FR→FL→RR→RL).
tau_grav_all = np.array([data.qfrc_bias[_dof_adr[i]] - data.qfrc_constraint[_dof_adr[i]]
                         for i in range(12)])

# Actual standing pose in tasks.yaml / ctrl order.
nominal_pose = [data.qpos[_qa_adr[i]] for i in range(12)]

_labels = ["FR_hip","FR_thigh","FR_calf",
           "FL_hip","FL_thigh","FL_calf",
           "RR_hip","RR_thigh","RR_calf",
           "RL_hip","RL_thigh","RL_calf"]
print("Steady-state joint torques (N·m)  |  sim pose vs tasks.yaml nominal_pose (Δ rad):")
for lbl, tg, qs, qn in zip(_labels, tau_grav_all, nominal_pose, _yaml_nominal):
    print(f"  {lbl:12s}: tau={tg:+.4f}  sim={qs:+.5f}  yaml={qn:+.5f}  Δ={qs-qn:+.5f}")

tau_grav = tau_grav_all[:3]   # FR leg is representative

# ── sweep & compute ───────────────────────────────────────────────────────────
N = 500
a2_sweep = np.linspace(0.0, 1.0, N)

fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
fig2, ax2   = plt.subplots(figsize=(9, 6))

joints_data = {}

for j, (name, color) in enumerate(zip(JOINT_NAMES, COLORS)):
    q_nom = nominal_pose[j]
    lce1_0, lce2_0 = lce_pair(q_nom, j)
    r = _r1(j)
    b = 0.5 * (lce_max[j] + 1.0)

    FL1_0 = active_fl(lce1_0, lce_min[j], lce_max[j])
    FL2_0 = active_fl(lce2_0, lce_min[j], lce_max[j])
    P1_0  = passive_fl(lce1_0, pFLmax[j], b)
    P2_0  = passive_fl(lce2_0, pFLmax[j], b)

    if abs(FL1_0) < 1e-8:
        print(f"\n{name}: FL1=0 at nominal pose — cannot invert constraint, skipping.")
        continue

    C_j     = tau_grav[j] / (-r * peak_force[j]) - (P1_0 - P2_0)
    joints_data[j] = {'C': C_j, 'FL1': FL1_0, 'FL2': FL2_0, 'r': r}
    a1_line = (C_j + FL2_0 * a2_sweep) / FL1_0

    valid    = (a1_line >= 0.0) & (a1_line <= 1.0)
    a2_valid = a2_sweep[valid]
    a1_valid = a1_line[valid]

    K_valid   = np.array([joint_stiffness(q_nom, a1v, a2v, j)
                          for a1v, a2v in zip(a1_valid, a2_valid)])
    total_act = a1_valid + a2_valid

    # ── constraint line subplot (coloured by K) ───────────────────────────────
    ax = axes1[j]
    sc = ax.scatter(a2_valid, a1_valid, c=K_valid, cmap="viridis", s=6, zorder=2)
    fig1.colorbar(sc, ax=ax, label="K (N·m/rad)")
    ax.plot([0, 1], [0, 1], color="gray", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("a2 (antagonist)", fontsize=11)
    ax.set_ylabel("a1 (agonist)", fontsize=11)
    ax.set_title(f"{name}\nlce1={lce1_0:.3f}  lce2={lce2_0:.3f}\n"
                 f"FL1={FL1_0:.3f}  FL2={FL2_0:.3f}  slope={FL2_0/FL1_0:.3f}",
                 fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # ── muscle K curve ────────────────────────────────────────────────────────
    if len(K_valid) > 0:
        ax2.plot(total_act, K_valid, color=color, linewidth=2.5, label=f"{name}")

    print(f"\n{name}  (q_nom={q_nom:.4f} rad):")
    print(f"  lce1={lce1_0:.4f}  lce2={lce2_0:.4f}")
    print(f"  FL1={FL1_0:.4f}  FL2={FL2_0:.4f}  constraint slope={FL2_0/FL1_0:.4f}")
    print(f"  C_j={C_j:.6f}")
    if len(a2_valid):
        print(f"  Valid a2 range: [{a2_valid.min():.3f}, {a2_valid.max():.3f}]")
        print(f"  K range:        [{K_valid.min():.2f}, {K_valid.max():.2f}] N·m/rad")
    else:
        print("  No valid activation range found (constraint outside [0,1]²)")

# ── midpoint anchors — solved per leg to handle asymmetric gravity torques ────
# Thigh/calf gravity torques are identical across all legs, but hip torques
# are opposite for left (FL/RL) vs right (FR/RR) legs, requiring separate solutions.
LEG_NAMES = ["FR", "FL", "RR", "RL"]
new_ga    = [0.0] * 24
new_gc    = [0.0] * 12   # gravity_C[j]: FL1*a1_anchor - FL2*a2_anchor per joint

print("\n── Constraint-line anchors (per leg) ────────────────────────────────────")
print(f"  {'Leg':3s}  {'Joint':6s}  {'a1':>7s}  {'a2':>7s}  {'K_anchor':>10s}  {'tau_grav':>9s}")

for leg in range(4):
    for jt in range(3):
        q_nom_lj    = nominal_pose[leg * 3 + jt]
        tau_grav_lj = tau_grav_all[leg * 3 + jt]

        lce1_lj, lce2_lj = lce_pair(q_nom_lj, jt)
        r  = _r1(jt)
        b  = 0.5 * (lce_max[jt] + 1.0)
        FL1_lj = active_fl(lce1_lj, lce_min[jt], lce_max[jt])
        FL2_lj = active_fl(lce2_lj, lce_min[jt], lce_max[jt])
        P1_lj  = passive_fl(lce1_lj, pFLmax[jt], b)
        P2_lj  = passive_fl(lce2_lj, pFLmax[jt], b)

        if abs(FL1_lj) < 1e-8:
            print(f"  {LEG_NAMES[leg]:3s}  {JOINT_NAMES[jt]:6s}  FL1=0 at nominal pose, skipping")
            continue

        C_lj  = tau_grav_lj / (-r * peak_force[jt]) - (P1_lj - P2_lj)
        a2_lo = max(0.0, -C_lj / FL2_lj)
        a2_hi = min(1.0, (FL1_lj - C_lj) / FL2_lj)
        if a2_lo > a2_hi:
            a2_lo, a2_hi = 0.0, 0.0

        a2_anchor = (a2_lo + a2_hi) / 2.0
        a1_anchor = np.clip((C_lj + FL2_lj * a2_anchor) / FL1_lj, 0.0, 1.0)
        K_anchor  = joint_stiffness(q_nom_lj, a1_anchor, a2_anchor, jt)

        C_lj_val = FL1_lj * a1_anchor - FL2_lj * a2_anchor

        idx             = leg * 6 + jt * 2
        new_ga[idx]     = round(float(a1_anchor), 4)
        new_ga[idx + 1] = round(float(a2_anchor), 4)
        new_gc[leg * 3 + jt] = round(float(C_lj_val), 6)

        print(f"  {LEG_NAMES[leg]:3s}  {JOINT_NAMES[jt]:6s}  "
              f"{a1_anchor:7.4f}  {a2_anchor:7.4f}  {K_anchor:10.2f}  {tau_grav_lj:+9.4f}  C={C_lj_val:+.6f}")

print("\n  ── gravity_act (paste into tasks.yaml stand section) ──────────────")
for leg in range(4):
    b    = leg * 6
    vals = (f"{new_ga[b]}, {new_ga[b+1]},  "
            f"{new_ga[b+2]}, {new_ga[b+3]},  "
            f"{new_ga[b+4]}, {new_ga[b+5]}")
    if leg == 0:
        print(f"  gravity_act: [{vals},")
    elif leg < 3:
        print(f"               {vals},")
    else:
        print(f"               {vals}]")

print("\n  ── posture_bias (paste into tasks.yaml stand section) ─────────────")
print(f"  # Normalized gravity torque: tau_grav/(-r*peak_force) - (P1-P2), one per joint")
lines = []
for leg in range(4):
    lines.append(f"{new_gc[leg*3]}, {new_gc[leg*3+1]}, {new_gc[leg*3+2]}")
print(f"  posture_bias: [{lines[0]},  {lines[1]},")
print(f"                 {lines[2]},  {lines[3]}]")

# ── per-leg FL1/FL2 at nominal pose ───────────────────────────────────────────
new_fl1 = [0.0] * 12
new_fl2 = [0.0] * 12

for leg in range(4):
    for jt in range(3):
        q_nom_lj   = nominal_pose[leg * 3 + jt]
        lce1_lj, lce2_lj = lce_pair(q_nom_lj, jt)
        new_fl1[leg * 3 + jt] = round(active_fl(lce1_lj, lce_min[jt], lce_max[jt]), 6)
        new_fl2[leg * 3 + jt] = round(active_fl(lce2_lj, lce_min[jt], lce_max[jt]), 6)

print("\n  ── posture_FL1 / posture_FL2 (paste into tasks.yaml stand section) ──")
print("  # Active force-length values at the nominal standing pose (fixed slope).")
fl1_lines = [f"{new_fl1[l*3]}, {new_fl1[l*3+1]}, {new_fl1[l*3+2]}" for l in range(4)]
fl2_lines = [f"{new_fl2[l*3]}, {new_fl2[l*3+1]}, {new_fl2[l*3+2]}" for l in range(4)]
print(f"  posture_FL1: [{fl1_lines[0]},  {fl1_lines[1]},")
print(f"                {fl1_lines[2]},  {fl1_lines[3]}]")
print(f"  posture_FL2: [{fl2_lines[0]},  {fl2_lines[1]},")
print(f"                {fl2_lines[2]},  {fl2_lines[3]}]")


# ── save figures ──────────────────────────────────────────────────────────────
fig1.suptitle("Torque-balance constraint lines at nominal pose\n"
              "(colour = muscle stiffness K)", fontsize=12)
fig1.tight_layout()
out1 = os.path.join(_DIR, "constraint_lines.png")
fig1.savefig(out1, dpi=150)
print(f"\nSaved → {out1}")

ax2.set_xlabel("Total activation  a1 + a2", fontsize=12)
ax2.set_ylabel("Stiffness  K  (N·m/rad)", fontsize=12)
ax2.set_title("Muscle stiffness vs. co-contraction at nominal standing pose", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
out2 = os.path.join(_DIR, "stiffness_cocontraction.png")
fig2.savefig(out2, dpi=150)
print(f"Saved → {out2}")
