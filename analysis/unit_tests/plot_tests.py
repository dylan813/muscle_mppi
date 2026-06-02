"""
Visualizations for the 4 muscle model unit tests.
Loads parameters from tasks.yaml and re-implements muscle.h math in Python.

Run from this directory:
    python3 plot_tests.py
Output: unit_test_results.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

# ---------------------------------------------------------------------------
# Load parameters from tasks.yaml
# ---------------------------------------------------------------------------
yaml_path = os.path.join(os.path.dirname(__file__), "../muscle_mppi/utils/tasks.yaml")
with open(yaml_path) as f:
    cfg = yaml.safe_load(f)

m = cfg["default_muscle"]
task = cfg["reach"]

NUM_JOINTS  = 3
NUM_MUSCLES = 6

act_bandwidth = float(m["act_bandwidth"])
peak_force = np.array(m["peak_force"], dtype=float)
lce_min    = np.array(m["lce_min"],    dtype=float)
lce_max    = np.array(m["lce_max"],    dtype=float)
phi_min    = np.array(m["phi_min"],    dtype=float)
phi_max    = np.array(m["phi_max"],    dtype=float)
vmax_arr   = np.array(m["vmax"],       dtype=float)
FVmax_arr  = np.array(m["FVmax"],      dtype=float)
pFLmax_arr = np.array(m["pFLmax"],     dtype=float)
kd_sim_arr = np.array(m["kd_sim"],     dtype=float)
nominal_pose = np.array(task["nominal_pose"], dtype=float)
DT = float(task["dt"])

JOINT_NAMES  = ["Hip", "Thigh", "Calf"]
COLORS       = ["#2196F3", "#FF9800", "#4CAF50"]
C_AGONIST    = "#1565C0"
C_ANTAGONIST = "#C62828"

# ---------------------------------------------------------------------------
# Hill model helpers — exact reimplementation of muscle.h
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


def fl_total(lce, j):
    lmin, lmax = lce_min[j], lce_max[j]
    return (active_force_length(lce, lmin, 1.0, lmax)
            + 0.15 * active_force_length(lce, lmin, 0.5 * (lmin + 0.95), 0.95))


def force_vel(velocity, c, vmax, FVmax):
    eff = velocity / vmax
    if eff < -1.0: return 0.0
    if eff <= 0.0: return (eff + 1.0) ** 2
    if eff <= c:   return FVmax - (c - eff) ** 2 / c
    return FVmax


def passive_force_length(length, max_val, b):
    if length <= 1.0: return 0.0
    if length <= b:
        temp = (length - 1.0) / (b - 1.0)
        return 0.25 * max_val * temp ** 3
    temp = (length - b) / (b - 1.0)
    return 0.25 * max_val * (1.0 + 3.0 * temp)


def hill_step(act_cmd, q, dq, dt, activation):
    """One step of hill_compute_torques. Modifies activation in-place; returns tau."""
    alpha = act_bandwidth * dt
    tau   = np.zeros(NUM_JOINTS)
    eps   = 1e-6

    for j in range(NUM_JOINTS):
        r1   = (lce_max[j] - lce_min[j] + eps) / (phi_max[j] - phi_min[j] + eps)
        lce1 = q[j] * r1   + (lce_min[j] - r1  * phi_min[j])
        lce2 = q[j] * (-r1) + (lce_min[j] + r1  * phi_max[j])

        for k in range(2):
            ctrl = np.clip(act_cmd[2*j + k], 0.0, 1.0)
            activation[2*j + k] = np.clip(
                activation[2*j + k] + alpha * (ctrl - activation[2*j + k]), 0.0, 1.0)
        act1, act2 = activation[2*j], activation[2*j + 1]

        FL1, FL2 = fl_total(lce1, j), fl_total(lce2, j)
        c = FVmax_arr[j] - 1.0
        FV1 = force_vel( r1 * dq[j], c, vmax_arr[j], FVmax_arr[j])
        FV2 = force_vel(-r1 * dq[j], c, vmax_arr[j], FVmax_arr[j])
        b_p = 0.5 * (lce_max[j] + 1.0)
        F1 = (FL1 * FV1 * act1 + passive_force_length(lce1, pFLmax_arr[j], b_p)) * peak_force[j]
        F2 = (FL2 * FV2 * act2 + passive_force_length(lce2, pFLmax_arr[j], b_p)) * peak_force[j]
        tau[j] = -(F1 * r1 + F2 * (-r1))

    return tau


def warm_up(cmd, q, dq, activation, steps=500):
    for _ in range(steps):
        hill_step(cmd, q, dq, DT, activation)


# ---------------------------------------------------------------------------
# Data generators (one per test)
# ---------------------------------------------------------------------------

def gen_test1():
    """Activation rise to 1 then decay to 0. Returns (time_ms, activation)."""
    act = np.zeros(NUM_MUSCLES)
    q, dq = np.zeros(NUM_JOINTS), np.zeros(NUM_JOINTS)
    times, vals = [], []
    for i in range(50):
        hill_step(np.ones(NUM_MUSCLES), q, dq, DT, act)
        times.append(i * DT * 1000)
        vals.append(act[0])
    for i in range(50):
        hill_step(np.zeros(NUM_MUSCLES), q, dq, DT, act)
        times.append((50 + i) * DT * 1000)
        vals.append(act[0])
    return np.array(times), np.array(vals)


def gen_test2():
    """Steady-state torque with pure agonist vs pure antagonist for each joint."""
    q_mid = 0.5 * (phi_min + phi_max)
    dq    = np.zeros(NUM_JOINTS)
    tau_ag, tau_ant = np.zeros(NUM_JOINTS), np.zeros(NUM_JOINTS)

    for j in range(NUM_JOINTS):
        cmd = np.full(NUM_MUSCLES, 0.5)
        cmd[2*j] = 1.0; cmd[2*j + 1] = 0.0
        act = np.zeros(NUM_MUSCLES)
        warm_up(cmd, q_mid, dq, act)
        tau_ag[j] = hill_step(cmd, q_mid, dq, DT, act)[j]

        cmd = np.full(NUM_MUSCLES, 0.5)
        cmd[2*j] = 0.0; cmd[2*j + 1] = 1.0
        act = np.zeros(NUM_MUSCLES)
        warm_up(cmd, q_mid, dq, act)
        tau_ant[j] = hill_step(cmd, q_mid, dq, DT, act)[j]

    return tau_ag, tau_ant


def gen_test3():
    """Net torque vs antagonist fraction α ∈ [0,1] at mid-range, per joint."""
    q_mid  = 0.5 * (phi_min + phi_max)
    dq     = np.zeros(NUM_JOINTS)
    fracs  = np.linspace(0.0, 1.0, 80)
    torques = np.zeros((NUM_JOINTS, len(fracs)))

    for fi, frac in enumerate(fracs):
        cmd = np.array([v for j in range(NUM_JOINTS) for v in (1.0 - frac, frac)])
        act = np.zeros(NUM_MUSCLES)
        warm_up(cmd, q_mid, dq, act)
        torques[:, fi] = hill_step(cmd, q_mid, dq, DT, act)

    return fracs, torques


def gen_test4():
    """Torque-based PD position hold: start → nominal_pose."""
    q_target = nominal_pose.copy()
    inertia  = np.array([0.05, 0.25, 0.08])
    kp_tau, kd_tau = 30.0, 8.0

    q   = np.array([0.2, 0.3, -1.6])
    dq  = np.zeros(NUM_JOINTS)
    act = np.zeros(NUM_MUSCLES)
    eps = 1e-6

    n_steps = int(4.0 / DT)
    t_log   = np.arange(1, n_steps + 1) * DT
    q_log   = np.zeros((NUM_JOINTS, n_steps))

    for i in range(n_steps):
        cmd = np.zeros(NUM_MUSCLES)
        for j in range(NUM_JOINTS):
            r1   = (lce_max[j] - lce_min[j] + eps) / (phi_max[j] - phi_min[j] + eps)
            lce1 = q[j] * r1   + (lce_min[j] - r1  * phi_min[j])
            lce2 = q[j] * (-r1) + (lce_min[j] + r1  * phi_max[j])
            FL1, FL2 = fl_total(lce1, j), fl_total(lce2, j)
            tau_des  = kp_tau * (q_target[j] - q[j]) - kd_tau * dq[j]
            sum_FL   = FL1 + FL2
            delta    = ((0.5 * (FL1 - FL2) + tau_des / (r1 * peak_force[j])) / sum_FL
                        if sum_FL > 0.01 else 0.0)
            cmd[2*j    ] = np.clip(0.5 - delta, 0.0, 1.0)
            cmd[2*j + 1] = np.clip(0.5 + delta, 0.0, 1.0)

        tau = hill_step(cmd, q, dq, DT, act)
        dq += (tau - kd_sim_arr * dq) / inertia * DT
        q  += dq * DT
        q_log[:, i] = q

    return t_log, q_log, q_target


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def main():
    print("Generating data ...")
    t1_t, t1_a           = gen_test1()
    t2_ag, t2_ant        = gen_test2()
    t3_fracs, t3_tau     = gen_test3()
    t4_t, t4_q, t4_tgt   = gen_test4()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Muscle Model Unit Tests", fontsize=14, fontweight="bold")

    # ── Test 1: Activation dynamics ─────────────────────────────────────────
    ax = axes[0, 0]
    split = 50 * DT * 1000   # ms where command flips
    ax.plot(t1_t, t1_a, color=C_AGONIST, linewidth=2)
    ax.axhline(0.99, color="#388E3C", linestyle="--", linewidth=1, label="99% / 1%")
    ax.axhline(0.01, color="#388E3C", linestyle="--", linewidth=1)
    ax.axvline(split, color="#757575", linestyle=":", linewidth=1.5, alpha=0.7,
               label="cmd flip 1 → 0")
    ax.fill_between(t1_t[:50], t1_a[:50], alpha=0.12, color=C_AGONIST)
    ax.fill_between(t1_t[50:], t1_a[50:], alpha=0.12, color=C_ANTAGONIST)

    # Annotate the 5τ crossing points
    ax.annotate(f"5τ  {t1_a[24]:.4f}", xy=(t1_t[24], t1_a[24]),
                xytext=(35, 0.72), fontsize=8, color=C_AGONIST,
                arrowprops=dict(arrowstyle="->", color=C_AGONIST, lw=1.1))
    ax.annotate(f"5τ  {t1_a[74]:.4f}", xy=(t1_t[74], t1_a[74]),
                xytext=(115, 0.22), fontsize=8, color=C_ANTAGONIST,
                arrowprops=dict(arrowstyle="->", color=C_ANTAGONIST, lw=1.1))

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Activation")
    ax.set_title("Test 1 — Activation Dynamics\n"
                 r"bandwidth=100 Hz,  $\tau$=10 ms,  5$\tau$=50 ms")
    ax.legend(fontsize=8, loc="right")
    ax.set_ylim(-0.05, 1.18)
    ax.grid(True, alpha=0.3)

    # ── Test 2: Torque sign ──────────────────────────────────────────────────
    ax = axes[0, 1]
    x, w = np.arange(NUM_JOINTS), 0.35
    b1 = ax.bar(x - w/2, t2_ag,  w, color=C_AGONIST,    alpha=0.85, label="Agonist only")
    b2 = ax.bar(x + w/2, t2_ant, w, color=C_ANTAGONIST, alpha=0.85, label="Antagonist only")
    ax.axhline(0, color="black", linewidth=0.8)
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + w/2, h - 1.0, f"{h:.1f}",
                ha="center", va="top", fontsize=8.5, color="white", fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + w/2, h + 0.5, f"+{h:.1f}",
                ha="center", va="bottom", fontsize=8.5, color=C_ANTAGONIST, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_NAMES)
    ax.set_ylabel("Net torque (N·m)")
    ax.set_title("Test 2 — Torque Sign Verification\n"
                 "full activation at mid-range, zero velocity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Test 3: Co-contraction sweep ─────────────────────────────────────────
    ax = axes[1, 0]
    for j in range(NUM_JOINTS):
        ax.plot(t3_fracs, t3_tau[j], color=COLORS[j], linewidth=2, label=JOINT_NAMES[j])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0.5, color="#757575", linestyle="--", linewidth=1.5, alpha=0.8, label="α = 0.5")
    ax.text(0.52, 2.5, "net torque = 0\nat α = 0.5", fontsize=8, color="#555555")
    ax.set_xlabel("Antagonist activation fraction (α)")
    ax.set_ylabel("Net torque (N·m)")
    ax.set_title("Test 3 — Co-contraction: Torque vs Activation Split\n"
                 "mid-range position, zero velocity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Test 4: Position hold trajectory ────────────────────────────────────
    ax = axes[1, 1]
    q_start = [0.2, 0.3, -1.6]
    for j in range(NUM_JOINTS):
        ax.plot(t4_t, t4_q[j], color=COLORS[j], linewidth=1.8, label=JOINT_NAMES[j])
        ax.axhline(t4_tgt[j], color=COLORS[j], linestyle="--", linewidth=1.2,
                   alpha=0.55, label=f"target {JOINT_NAMES[j]}")
        ax.plot(0, q_start[j], "o", color=COLORS[j], markersize=6, zorder=5)

    # Final-error annotation for each joint
    for j in range(NUM_JOINTS):
        err = abs(t4_q[j, -1] - t4_tgt[j])
        ax.text(3.6, t4_tgt[j] + 0.04, f"err={err:.4f}", fontsize=7.5,
                color=COLORS[j], ha="right")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint angle (rad)")
    ax.set_title("Test 4 — FL Leg Position Hold (torque-based PD)\n"
                 "start=[0.2, 0.3, −1.6]  →  target=[0.0, 0.67, −1.3] rad")
    # Keep legend compact: only the solid-line entries
    handles = [plt.Line2D([0], [0], color=COLORS[j], linewidth=2, label=JOINT_NAMES[j])
               for j in range(NUM_JOINTS)]
    handles += [plt.Line2D([0], [0], color="gray", linewidth=1, linestyle="--",
                            label="targets")]
    ax.legend(handles=handles, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 4)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "unit_test_results.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
