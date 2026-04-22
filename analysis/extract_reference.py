"""
extract_reference.py

Loads a dial-mpc _states.npy rollout, remaps joint order from dial-mpc
body-tree order to muscle_mppi LowState order, then inverts the Hill model
to produce reference activation signals.

Outputs (saved next to the input file):
  *_ref_torques.npy      — (T, 16) torques in LowState order
  *_ref_activations.npy  — (T, 16) inverted activations in LowState order
  *_ref_activations.csv  — same, for easy loading in C++

Usage:
  python3 extract_reference.py <path/to/states.npy>
"""

import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# Joint ordering
# ---------------------------------------------------------------------------
# dial-mpc body-tree DOF order (same for qpos[7:] and ctrl[]):
#   0:FL_hip  1:FL_thigh  2:FL_calf  3:FL_wheel
#   4:FR_hip  5:FR_thigh  6:FR_calf  7:FR_wheel
#   8:RL_hip  9:RL_thigh 10:RL_calf 11:RL_wheel
#  12:RR_hip 13:RR_thigh 14:RR_calf 15:RR_wheel
#
# LowState / muscle_mppi order:
#   0:FR_hip  1:FR_thigh  2:FR_calf
#   3:FL_hip  4:FL_thigh  5:FL_calf
#   6:RR_hip  7:RR_thigh  8:RR_calf
#   9:RL_hip 10:RL_thigh 11:RL_calf
#  12:FR_wheel 13:FL_wheel 14:RR_wheel 15:RL_wheel

# Mapping: DIALMPC_TO_LS[i] = dial-mpc index that goes into LowState slot i
DIALMPC_TO_LS = np.array([
     4,  5,  6,   # LS 0-2  : FR leg
     0,  1,  2,   # LS 3-5  : FL leg
    12, 13, 14,   # LS 6-8  : RR leg
     8,  9, 10,   # LS 9-11 : RL leg
     7,           # LS 12   : FR_wheel
     3,           # LS 13   : FL_wheel
    15,           # LS 14   : RR_wheel
    11,           # LS 15   : RL_wheel
], dtype=int)

# Inverse: LS_TO_DIALMPC[i] = LowState index that goes into dial-mpc slot i
LS_TO_DIALMPC = np.argsort(DIALMPC_TO_LS)

NUM_JOINTS     = 16
NUM_LEG_JOINTS = 12

# ---------------------------------------------------------------------------
# Muscle parameters (must match tasks.h / MuscleParams defaults exactly)
# ---------------------------------------------------------------------------
tau_max = np.array([
    23.7, 23.7, 35.55,
    23.7, 23.7, 35.55,
    23.7, 23.7, 35.55,
    23.7, 23.7, 35.55,
    23.7, 23.7, 23.7, 23.7,
])

dq_max = np.array([
    30.1, 30.1, 20.07,
    30.1, 30.1, 20.07,
    30.1, 30.1, 20.07,
    30.1, 30.1, 20.07,
    30.1, 30.1, 30.1, 30.1,
])

b_damp = np.array([
    0.0, 0.5, 0.5,
    0.0, 0.5, 0.5,
    0.0, 0.5, 0.5,
    0.0, 0.5, 0.5,
    0.2, 0.2, 0.2, 0.2,
])

# Passive spring onset angles (0.1 rad inside hard limits)
q_plus0 = np.array([
     0.9472,  3.3907, -0.9378,
     0.9472,  3.3907, -0.9378,
     0.9472,  4.4379, -0.9378,
     0.9472,  4.4379, -0.9378,
     1e9,     1e9,    1e9,    1e9,
])
q_minus0 = np.array([
    -0.9472, -1.4708, -2.6227,
    -0.9472, -1.4708, -2.6227,
    -0.9472, -0.4236, -2.6227,
    -0.9472, -0.4236, -2.6227,
    -1e9,   -1e9,   -1e9,   -1e9,
])

k_plus  = np.array([5.0]*12 + [0.0]*4)
k_minus = np.array([5.0]*12 + [0.0]*4)
alpha_plus  = np.array([10.0]*12 + [0.0]*4)
alpha_minus = np.array([10.0]*12 + [0.0]*4)

# kd_sim mirrors mppi_controller.cpp kd_ (subtracted in simulation)
kd_sim = np.array([
    2.0, 3.5, 3.5,
    2.0, 3.5, 3.5,
    2.0, 3.5, 3.5,
    2.0, 3.5, 3.5,
    2.0, 2.0, 2.0, 2.0,
])


# ---------------------------------------------------------------------------
# Hill model helpers (vectorised over joints, one timestep at a time)
# ---------------------------------------------------------------------------

def passive_torque(q):
    """Exponential unilateral spring torques."""
    tau_p = np.zeros(NUM_JOINTS)
    d_plus  = q - q_plus0
    d_minus = q_minus0 - q
    mask_p = d_plus  > 0.0
    mask_m = d_minus > 0.0
    tau_p -= k_plus  * (np.exp(alpha_plus  * d_plus)  - 1.0) * mask_p
    tau_p += k_minus * (np.exp(alpha_minus * d_minus) - 1.0) * mask_m
    return tau_p


def inverse_hill(tau_ref_ls, q_ls, dq_ls):
    """
    Given reference torque, joint position, joint velocity (all in LowState order),
    return the activation ∈ [-1, 1] that would produce tau_ref through the Hill model.

    tau_ref = act * tau_max * vel_factor + tau_passive + tau_damp - kd_sim * dq
    (the kd_sim term is the hardware damping subtracted before sending to actuator)

    Inverse:
      tau_active_needed = tau_ref + kd_sim * dq - tau_passive - tau_damp_b
      act = tau_active_needed / (tau_max * vel_factor)

    vel_factor = min(1 - sign(act) * dq / dq_max, 2)
    We resolve the sign dependency with one iteration.
    """
    tau_p    = passive_torque(q_ls)
    tau_damp = -b_damp * dq_ls            # viscous damping from MuscleParams

    # Net torque that the active (Hill) component must provide.
    # Add back kd_sim*dq because that is subtracted in the controller loop
    # (d->ctrl[j] = tau_hill[j] - kd_sim[j]*dq[j])
    tau_active_needed = tau_ref_ls + kd_sim * dq_ls - tau_p - tau_damp

    # First estimate of vel_factor assuming activation sign == tau_active sign
    sign_est = np.sign(tau_active_needed)
    sign_est[sign_est == 0] = 1.0
    vel_factor = np.minimum(1.0 - sign_est * dq_ls / dq_max, 2.0)

    # Avoid near-zero vel_factor (singular point)
    vf_safe = np.where(np.abs(vel_factor) > 0.05, vel_factor, 0.05)
    act_est  = tau_active_needed / (tau_max * vf_safe)

    # One refinement pass with updated sign
    sign_ref   = np.sign(act_est)
    sign_ref[sign_ref == 0] = 1.0
    vel_factor = np.minimum(1.0 - sign_ref * dq_ls / dq_max, 2.0)
    vf_safe    = np.where(np.abs(vel_factor) > 0.05, vel_factor, 0.05)
    act_out    = tau_active_needed / (tau_max * vf_safe)

    return np.clip(act_out, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(states_path: str):
    data = np.load(states_path)          # (T, 62)
    T = data.shape[0]

    # Layout: [step(1) | qpos(23) | qvel(22) | ctrl(16)]
    # qpos[7:] = joint positions in dial-mpc body-tree order
    # ctrl      = torques          in dial-mpc body-tree order
    q_dialmpc   = data[:, 1 + 7  : 1 + 7  + 16]   # (T, 16) joint pos
    dq_dialmpc  = data[:, 1 + 23 + 6 : 1 + 23 + 6 + 16]   # (T, 16) joint vel
    tau_dialmpc = data[:, 1 + 23 + 22 : ]           # (T, 16) ctrl torques

    # Remap to LowState order
    q_ls   = q_dialmpc[:,   DIALMPC_TO_LS]
    dq_ls  = dq_dialmpc[:,  DIALMPC_TO_LS]
    tau_ls = tau_dialmpc[:, DIALMPC_TO_LS]

    # Invert Hill model → reference activations
    ref_act = np.zeros((T, NUM_JOINTS))
    for t in range(T):
        ref_act[t] = inverse_hill(tau_ls[t], q_ls[t], dq_ls[t])

    # Forward-simulate the Hill model activation dynamics to match muscle_mppi exactly.
    #
    # muscle_mppi applies:  a[sub] = tanh(act_cmd) * alpha + a[sub-1] * (1-alpha)
    # once per physics substep (dt=0.002 s), with n_substeps=10 per 0.02 s control step.
    # Effective blending per control step = 1 - (1-alpha)^n_substeps ≈ 80%.
    #
    # extract_reference.py has one row per 0.02 s control step, so we must apply
    # the alpha update n_substeps times per row — NOT once — or the smoothed
    # reference will have a ~5x slower time constant than the real controller.
    alpha      = 0.15
    n_substeps = 10   # must match TaskConfig::substeps (dt_ctrl/dt = 0.02/0.002)

    act_smooth = np.zeros_like(ref_act)
    act_smooth[0] = ref_act[0]
    for t in range(1, T):
        a = act_smooth[t - 1]
        for _ in range(n_substeps):
            a = np.tanh(ref_act[t]) * alpha + a * (1.0 - alpha)
        act_smooth[t] = a

    # Save
    stem = states_path.replace("_states.npy", "")
    np.save(stem + "_ref_torques.npy",     tau_ls)
    np.save(stem + "_ref_activations.npy", act_smooth)
    np.savetxt(stem + "_ref_activations.csv", act_smooth, delimiter=",",
               header=",".join([f"j{i}" for i in range(NUM_JOINTS)]),
               comments="")

    print(f"Saved {T} timesteps of reference data.")
    print(f"  Torques:     {stem}_ref_torques.npy     range [{tau_ls.min():.2f}, {tau_ls.max():.2f}] Nm")
    print(f"  Activations: {stem}_ref_activations.npy range [{act_smooth.min():.3f}, {act_smooth.max():.3f}]")
    print(f"  CSV:         {stem}_ref_activations.csv")

    _plot(tau_ls, act_smooth, stem)
    return act_smooth, tau_ls


def _plot(tau_ls, act_ls, stem):
    try:
        import matplotlib.pyplot as plt

        T = tau_ls.shape[0]
        t = np.arange(T) * 0.02  # each step = 20 ms

        joint_labels = [
            "FR_hip", "FR_thigh", "FR_calf",
            "FL_hip", "FL_thigh", "FL_calf",
            "RR_hip", "RR_thigh", "RR_calf",
            "RL_hip", "RL_thigh", "RL_calf",
            "FR_whl", "FL_whl", "RR_whl", "RL_whl",
        ]

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        for j in range(NUM_LEG_JOINTS):
            axes[0].plot(t, tau_ls[:, j], label=joint_labels[j], alpha=0.7)
        axes[0].set_ylabel("Torque (Nm)")
        axes[0].set_title("Reference torques (LowState order, leg joints)")
        axes[0].legend(ncol=4, fontsize=7)
        axes[0].grid(True, alpha=0.3)

        for j in range(NUM_LEG_JOINTS):
            axes[1].plot(t, act_ls[:, j], label=joint_labels[j], alpha=0.7)
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Activation")
        axes[1].set_title("Reference activations (smoothed, LowState order, leg joints)")
        axes[1].legend(ncol=4, fontsize=7)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out = stem + "_reference_plot.pdf"
        plt.savefig(out)
        print(f"  Plot:        {out}")
        plt.close()
    except ImportError:
        print("  (matplotlib not available — skipping plot)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 extract_reference.py <path/to/states.npy>")
        sys.exit(1)
    main(sys.argv[1])
