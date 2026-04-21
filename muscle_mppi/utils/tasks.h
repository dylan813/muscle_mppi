#pragma once

#include <string>

static constexpr int NUM_JOINTS     = 16;
static constexpr int NUM_LEG_JOINTS = 12;
static constexpr int NUM_WHEELS     = 4;

// -----------------------------------------------------------------------
// Joint index mapping: LowState/actuator order → MuJoCo qpos/qvel DOF offset
//
// LowState (= d->ctrl) order:  FR(hip,thigh,calf), FL(...), RR(...), RL(...),
//                               FR_wheel, FL_wheel, RR_wheel, RL_wheel
// MuJoCo tree order (qpos[7+], qvel[6+]):
//                               FL(hip,thigh,calf,wheel), FR(...),
//                               RL(...), RR(...)
//
// Usage:  d->qpos[7 + LS_TO_QPOS[j]] = state.q[j]
//         q_cur[j]  = d->qpos[7 + LS_TO_QPOS[j]]
//         dq_cur[j] = d->qvel[6 + LS_TO_QPOS[j]]
//         d->ctrl[j] = tau_out[j]  (ctrl order == LowState order, no mapping needed)
// -----------------------------------------------------------------------
static constexpr int LS_TO_QPOS[NUM_JOINTS] = {
     4,  5,  6,   // FR: hip→4,  thigh→5,  calf→6
     0,  1,  2,   // FL: hip→0,  thigh→1,  calf→2
    12, 13, 14,   // RR: hip→12, thigh→13, calf→14
     8,  9, 10,   // RL: hip→8,  thigh→9,  calf→10
     7,           // FR_wheel → 7
     3,           // FL_wheel → 3
    15,           // RR_wheel → 15
    11            // RL_wheel → 11
};

// Hill model parameters — one set per task (could be tuned per robot variant)
struct MuscleParams {
    // Per-joint torque limits (Nm) — from UNITREE_GO2W_SATA_CFG (go2w_torque.urdf)
    double tau_max[NUM_JOINTS] = {
        23.7,  23.7,  35.55,       // FR hip, thigh, calf
        23.7,  23.7,  35.55,       // FL
        23.7,  23.7,  35.55,       // RR
        23.7,  23.7,  35.55,       // RL
        23.7,  23.7,  23.7,  23.7  // wheels
    };

    // Per-joint velocity limits (rad/s) — from UNITREE_GO2W_SATA_CFG
    double dq_max[NUM_JOINTS] = {
        30.1, 30.1, 20.07,         // FR hip, thigh, calf
        30.1, 30.1, 20.07,         // FL
        30.1, 30.1, 20.07,         // RR
        30.1, 30.1, 20.07,         // RL
        30.1, 30.1, 30.1,  30.1   // wheels
    };

    // Activation dynamics blend factor per physics substep (dt = 0.002 s).
    // Derived from SATA alpha=0.6 at dt~0.01 s, scaled to dt=0.002 s.
    // a[t] = tanh(act_cmd) * alpha + a[t-1] * (1 - alpha)
    double activation_alpha = 0.15;

    // -----------------------------------------------------------------------
    // Passive elastic torque — exponential springs near joint limits.
    // Only non-zero for leg joints; wheels have no position limits.
    //
    // tau_passive_j =
    //   + k_plus[j]  * (exp(alpha_plus[j]  * max(q-q_plus0,  0)) - 1)  [push back from upper limit]
    //   - k_minus[j] * (exp(alpha_minus[j] * max(q0_minus-q, 0)) - 1)  [push back from lower limit]
    //
    // q_plus0/q_minus0 are 0.1 rad inside the hard XML limits, so resistance
    // ramps up before the constraint solver fires.
    // -----------------------------------------------------------------------

    // Angles (rad) at which passive resistance begins (0.1 rad inside hard limits).
    // All joints of the same class share the same range in their local frame —
    // left/right symmetry is handled by MuJoCo body orientation, NOT by flipping limits.
    //
    // Hard limits from go2w.xml:
    //   abduction (all):   [-1.0472,  1.0472]
    //   front thigh:       [-1.5708,  3.4907]
    //   back  thigh:       [-0.5236,  4.5379]
    //   calf  (knee):      [-2.7227, -0.83776]
    double q_plus0[NUM_JOINTS] = {
         0.9472,  3.3907, -0.9378,   // FR  hip/thigh/calf
         0.9472,  3.3907, -0.9378,   // FL  (same range as FR in local frame)
         0.9472,  4.4379, -0.9378,   // RR  back thigh upper = 4.5379
         0.9472,  4.4379, -0.9378,   // RL
         1e9, 1e9, 1e9, 1e9          // wheels: no upper limit
    };
    double q_minus0[NUM_JOINTS] = {
        -0.9472, -1.4708, -2.6227,   // FR
        -0.9472, -1.4708, -2.6227,   // FL
        -0.9472, -0.4236, -2.6227,   // RR  back thigh lower = -0.5236
        -0.9472, -0.4236, -2.6227,   // RL
        -1e9, -1e9, -1e9, -1e9       // wheels: no lower limit
    };

    // Passive stiffness magnitudes (Nm). Tuned so that at the hard limit
    // (0.1 rad past q_plus0/q_minus0) tau_passive ≈ 36 Nm ≈ peak gravity torque.
    double k_plus[NUM_JOINTS] = {
        5.0, 5.0, 5.0,
        5.0, 5.0, 5.0,
        5.0, 5.0, 5.0,
        5.0, 5.0, 5.0,
        0.0, 0.0, 0.0, 0.0   // wheels
    };
    double k_minus[NUM_JOINTS] = {
        5.0, 5.0, 5.0,
        5.0, 5.0, 5.0,
        5.0, 5.0, 5.0,
        5.0, 5.0, 5.0,
        0.0, 0.0, 0.0, 0.0   // wheels
    };

    // Exponential growth rates (1/rad). alpha=10 gives e^1 ≈ 2.7x at 0.1 rad
    // past threshold, so k*(e-1) ≈ 5*(1.718) ≈ 8.6 Nm at threshold+0.1 rad.
    double alpha_plus[NUM_JOINTS] = {
        10.0, 10.0, 10.0,
        10.0, 10.0, 10.0,
        10.0, 10.0, 10.0,
        10.0, 10.0, 10.0,
        0.0, 0.0, 0.0, 0.0
    };
    double alpha_minus[NUM_JOINTS] = {
        10.0, 10.0, 10.0,
        10.0, 10.0, 10.0,
        10.0, 10.0, 10.0,
        10.0, 10.0, 10.0,
        0.0, 0.0, 0.0, 0.0
    };

    // Viscous damping (Nm·s/rad).
    // Abduction joints already have damping=0.1 in the XML — set to 0 here.
    // Thigh/calf have no XML damping — add 0.5 Nm·s/rad.
    // Wheels: light damping for rolling stability.
    double b_damp[NUM_JOINTS] = {
        0.0, 0.5, 0.5,   // FR  hip=0 (XML has 0.1), thigh/calf=0.5
        0.0, 0.5, 0.5,   // FL
        0.0, 0.5, 0.5,   // RR
        0.0, 0.5, 0.5,   // RL
        0.2, 0.2, 0.2, 0.2  // wheels
    };

    // Hardware kd mirrored in rollout simulation so MPPI optimises for real-robot dynamics.
    // Must match kd_[] in mppi_controller.cpp exactly.
    double kd_sim[NUM_JOINTS] = {
        2.0, 3.5, 3.5,   // FR  hip / thigh / calf
        2.0, 3.5, 3.5,   // FL
        2.0, 3.5, 3.5,   // RR
        2.0, 3.5, 3.5,   // RL
        2.0, 2.0, 2.0, 2.0  // wheels — match hardware kd to damp residual wheel torques
    };
};

// Cost weights matching paper Eq. (17-18), Table II (walking).
// Running: c_t = w_h|Δz| + w_orient*θ² + w_q*||q-q0||² + w_cv*||v_c||₁ + w_cf*||f_c-f0||₁
// Terminal: c_T = w_H * ||p_H - p_target||₁
struct CostWeights {
    double height        = 350.0;  // w_h:       L1 body height deviation
    double orientation   =  10.0;  // w_orient:  geodesic angle² from upright
    double posture       =   5.0;  // w_q:       joint deviation from nominal standing pose
    double contact_vel   =   0.0;  // w_c,vel:   L1 wheel body velocity (0 = off for wheeled)
    double contact_force =   0.0;  // w_c,force: disabled — height term handles vertical stability
    double terminal      = 500.0;  // w_H:       L1 terminal displacement
    double act_smooth    =   0.1;  // activation rate-of-change penalty
};

// Full task specification
struct TaskConfig {
    const char* model_path = "../../unitree_mujoco/unitree_robots/go2w/scene_terrain.xml";

    double goal_pos[3]   = {0.0, 0.0, 0.0};
    double height_target = 0.464;

    double nominal_pose[NUM_JOINTS] = {};

    CostWeights  cost;
    MuscleParams muscle;

    int    n_samples    = 16;
    int    horizon      = 45;
    int    substeps     = 5;
    int    n_iterations = 3;    // inner planning loops per control step (Sec. III-D)
    double lambda       = 0.1;  // range-normalised: 0→winner-takes-all, 1→uniform weights
    double beta1        = 3.0;  // noise annealing: iteration decay (Eq. 8)
    double beta2        = 3.0;  // noise annealing: horizon decay (Eq. 8)
    double dt           = 0.002;
    // Note: no kp/kd — muscle model replaces PD entirely

    // Noise in activation space (dimensionless, ∈ [-1,1]).
    //
    // With PD control, large noise is safe because PD provides local stability.
    // With Hill model torque control, activation noise directly maps to torque
    // variation: sigma=0.08 on a thigh (tau_max=23.7 Nm) can reduce torque from
    // 8 Nm to 2.6 Nm in one step, causing leg collapse.
    //
    // Sizing rule: sigma * tau_max << gravity_torque_margin (~3–5 Nm for legs).
    //   hip   (tau_max=23.7): sigma=0.01 → ±0.24 Nm noise (small)
    //   thigh (tau_max=23.7): sigma=0.02 → ±0.47 Nm noise
    //   calf  (tau_max=45.4): sigma=0.02 → ±0.91 Nm noise
    //   wheel (tau_max=15.0): sigma=0.05 → ±0.75 Nm (wheels don't support weight)
    double noise_sigma[NUM_JOINTS] = {
        0.05, 0.08, 0.08,   // FR leg
        0.05, 0.08, 0.08,   // FL leg
        0.05, 0.08, 0.08,   // RR leg
        0.05, 0.08, 0.08,   // RL leg
        0.15, 0.15, 0.15, 0.15  // wheels
    };
};

// -----------------------------------------------------------------------
// Motion command — set externally (joystick, nav stack, etc.)
// -----------------------------------------------------------------------
struct MotionCommand {
    double vx     = 0.0;    // desired forward  body velocity (m/s)
    double vy     = 0.0;    // desired lateral  body velocity (m/s)
    double wz     = 0.0;    // desired yaw rate (rad/s)
    double height = 0.464;  // desired body height (m)
};

TaskConfig get_task(const std::string& name);
