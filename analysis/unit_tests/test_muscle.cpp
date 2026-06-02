// Unit tests for hill_compute_torques (muscle.h).
//
// No external dependencies — standard library only.
// Build from this directory:
//   g++ -std=c++17 -O2 test_muscle.cpp -o test_muscle && ./test_muscle
//
// Joint order (NUM_JOINTS=3, FL leg only):
//   0: FL_hip   1: FL_thigh   2: FL_calf
//
// Muscle cmd layout (NUM_MUSCLES=6, interleaved per joint):
//   [hip_agonist, hip_antagonist, thigh_agonist, thigh_antagonist, calf_agonist, calf_antagonist]
//
// Sign convention (from muscle.h):
//   tau[j] = -(F1*r1 + F2*r2),  r2 = -r1,  r1 > 0 for all joints
//   => antagonist dominant (act2 > act1)  => tau > 0  (raises joint angle)
//   => agonist   dominant (act1 > act2)  => tau < 0  (lowers joint angle)

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <string>

#include "../muscle_mppi/control/muscle.h"  // pulls in ../utils/tasks.h

// ---------------------------------------------------------------------------
// Params matching tasks.yaml default_muscle block (current values)
// ---------------------------------------------------------------------------
static MuscleParams default_params()
{
    MuscleParams p;
    p.act_bandwidth = 100.0;

    // peak_force = tau_max / r (moment arm)
    p.peak_force[0] = 149.7;   // hip   — 23.7 Nm / 0.1584 m
    p.peak_force[1] = 384.1;   // thigh — 23.7 Nm / 0.0617 m
    p.peak_force[2] = 255.1;   // calf  — 45.4 Nm / 0.1781 m

    p.lce_min[0] = 0.75;   p.lce_min[1] = 0.75;   p.lce_min[2] = 0.75;
    p.lce_max[0] = 1.05;   p.lce_max[1] = 1.05;   p.lce_max[2] = 1.05;

    // phi range from go2.xml, 0.1 rad inside hard stops
    p.phi_min[0] = -0.9472;   p.phi_min[1] = -1.4708;   p.phi_min[2] = -2.6227;
    p.phi_max[0] =  0.9472;   p.phi_max[1] =  3.3907;   p.phi_max[2] = -0.9378;

    // vmax = r * dq_max (from Go2 URDF)
    p.vmax[0] = 4.77;   // hip   — 0.1584 * 30.1 rad/s
    p.vmax[1] = 1.86;   // thigh — 0.0617 * 30.1 rad/s
    p.vmax[2] = 2.80;   // calf  — 0.1781 * 15.7 rad/s

    p.FVmax[0] = 1.33;   p.FVmax[1] = 1.33;   p.FVmax[2] = 1.33;
    p.pFLmax[0] = 1.3;   p.pFLmax[1] = 1.3;   p.pFLmax[2] = 1.3;
    p.kd_sim[0] = 2.0;   p.kd_sim[1] = 3.5;   p.kd_sim[2] = 3.5;
    return p;
}

static bool check(bool cond, const char* msg)
{
    printf("%s  %s\n", cond ? "PASS" : "FAIL", msg);
    return cond;
}

static void warm_up(const double cmd[NUM_MUSCLES],
                    const double q[NUM_JOINTS],
                    const double dq[NUM_JOINTS],
                    const MuscleParams& p, double dt,
                    double act[NUM_MUSCLES],
                    int steps = 500)
{
    double tau[NUM_JOINTS];
    for (int i = 0; i < steps; ++i)
        hill_compute_torques(cmd, q, dq, p, dt, act, tau);
}

// ===========================================================================
// Test 1 — Activation dynamics: first-order filter converges to target
// ===========================================================================
static bool test_activation_dynamics()
{
    printf("[1] Activation dynamics\n");

    const MuscleParams p = default_params();
    const double dt = 0.002;
    // alpha = bandwidth * dt = 100 * 0.002 = 0.2 per step
    // 5 time-constants = 5 * (1/100 Hz) = 50 ms = 25 steps

    double q[NUM_JOINTS]  = {};
    double dq[NUM_JOINTS] = {};
    double tau[NUM_JOINTS];
    double act[NUM_MUSCLES] = {};

    double cmd_on[NUM_MUSCLES];
    std::fill(cmd_on, cmd_on + NUM_MUSCLES, 1.0);
    for (int i = 0; i < 25; ++i)
        hill_compute_torques(cmd_on, q, dq, p, dt, act, tau);

    bool ok = true;
    for (int m = 0; m < NUM_MUSCLES; ++m) {
        printf("  act[%d] after rise  = %.4f\n", m, act[m]);
        ok &= (act[m] > 0.99);
    }

    double cmd_off[NUM_MUSCLES] = {};
    for (int i = 0; i < 25; ++i)
        hill_compute_torques(cmd_off, q, dq, p, dt, act, tau);

    for (int m = 0; m < NUM_MUSCLES; ++m) {
        printf("  act[%d] after decay = %.4f\n", m, act[m]);
        ok &= (act[m] < 0.01);
    }

    return check(ok, "activation filter: reaches 1 and decays to 0 within 5 time-constants");
}

// ===========================================================================
// Test 2 — Torque sign: pure agonist vs pure antagonist produce opposite signs
// ===========================================================================
static bool test_torque_sign()
{
    printf("[2] Torque sign (agonist vs antagonist)\n");

    const MuscleParams p = default_params();
    const double dt = 0.002;

    // Geometric midpoint of each joint's range (zero velocity)
    const double q_mid[NUM_JOINTS] = {
        0.5 * (p.phi_min[0] + p.phi_max[0]),
        0.5 * (p.phi_min[1] + p.phi_max[1]),
        0.5 * (p.phi_min[2] + p.phi_max[2]),
    };
    const double dq_zero[NUM_JOINTS] = {};

    bool ok = true;
    for (int j = 0; j < NUM_JOINTS; ++j) {
        // Pure agonist for joint j; 0.5 co-contraction on the other two
        double cmd_ag[NUM_MUSCLES], cmd_ant[NUM_MUSCLES];
        for (int m = 0; m < NUM_MUSCLES; ++m) {
            const bool this_joint = (m / 2 == j);
            cmd_ag[m]  = this_joint ? (m % 2 == 0 ? 1.0 : 0.0) : 0.5;
            cmd_ant[m] = this_joint ? (m % 2 == 0 ? 0.0 : 1.0) : 0.5;
        }

        double act_ag[NUM_MUSCLES]  = {};
        double act_ant[NUM_MUSCLES] = {};
        double tau_ag[NUM_JOINTS],  tau_ant[NUM_JOINTS];

        warm_up(cmd_ag,  q_mid, dq_zero, p, dt, act_ag);
        warm_up(cmd_ant, q_mid, dq_zero, p, dt, act_ant);
        hill_compute_torques(cmd_ag,  q_mid, dq_zero, p, dt, act_ag,  tau_ag);
        hill_compute_torques(cmd_ant, q_mid, dq_zero, p, dt, act_ant, tau_ant);

        printf("  joint %d (q_mid=%.3f):  tau_agonist=%+.2f Nm  tau_antagonist=%+.2f Nm\n",
               j, q_mid[j], tau_ag[j], tau_ant[j]);

        ok &= (tau_ag[j]  * tau_ant[j] < 0.0);
        ok &= (std::abs(tau_ag[j])  > 1.0);
        ok &= (std::abs(tau_ant[j]) > 1.0);
    }
    return check(ok, "agonist and antagonist produce opposite-sign nonzero torques");
}

// ===========================================================================
// Test 3 — Co-contraction: equal 0.5/0.5 activation gives near-zero net torque
// ===========================================================================
static bool test_co_contraction()
{
    printf("[3] Co-contraction cancellation\n");

    const MuscleParams p = default_params();
    const double dt = 0.002;

    const double q_mid[NUM_JOINTS] = {
        0.5 * (p.phi_min[0] + p.phi_max[0]),
        0.5 * (p.phi_min[1] + p.phi_max[1]),
        0.5 * (p.phi_min[2] + p.phi_max[2]),
    };
    const double dq_zero[NUM_JOINTS] = {};

    double act[NUM_MUSCLES] = {};
    double cmd[NUM_MUSCLES];
    std::fill(cmd, cmd + NUM_MUSCLES, 0.5);
    double tau[NUM_JOINTS];

    warm_up(cmd, q_mid, dq_zero, p, dt, act);
    hill_compute_torques(cmd, q_mid, dq_zero, p, dt, act, tau);

    bool ok = true;
    for (int j = 0; j < NUM_JOINTS; ++j) {
        printf("  joint %d: net torque = %+.4f Nm\n", j, tau[j]);
        ok &= (std::abs(tau[j]) < 5.0);  // small relative to ~24–45 Nm peak
    }
    return check(ok, "equal co-contraction at mid-range gives |tau| < 5 Nm");
}

// ===========================================================================
// Test 4 — FL leg position hold via torque-based PD + 1-DOF integrator
//
// Uses a torque-space PD controller with FL feedforward to account for the
// force-length asymmetry at arbitrary joint positions:
//
//   tau_des  = kp_tau * (q_target - q) - kd_tau * dq
//
//   At zero velocity, tau ≈ -r1 * peak_force * (FL1*act1 - FL2*act2)
//   With act1 = 0.5 - delta, act2 = 0.5 + delta:
//   tau ≈ -r1 * peak_force * (0.5*(FL1-FL2) - delta*(FL1+FL2))
//
//   Solving for delta:
//   delta = [0.5*(FL1-FL2) + tau_des/(r1*peak_force)] / (FL1+FL2)
//
//   This correctly offsets the equilibrium activation to counteract FL
//   asymmetry — e.g., if antagonist has higher FL at the target, the
//   controller assigns it less activation so forces balance to zero.
//
// Integrates: dq += (tau - kd_sim*dq) / inertia * dt;  q += dq * dt
//
// Target: nominal_pose from tasks.yaml = [0.0, 0.67, -1.3] rad
// Start:  perturbed                    = [0.2, 0.3, -1.6]  rad
// Pass:   all joints within 0.05 rad of target after 4 simulated seconds
// ===========================================================================
static bool test_position_hold()
{
    printf("[4] FL leg position hold (torque-based PD, 1-DOF integrator)\n");

    const MuscleParams p = default_params();
    const double dt = 0.002;

    const double q_target[NUM_JOINTS] = {0.0, 0.67, -1.3};   // nominal_pose
    const double inertia[NUM_JOINTS]  = {0.05, 0.25, 0.08};   // approx effective inertia (kg·m²)

    const double kp_tau = 30.0;   // N·m / rad
    const double kd_tau =  8.0;   // N·m·s / rad

    double q[NUM_JOINTS]  = {0.2, 0.3, -1.6};
    double dq[NUM_JOINTS] = {};
    double act[NUM_MUSCLES] = {};

    const int steps = static_cast<int>(4.0 / dt);

    printf("  start  q = [%+.3f, %+.3f, %+.3f] rad\n", q[0], q[1], q[2]);
    printf("  target q = [%+.3f, %+.3f, %+.3f] rad\n",
           q_target[0], q_target[1], q_target[2]);

    for (int i = 0; i < steps; ++i) {
        double cmd[NUM_MUSCLES];

        for (int j = 0; j < NUM_JOINTS; ++j) {
            // Moment arm (same formula as muscle.h)
            static constexpr double eps = 1e-6;
            const double r1 = (p.lce_max[j] - p.lce_min[j] + eps)
                            / (p.phi_max[j]  - p.phi_min[j]  + eps);

            // Fiber lengths at current q
            const double lce1 = q[j] * r1  + (p.lce_min[j] - r1  * p.phi_min[j]);
            const double lce2 = q[j] * (-r1) + (p.lce_min[j] + r1 * p.phi_max[j]);

            // Force-length (same formula as muscle.h, including secondary shoulder)
            const double FL1 = active_force_length(lce1, p.lce_min[j], 1.0, p.lce_max[j])
                             + 0.15 * active_force_length(lce1, p.lce_min[j],
                                                          0.5*(p.lce_min[j] + 0.95), 0.95);
            const double FL2 = active_force_length(lce2, p.lce_min[j], 1.0, p.lce_max[j])
                             + 0.15 * active_force_length(lce2, p.lce_min[j],
                                                          0.5*(p.lce_min[j] + 0.95), 0.95);

            // Desired torque from PD law
            const double tau_des = kp_tau * (q_target[j] - q[j]) - kd_tau * dq[j];

            // Activation delta: solves tau_des = -r1*peak*(0.5*(FL1-FL2) - delta*(FL1+FL2))
            const double sum_FL = FL1 + FL2;
            const double delta  = (sum_FL > 0.01)
                                ? (0.5*(FL1 - FL2) + tau_des / (r1 * p.peak_force[j])) / sum_FL
                                : 0.0;

            cmd[2*j    ] = std::max(0.0, std::min(1.0, 0.5 - delta));   // agonist
            cmd[2*j + 1] = std::max(0.0, std::min(1.0, 0.5 + delta));   // antagonist
        }

        double tau[NUM_JOINTS];
        hill_compute_torques(cmd, q, dq, p, dt, act, tau);

        for (int j = 0; j < NUM_JOINTS; ++j) {
            dq[j] += (tau[j] - p.kd_sim[j] * dq[j]) / inertia[j] * dt;
            q[j]  += dq[j] * dt;
        }
    }

    printf("  final  q = [%+.3f, %+.3f, %+.3f] rad\n", q[0], q[1], q[2]);

    bool ok = true;
    for (int j = 0; j < NUM_JOINTS; ++j) {
        const double err = std::abs(q[j] - q_target[j]);
        printf("  joint %d error = %.4f rad  (%s)\n", j, err, err < 0.05 ? "ok" : "FAIL");
        ok &= (err < 0.05);
    }
    return check(ok, "FL leg converges to nominal_pose within 0.05 rad in 4 s");
}

// ===========================================================================
int main()
{
    printf("=== muscle model unit tests ===\n\n");

    int passed = 0;
    passed += test_activation_dynamics(); printf("\n");
    passed += test_torque_sign();         printf("\n");
    passed += test_co_contraction();      printf("\n");
    passed += test_position_hold();       printf("\n");

    const int total = 4;
    printf("=== %d / %d passed ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}
