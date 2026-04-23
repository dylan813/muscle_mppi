// generate_reference.cpp
//
// Runs ref_free_mppi in MuJoCo simulation, collects per-step torques via
// get_torques(), inverts the Hill model, and writes a reference activation
// CSV for muscle_mppi's act_reference cost term.
//
// Replaces the dial-mpc → extract_reference.py pipeline entirely.
//
// Usage (run from build/):
//   ./generate_reference [task] [n_steps] [output.csv]
//   e.g.: ./generate_reference walk_forward 300 walk_fwd_ref.csv
//
// *** Hill model parameters below MUST match muscle_mppi/utils/tasks.yaml ***

#include "control/ref_free_mppi.h"
#include "utils/tasks.h"

#include <mujoco/mujoco.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <stdexcept>

// ============================================================
// Hill model parameters — keep in sync with tasks.yaml [muscle]
// ============================================================
static constexpr double ALPHA      = 0.15;
static constexpr int    N_SUBSTEPS = 10;   // must match muscle_mppi substeps

static const double TAU_MAX[NUM_JOINTS] = {
    23.7, 23.7, 35.55,  23.7, 23.7, 35.55,
    23.7, 23.7, 35.55,  23.7, 23.7, 35.55,
    23.7, 23.7, 23.7,   23.7
};
static const double DQ_MAX[NUM_JOINTS] = {
    30.1, 30.1, 20.07,  30.1, 30.1, 20.07,
    30.1, 30.1, 20.07,  30.1, 30.1, 20.07,
    30.1, 30.1, 30.1,   30.1
};
static const double B_DAMP[NUM_JOINTS] = {
    0.0, 0.5, 0.5,   0.0, 0.5, 0.5,
    0.0, 0.5, 0.5,   0.0, 0.5, 0.5,
    0.2, 0.2, 0.2,   0.2
};
static const double KD_SIM[NUM_JOINTS] = {
    2.0, 3.5, 3.5,   2.0, 3.5, 3.5,
    2.0, 3.5, 3.5,   2.0, 3.5, 3.5,
    2.0, 2.0, 2.0,   2.0
};
static const double K_PLUS[NUM_JOINTS] = {
    5.0, 5.0, 5.0,   5.0, 5.0, 5.0,
    5.0, 5.0, 5.0,   5.0, 5.0, 5.0,
    0.0, 0.0, 0.0,   0.0
};
static const double K_MINUS[NUM_JOINTS] = {
    5.0, 5.0, 5.0,   5.0, 5.0, 5.0,
    5.0, 5.0, 5.0,   5.0, 5.0, 5.0,
    0.0, 0.0, 0.0,   0.0
};
static const double ALPHA_PLUS[NUM_JOINTS] = {
    10.0, 10.0, 10.0,  10.0, 10.0, 10.0,
    10.0, 10.0, 10.0,  10.0, 10.0, 10.0,
     0.0,  0.0,  0.0,   0.0
};
static const double ALPHA_MINUS[NUM_JOINTS] = {
    10.0, 10.0, 10.0,  10.0, 10.0, 10.0,
    10.0, 10.0, 10.0,  10.0, 10.0, 10.0,
     0.0,  0.0,  0.0,   0.0
};
// Passive spring onset angles (0.1 rad inside hard XML limits)
static const double Q_PLUS0[NUM_JOINTS] = {
     0.9472,  3.3907, -0.9378,   0.9472,  3.3907, -0.9378,
     0.9472,  4.4379, -0.9378,   0.9472,  4.4379, -0.9378,
     1.0e9,   1.0e9,   1.0e9,    1.0e9
};
static const double Q_MINUS0[NUM_JOINTS] = {
    -0.9472, -1.4708, -2.6227,  -0.9472, -1.4708, -2.6227,
    -0.9472, -0.4236, -2.6227,  -0.9472, -0.4236, -2.6227,
    -1.0e9,  -1.0e9,  -1.0e9,   -1.0e9
};

// ============================================================
// Hill model inverse — mirrors extract_reference.py::inverse_hill()
//
// tau_ref = act * tau_max * vel_factor + tau_passive + tau_damp - kd_sim*dq
//
// Inverse:
//   tau_active = tau_ref + kd_sim*dq - tau_passive - tau_damp
//   act        = tau_active / (tau_max * vel_factor)
//   vel_factor = min(1 - sign(act)*dq/dq_max, 2)
// One refinement pass resolves the sign dependency.
// ============================================================
static void inverse_hill(const double tau_ref[NUM_JOINTS],
                          const double q[NUM_JOINTS],
                          const double dq[NUM_JOINTS],
                          double act_out[NUM_JOINTS])
{
    for (int j = 0; j < NUM_JOINTS; ++j) {
        // Passive elastic torque
        double tau_p = 0.0;
        const double dp = q[j] - Q_PLUS0[j];
        if (dp > 0.0) tau_p -= K_PLUS[j]  * (std::exp(ALPHA_PLUS[j]  * dp) - 1.0);
        const double dm = Q_MINUS0[j] - q[j];
        if (dm > 0.0) tau_p += K_MINUS[j] * (std::exp(ALPHA_MINUS[j] * dm) - 1.0);

        const double tau_damp   = -B_DAMP[j] * dq[j];
        const double tau_active = tau_ref[j] + KD_SIM[j] * dq[j] - tau_p - tau_damp;

        // First estimate
        double sg = (tau_active >= 0.0) ? 1.0 : -1.0;
        double vf = std::min(1.0 - sg * dq[j] / DQ_MAX[j], 2.0);
        if (std::abs(vf) < 0.05) vf = std::copysign(0.05, vf);
        double act = tau_active / (TAU_MAX[j] * vf);

        // One refinement
        sg  = (act >= 0.0) ? 1.0 : -1.0;
        vf  = std::min(1.0 - sg * dq[j] / DQ_MAX[j], 2.0);
        if (std::abs(vf) < 0.05) vf = std::copysign(0.05, vf);
        act_out[j] = std::clamp(tau_active / (TAU_MAX[j] * vf), -1.0, 1.0);
    }
}

// ============================================================
// Forward-simulate activation dynamics (matches muscle_mppi substep loop)
// a[sub] = tanh(act_cmd)*alpha + a[sub-1]*(1-alpha), applied N_SUBSTEPS times
// ============================================================
static void smooth_activations(
    const std::vector<std::array<double, NUM_JOINTS>>& raw,
    std::vector<std::array<double, NUM_JOINTS>>&       smooth)
{
    const int T = static_cast<int>(raw.size());
    smooth.resize(T);
    smooth[0] = raw[0];
    for (int t = 1; t < T; ++t) {
        auto a = smooth[t - 1];
        for (int sub = 0; sub < N_SUBSTEPS; ++sub)
            for (int j = 0; j < NUM_JOINTS; ++j)
                a[j] = std::tanh(raw[t][j]) * ALPHA + a[j] * (1.0 - ALPHA);
        smooth[t] = a;
    }
}

// ============================================================
int main(int argc, const char** argv)
// ============================================================
{
    const std::string task    = (argc >= 2) ? argv[1] : "stand";
    const int         n_steps = (argc >= 3) ? std::stoi(argv[2]) : 250;
    const std::string out_csv = (argc >= 4) ? argv[3] : "ref_activations.csv";

    std::cout << "generate_reference\n"
              << "  task    = " << task    << "\n"
              << "  n_steps = " << n_steps << "\n"
              << "  output  = " << out_csv << "\n\n";

    // ---- Load model for forward simulation ----
    const TaskConfig cfg = get_task(task);
    char err[1000];
    mjModel* m = mj_loadXML(cfg.model_path, nullptr, err, sizeof(err));
    if (!m) throw std::runtime_error(std::string("mj_loadXML: ") + err);
    m->opt.timestep = cfg.dt;
    mjData* d = mj_makeData(m);

    // Build LowState-order → qpos/qvel address mapping
    int act_qpos_adr[NUM_JOINTS], act_qvel_adr[NUM_JOINTS];
    for (int j = 0; j < NUM_JOINTS; ++j) {
        const int jid    = m->actuator_trnid[2 * j];
        act_qpos_adr[j]  = m->jnt_qposadr[jid];
        act_qvel_adr[j]  = m->jnt_dofadr[jid];
    }

    // ---- Initialise at standing pose ----
    mj_resetData(m, d);
    d->qpos[2] = 0.45;   // nominal trunk height
    d->qpos[3] = 1.0;    // upright quaternion (w=1)
    for (int j = 0; j < NUM_JOINTS; ++j)
        d->qpos[act_qpos_adr[j]] = cfg.nominal_pose[j];
    mj_forward(m, d);

    // ---- Create MPPI planner ----
    RefFreeMPPI mppi(task);
    mppi.set_height_target(d->qpos[2]);

    // ---- Simulation loop ----
    std::vector<std::array<double, NUM_JOINTS>> raw_acts(n_steps);

    for (int t = 0; t < n_steps; ++t) {
        // Build RobotState from mjData
        RobotState state;
        state.pos[0]  = d->qpos[0]; state.pos[1] = d->qpos[1]; state.pos[2] = d->qpos[2];
        state.quat[0] = d->qpos[3]; state.quat[1] = d->qpos[4];
        state.quat[2] = d->qpos[5]; state.quat[3] = d->qpos[6];
        state.vel[0]  = d->qvel[0]; state.vel[1] = d->qvel[1]; state.vel[2] = d->qvel[2];
        state.gyro[0] = d->qvel[3]; state.gyro[1] = d->qvel[4]; state.gyro[2] = d->qvel[5];
        for (int j = 0; j < NUM_JOINTS; ++j) {
            state.q[j]  = d->qpos[act_qpos_adr[j]];
            state.dq[j] = d->qvel[act_qvel_adr[j]];
        }
        state.valid = true;

        // MPPI solve → joint targets
        double q_out[NUM_JOINTS], dq_out[NUM_JOINTS];
        mppi.update(state, q_out, dq_out);
        mppi.set_predict_delay(0.02);  // assume one control step of latency

        // Get reference torques (leg PD + wheel rolling constraint)
        double tau_ref[NUM_JOINTS];
        mppi.get_torques(state, tau_ref);

        // Invert Hill model → raw activation command
        inverse_hill(tau_ref, state.q, state.dq, raw_acts[t].data());

        // Advance simulation: leg PD + wheel rolling constraint
        for (int sub = 0; sub < cfg.substeps; ++sub) {
            for (int j = 0; j < NUM_LEG_JOINTS; ++j) {
                d->ctrl[j] = cfg.kp * (q_out[j]  - d->qpos[act_qpos_adr[j]])
                           + cfg.kd * (dq_out[j] - d->qvel[act_qvel_adr[j]]);
            }
            const double omega_cmd = d->qvel[0] / cfg.wheel_radius;
            for (int j = NUM_LEG_JOINTS; j < NUM_JOINTS; ++j)
                d->ctrl[j] = cfg.kp_wheel * (omega_cmd - d->qvel[act_qvel_adr[j]]);
            mj_step(m, d);
        }

        // Bail out if simulation diverged
        if (!std::isfinite(d->qpos[2]) || d->qpos[2] < 0.1) {
            std::cerr << "Simulation diverged at step " << t << "\n";
            raw_acts.resize(t);
            break;
        }

        if (t % 50 == 0)
            std::cout << "  step " << t << "/" << n_steps
                      << "  z=" << d->qpos[2]
                      << "  vx=" << d->qvel[0] << "\n";
    }

    mj_deleteData(d);
    mj_deleteModel(m);

    // ---- Forward-simulate activation dynamics ----
    // Matches muscle_mppi: alpha applied N_SUBSTEPS=10 times per control step.
    std::vector<std::array<double, NUM_JOINTS>> act_smooth;
    smooth_activations(raw_acts, act_smooth);

    const int T = static_cast<int>(act_smooth.size());

    // ---- Write CSV ----
    std::ofstream f(out_csv);
    if (!f) throw std::runtime_error("Cannot open: " + out_csv);

    // Header row
    for (int j = 0; j < NUM_JOINTS; ++j) {
        f << "j" << j;
        if (j < NUM_JOINTS - 1) f << ",";
    }
    f << "\n";

    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < NUM_JOINTS; ++j) {
            f << act_smooth[t][j];
            if (j < NUM_JOINTS - 1) f << ",";
        }
        f << "\n";
    }

    // Print summary
    double act_min = 1.0, act_max = -1.0;
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < NUM_JOINTS; ++j) {
            act_min = std::min(act_min, act_smooth[t][j]);
            act_max = std::max(act_max, act_smooth[t][j]);
        }

    std::cout << "\nSaved " << T << " steps × " << NUM_JOINTS
              << " joints to " << out_csv << "\n"
              << "  Activation range: [" << act_min << ", " << act_max << "]\n"
              << "  Duration: " << T * cfg.dt_ctrl << " s\n";

    return 0;
}
