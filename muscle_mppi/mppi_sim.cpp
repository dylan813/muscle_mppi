// Standalone MuJoCo simulation test for MPPILocomotion.
// No DDS, no real-time constraint — MPPI runs as fast as possible against a
// local mjData simulation. Useful for verifying the controller works before
// worrying about latency.
//
// Run from muscle_mppi/muscle_mppi/:
//   ./build/mppi_sim [task] [yaml] [output.csv]
//
// Output CSV columns:
//   t, px, py, pz, vx, vy, vz, qw, height_cost, orient_cost, vel_cost, gait_cost

#include <mujoco/mujoco.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "control/mppi_locomotion.h"

// ── stand-up parameters (mirror mppi_controller.cpp) ─────────────────────────
static const double STAND_DOWN[NUM_JOINTS] = {
     0.0473455,  1.22187, -2.44375,
    -0.0473455,  1.22187, -2.44375,
     0.0473455,  1.22187, -2.44375,
    -0.0473455,  1.22187, -2.44375,
};
static const double STAND_UP[NUM_JOINTS] = {
    0.0, 0.67, -1.3,   0.0, 0.67, -1.3,
    0.0, 0.67, -1.3,   0.0, 0.67, -1.3,
};
static constexpr double STANDUP_SECS = 3.0;
static constexpr double HOLD_SECS    = 1.0;   // hold pose before handing to MPPI
static constexpr int    CONVERGENCE_SOLVES = 10;

// ── helpers ───────────────────────────────────────────────────────────────────
static RobotState read_state(const mjModel* m, const mjData* d,
                             const int qa[NUM_JOINTS], const int qv[NUM_JOINTS])
{
    RobotState s;
    s.pos[0]  = d->qpos[0]; s.pos[1]  = d->qpos[1]; s.pos[2]  = d->qpos[2];
    s.quat[0] = d->qpos[3]; s.quat[1] = d->qpos[4];
    s.quat[2] = d->qpos[5]; s.quat[3] = d->qpos[6];
    s.vel[0]  = d->qvel[0]; s.vel[1]  = d->qvel[1]; s.vel[2]  = d->qvel[2];
    s.gyro[0] = d->qvel[3]; s.gyro[1] = d->qvel[4]; s.gyro[2] = d->qvel[5];
    for (int j = 0; j < NUM_JOINTS; ++j) {
        s.q[j]  = d->qpos[qa[j]];
        s.dq[j] = d->qvel[qv[j]];
    }
    s.valid = true;
    return s;
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    const std::string task_name = (argc >= 2) ? argv[1] : "walk";
    const std::string yaml_path = (argc >= 3) ? argv[2] : "../utils/tasks.yaml";
    const std::string csv_path  = (argc >= 4) ? argv[3] : "../mppi_sim/mppi_sim.csv";

    printf("Task: %s  |  YAML: %s  |  CSV: %s\n",
           task_name.c_str(), yaml_path.c_str(), csv_path.c_str());

    // ── load MPPI (also loads the model internally) ──────────────────────────
    MPPILocomotion mppi(task_name, yaml_path);

    // ── load a separate sim model/data ───────────────────────────────────────
    const TaskConfig& task = mppi.task_ref();   // public accessor we'll add
    char err[1000];
    mjModel* m = mj_loadXML(task.model_path.c_str(), nullptr, err, sizeof(err));
    if (!m) { fprintf(stderr, "mj_loadXML: %s\n", err); return 1; }
    m->opt.timestep = task.dt;
    mjData* d = mj_makeData(m);

    // set joint damping to match MPPI's internal model
    int qa[NUM_JOINTS], qv[NUM_JOINTS];
    for (int j = 0; j < NUM_JOINTS; ++j) {
        int jid = m->actuator_trnid[2 * (JOINT_OFFSET + j)];
        qa[j]   = m->jnt_qposadr[jid];
        qv[j]   = m->jnt_dofadr[jid];
        m->dof_damping[qv[j]] = task.muscle.kd_sim[j];
    }

    // place robot feet on ground
    mj_resetData(m, d);
    d->qpos[2] = 0.5; d->qpos[3] = 1.0;
    for (int j = 0; j < NUM_JOINTS; ++j) d->qpos[qa[j]] = STAND_DOWN[j];
    mj_forward(m, d);
    int fl = mj_name2id(m, mjOBJ_BODY, "FL_foot");
    int fr = mj_name2id(m, mjOBJ_BODY, "FR_foot");
    int rl = mj_name2id(m, mjOBJ_BODY, "RL_foot");
    int rr = mj_name2id(m, mjOBJ_BODY, "RR_foot");
    double min_z = 1e9;
    for (int b : {fl, fr, rl, rr}) if (b >= 0) min_z = std::min(min_z, d->xpos[3*b+2]);
    d->qpos[2] -= min_z;
    mj_forward(m, d);

    // ── output files ─────────────────────────────────────────────────────────
    std::ofstream csv(csv_path);
    csv << "t,px,py,pz,vx,vy,vz,qw,roll_deg\n";

    const std::string qpos_path = csv_path.substr(0, csv_path.rfind('.')) + "_qpos.csv";
    std::ofstream qpos_log(qpos_path);

    // ── stand-up phase (PD, mirrors mppi_controller.cpp) ─────────────────────
    printf("Standing up (%.1f s)...\n", STANDUP_SECS + HOLD_SECS);
    const double total_standup = STANDUP_SECS + HOLD_SECS;
    for (double t = 0.0; t < total_standup; t += task.dt) {
        const double phase = std::tanh(t / 1.2);
        const double kp    = phase * 50.0 + (1.0 - phase) * 20.0;
        for (int j = 0; j < NUM_JOINTS; ++j) {
            const double q_des = phase * STAND_UP[j] + (1.0 - phase) * STAND_DOWN[j];
            d->ctrl[JOINT_OFFSET + j] =
                kp * (q_des - d->qpos[qa[j]]) + 3.5 * (-d->qvel[qv[j]]);
        }
        mj_step(m, d);
    }
    printf("Stand-up complete. Body height: %.3f m\n", d->qpos[2]);

    // ── MPPI loop ─────────────────────────────────────────────────────────────
    printf("Running MPPI for %d convergence solves then logging...\n",
           CONVERGENCE_SOLVES);

    const double sim_duration = 10.0;   // seconds of MPPI control to record
    double sim_t = 0.0;
    int solve_count = 0;
    double solve_sum_ms = 0.0;
    bool converged = false;

    while (sim_t < sim_duration) {
        // --- MPPI solve ---
        RobotState state = read_state(m, d, qa, qv);

        auto t0 = std::chrono::steady_clock::now();
        double tau[NUM_JOINTS] = {};
        mppi.update(state, tau);
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();

        solve_sum_ms += ms;
        ++solve_count;

        if (!converged && solve_count >= CONVERGENCE_SOLVES) {
            printf("Converged (avg solve %.1f ms). Starting trajectory logging.\n",
                   solve_sum_ms / solve_count);
            converged = true;
        }

        // --- apply torques for one control step (substeps physics steps) ---
        for (int sub = 0; sub < task.substeps; ++sub) {
            for (int j = 0; j < NUM_JOINTS; ++j)
                d->ctrl[JOINT_OFFSET + j] = tau[j];
            mj_step(m, d);
        }
        sim_t += task.substeps * task.dt;

        // --- log ---
        if (converged) {
            const double qw  = d->qpos[3];
            const double roll = 2.0 * std::acos(std::clamp(std::abs(qw), 0.0, 1.0))
                                * 180.0 / M_PI;
            csv << sim_t << ","
                << d->qpos[0] << "," << d->qpos[1] << "," << d->qpos[2] << ","
                << d->qvel[0] << "," << d->qvel[1] << "," << d->qvel[2] << ","
                << qw << "," << roll << "\n";

            // save full qpos for GIF rendering
            for (int i = 0; i < m->nq; ++i)
                qpos_log << d->qpos[i] << (i < m->nq - 1 ? "," : "\n");
        }

        // --- safety: stop if robot falls ---
        if (d->qpos[2] < 0.1) {
            printf("Robot fell at t=%.2f s — stopping.\n", sim_t);
            break;
        }
    }

    printf("Done. Logged to %s and %s\n", csv_path.c_str(), qpos_path.c_str());
    printf("Avg MPPI solve: %.1f ms over %d solves\n",
           solve_sum_ms / solve_count, solve_count);

    mj_deleteData(d);
    mj_deleteModel(m);
    return 0;
}
