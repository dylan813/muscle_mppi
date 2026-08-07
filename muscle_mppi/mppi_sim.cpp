// Standalone MuJoCo simulation test for MPPILocomotion.
// No DDS, no real-time constraint — MPPI runs as fast as possible against a
// local mjData simulation. Useful for verifying the controller works before
// worrying about latency.
//
// Run from muscle_mppi/muscle_mppi/:
//   ./build/mppi_sim [task] [yaml] [output.csv]
//
// Output CSV columns:
//   t, px, py, pz, vx, vy, vz, qw, roll_deg, dq_j0..dq_j{NUM_JOINTS-1}, act_m0..act_m{NUM_MUSCLES-1}
//   px/py/pz are the whole-robot (trunk + legs) center of mass, i.e. the base
//   body's subtree_com — matching what step_cost() scores against goal_pos
//   in mppi_locomotion.cpp, not the trunk frame origin.
//   vx/vy/vz are body-frame linear velocity (rotated from the free joint's
//   world-frame qvel[0:3]), matching the body-frame convention step_cost()
//   uses when comparing against cmd_.vx/vy in mppi_locomotion.cpp.

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

    // Resolve the base body (mirrors MPPILocomotion::base_bid_ resolution)
    // so logged position matches the whole-robot CoM the cost function scores.
    int base_bid = 1;
    for (const char* name : {"trunk", "base", "base_link"}) {
        int bid = mj_name2id(m, mjOBJ_BODY, name);
        if (bid >= 0) { base_bid = bid; break; }
    }

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
    d->qpos[2] += task.spawn_height_offset - min_z;
    mj_forward(m, d);

    // ── output files ─────────────────────────────────────────────────────────
    std::ofstream csv(csv_path);
    csv << "t,px,py,pz,vx,vy,vz,qw,roll_deg";
    for (int j = 0; j < NUM_JOINTS; ++j) csv << ",dq_j" << j;
    for (int m = 0; m < NUM_MUSCLES; ++m) csv << ",act_m" << m;
    csv << "\n";

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

    const double sim_duration = task.sim_duration;   // seconds of MPPI control to record
    double sim_t = 0.0;
    int solve_count = 0;
    double solve_sum_ms = 0.0;
    bool converged = false;

    while (sim_t < sim_duration) {
        // --- MPPI solve ---
        RobotState state = read_state(m, d, qa, qv);

        auto t0 = std::chrono::steady_clock::now();
        double tau[NUM_JOINTS] = {};
        mppi.advance_phase(state);
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

        // --- apply torques for one control step ---
        for (int j = 0; j < NUM_JOINTS; ++j)
            d->ctrl[JOINT_OFFSET + j] = tau[j];
        mj_step(m, d);
        sim_t += task.dt;

        // --- log ---
        if (converged) {
            const double qw  = d->qpos[3];
            const double roll = 2.0 * std::acos(std::clamp(std::abs(qw), 0.0, 1.0))
                                * 180.0 / M_PI;

            // Rotate world-frame free-joint velocity into body frame (xmat
            // is the body->world rotation, so its transpose maps world->body).
            double xmat[9];
            mju_quat2Mat(xmat, d->qpos + 3);
            const double vwx = d->qvel[0], vwy = d->qvel[1], vwz = d->qvel[2];
            const double vx_body = vwx * xmat[0] + vwy * xmat[3] + vwz * xmat[6];
            const double vy_body = vwx * xmat[1] + vwy * xmat[4] + vwz * xmat[7];
            const double vz_body = vwx * xmat[2] + vwy * xmat[5] + vwz * xmat[8];

            const double* com = d->subtree_com + base_bid * 3;

            csv << sim_t << ","
                << com[0] << "," << com[1] << "," << com[2] << ","
                << vx_body << "," << vy_body << "," << vz_body << ","
                << qw << "," << roll;
            for (int j = 0; j < NUM_JOINTS; ++j) csv << "," << d->qvel[qv[j]];
            const double* act = mppi.activation();
            for (int j = 0; j < NUM_MUSCLES; ++j) csv << "," << act[j];
            csv << "\n";

            // save full qpos for GIF rendering
            for (int i = 0; i < m->nq; ++i)
                qpos_log << d->qpos[i] << (i < m->nq - 1 ? "," : "\n");
        }

        // --- safety: stop if robot falls ---
        if (d->qpos[2] < 0.1) {
            printf("Robot fell at t=%.2f s — stopping.\n", sim_t);
            break;
        }

        // --- stop once the task's final phase has been reached and held ---
        if (converged && mppi.task_success()) {
            printf("Task complete at t=%.2f s — stopping.\n", sim_t);
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
