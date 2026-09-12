#pragma once

// MuJoCo "real world" harness pieces shared by the standalone sims
// (muscle/mppi_sim.cpp, pd/pd_mppi_sim.cpp): model setup, spawn placement,
// stand-up, state readout and CSV logging.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <string>
#include <system_error>

#include <mujoco/mujoco.h>

#include "control_utils.h"
#include "standup.h"
#include "types.h"

// The sims stop once the trunk drops below this height (m).
static constexpr double kFallHeight = 0.1;

// Base (trunk) body id, trying the names used across Unitree models; falls back to 1.
inline int find_base_body(const mjModel* m)
{
    for (const char* name : {"trunk", "base", "base_link"}) {
        int bid = mj_name2id(m, mjOBJ_BODY, name);
        if (bid >= 0) return bid;
    }
    return 1;
}

// qpos/qvel addresses of the joint driven by each of the first NUM_JOINTS actuators.
inline void joint_addresses(const mjModel* m, int qa[NUM_JOINTS], int qv[NUM_JOINTS])
{
    for (int j = 0; j < NUM_JOINTS; ++j) {
        int jid = m->actuator_trnid[2 * j];
        qa[j]   = m->jnt_qposadr[jid];
        qv[j]   = m->jnt_dofadr[jid];
    }
}

inline RobotState read_state(const mjData* d, const int qa[NUM_JOINTS], const int qv[NUM_JOINTS])
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

// Reset to the stand-down pose and drop the robot so its lowest foot sits at
// z = spawn_height_offset (the terrain height under the spawn point).
inline void place_on_ground(const mjModel* m, mjData* d, const int qa[NUM_JOINTS],
                            double spawn_height_offset)
{
    mj_resetData(m, d);
    d->qpos[2] = 0.5; d->qpos[3] = 1.0;
    for (int j = 0; j < NUM_JOINTS; ++j) d->qpos[qa[j]] = kStandDownPose[j];
    mj_forward(m, d);
    int fl = mj_name2id(m, mjOBJ_BODY, "FL_foot");
    int fr = mj_name2id(m, mjOBJ_BODY, "FR_foot");
    int rl = mj_name2id(m, mjOBJ_BODY, "RL_foot");
    int rr = mj_name2id(m, mjOBJ_BODY, "RR_foot");
    double min_z = 1e9;
    for (int b : {fl, fr, rl, rr}) if (b >= 0) min_z = std::min(min_z, d->xpos[3*b+2]);
    d->qpos[2] += spawn_height_offset - min_z;
    mj_forward(m, d);
}

// Run the stand-up ramp plus hold (kStandupSecs + kHoldSecs) with software PD.
inline void run_standup(const mjModel* m, mjData* d,
                        const int qa[NUM_JOINTS], const int qv[NUM_JOINTS], double dt)
{
    const double total_standup = kStandupSecs + kHoldSecs;
    for (double t = 0.0; t < total_standup; t += dt) {
        double kp, q_des[NUM_JOINTS];
        standup_targets(t, kp, q_des);
        for (int j = 0; j < NUM_JOINTS; ++j)
            d->ctrl[j] = unitree_pd_torque(
                kp, kStandupKd, q_des[j], d->qpos[qa[j]], /*dq_des=*/0.0, d->qvel[qv[j]], /*tau_ff=*/0.0);
        mj_step(m, d);
    }
}

// Open an output log for writing. The output directory only holds gitignored
// CSVs, so a fresh checkout may not have it — create it rather than let
// ofstream fail silently. Prints an error and returns false on failure.
inline bool open_output(const std::string& path, std::ofstream& out)
{
    const std::filesystem::path dir = std::filesystem::path(path).parent_path();
    if (!dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
    }
    out.open(path);
    if (!out) { fprintf(stderr, "Cannot open %s for writing\n", path.c_str()); return false; }
    return true;
}

// Companion qpos log path: <name>.csv -> <name>_qpos.csv.
inline std::string qpos_path_for(const std::string& csv_path)
{
    return csv_path.substr(0, csv_path.rfind('.')) + "_qpos.csv";
}

// Columns every sim logs; each variant appends its own and then writes "\n".
inline void write_csv_header_base(std::ostream& csv)
{
    csv << "t,px,py,pz,vx,vy,vz,qw,roll_deg";
    for (int j = 0; j < NUM_JOINTS; ++j) csv << ",dq_j" << j;
}

// One row of the shared columns. `pos` is whichever position the variant's
// cost scores (whole-robot CoM for muscle, trunk origin for PD); velocity is
// the free joint's linear velocity in body-frame axes.
inline void write_csv_row_base(std::ostream& csv, double t, const double pos[3],
                               const mjData* d, const int qv[NUM_JOINTS])
{
    const double qw  = d->qpos[3];
    const double roll = 2.0 * std::acos(std::clamp(std::abs(qw), 0.0, 1.0))
                        * 180.0 / M_PI;

    // Rotate world-frame free-joint velocity into body frame (xmat
    // is the body->world rotation, so its transpose maps world->body).
    double xmat[9], v_body[3];
    mju_quat2Mat(xmat, d->qpos + 3);
    world_to_body(xmat, d->qvel, v_body);

    csv << t << ","
        << pos[0] << "," << pos[1] << "," << pos[2] << ","
        << v_body[0] << "," << v_body[1] << "," << v_body[2] << ","
        << qw << "," << roll;
    for (int j = 0; j < NUM_JOINTS; ++j) csv << "," << d->qvel[qv[j]];
}

// Full qpos row, used by analysis/render_gif.py.
inline void write_qpos_row(std::ostream& out, const mjModel* m, const mjData* d)
{
    for (int i = 0; i < m->nq; ++i)
        out << d->qpos[i] << (i < m->nq - 1 ? "," : "\n");
}

inline bool has_fallen(const mjData* d) { return d->qpos[2] < kFallHeight; }
