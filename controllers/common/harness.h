#pragma once

// Everything around the controller when running the robot: the stand-up
// procedure (used by the DDS controller and both sims), the MuJoCo sim
// harness helpers (model setup, spawn placement, state readout, CSV logging),
// and run_sim(), the standalone sim program both sims share. The --save trial
// copying itself lives separately in trial_log.h.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <ostream>
#include <string>
#include <system_error>
#include <vector>

#include <mujoco/mujoco.h>

#include "control_utils.h"
#include "task_config.h"
#include "trial_log.h"

// ============================================================================
// Stand-up procedure
// ============================================================================

// Stand-up procedure shared by the DDS controller (muscle/mppi_controller.cpp)
// and both standalone sims: a tanh-ramped PD blend from the crouched
// stand-down pose to the standing pose, mirroring unitree_mujoco's stand_go2.cpp.

// Crouched pose the robot starts from.
static constexpr double kStandDownPose[NUM_JOINTS] = {
     0.0473455,  1.22187, -2.44375,   // FR
    -0.0473455,  1.22187, -2.44375,   // FL
     0.0473455,  1.22187, -2.44375,   // RR
    -0.0473455,  1.22187, -2.44375,   // RL
};

// Standing pose the ramp converges to.
static constexpr double kStandUpPose[NUM_JOINTS] = {
    0.0, 0.67, -1.3,   // FR
    0.0, 0.67, -1.3,   // FL
    0.0, 0.67, -1.3,   // RR
    0.0, 0.67, -1.3,   // RL
};

static constexpr double kStandupSecs = 3.0;   // tanh ramp
static constexpr double kHoldSecs    = 1.0;   // hold pose before handing to MPPI
static constexpr double kStandupKd   = 3.5;   // PD damping during stand-up

// MPPI solves to run before handing control to it (and, in the sims, before logging).
static constexpr int    kConvergenceSolves = 10;

// PD targets at `t` seconds into the stand-up: kp ramps 20 -> 50 and the
// joint target blends stand-down -> stand-up, both along tanh(t / 1.2).
inline void standup_targets(double t, double& kp, double q_des[NUM_JOINTS])
{
    const double phase = std::tanh(t / 1.2);
    kp = phase * 50.0 + (1.0 - phase) * 20.0;
    for (int j = 0; j < NUM_JOINTS; ++j)
        q_des[j] = phase * kStandUpPose[j] + (1.0 - phase) * kStandDownPose[j];
}

// ============================================================================
// MuJoCo sim harness
// ============================================================================

// MuJoCo "real world" harness pieces shared by the standalone sims
// (muscle/mppi_sim.cpp, pd/pd_mppi_sim.cpp): model setup, spawn placement,
// stand-up, state readout and CSV logging.

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

// ============================================================================
// Standalone sim program
// ============================================================================

// What a standalone sim customises about run_sim(). Controller must provide
// task_ref() (with model_path, dt, sim_duration, spawn_height_offset),
// advance_phase(), update() and task_success().
template <class Controller>
struct SimSpec {
    std::string default_yaml;   // task file when [yaml] isn't given
    std::string default_csv;    // output CSV when [output.csv] isn't given

    // Joint damping for the simulated model, matching the controller's rollout model.
    std::function<const double*(const Controller&)> joint_damping;

    // Position logged as px/py/pz — whatever the controller's cost scores.
    std::function<const double*(const mjData*, int base_bid)> log_position;

    // Variant-specific CSV columns, each written with a leading ','.
    std::function<void(std::ostream&)>                    extra_header;
    std::function<void(std::ostream&, const Controller&)> extra_row;
};

// Standalone MuJoCo simulation around an MPPI controller. No DDS, no real-time
// constraint — MPPI runs as fast as possible against a local mjData
// simulation. Usage:
//   <sim> [task] [yaml] [output.csv] [--save <name>]
//
// Stands the robot up, runs kConvergenceSolves solves, then logs one CSV row
// (and one qpos row) per control step until sim_duration elapses, the robot
// falls, or the task's final phase is reached and held. The console lines are
// parsed by run_trials.sh, so keep their wording stable.
template <class Controller>
int run_sim(int argc, char** argv, const SimSpec<Controller>& spec)
{
    // --save is pulled out first so it can sit anywhere on the command line
    // without shifting the positional [task] [yaml] [output.csv] arguments.
    std::string trial_name;
    const std::vector<std::string> args = trial_log::parse_args(argc, argv, trial_name);
    const size_t nargs = args.size();

    const std::string task_name = (nargs >= 2) ? args[1] : "walk";
    const std::string yaml_path = (nargs >= 3) ? args[2] : spec.default_yaml;
    const std::string csv_path  = (nargs >= 4) ? args[3] : spec.default_csv;

    printf("Task: %s  |  YAML: %s  |  CSV: %s\n",
           task_name.c_str(), yaml_path.c_str(), csv_path.c_str());
    if (!trial_name.empty()) printf("Saving trial under: %s\n", trial_name.c_str());

    // ── load MPPI (also loads the model internally) ──────────────────────────
    Controller mppi(task_name, yaml_path);

    // ── load a separate sim model/data ───────────────────────────────────────
    const auto& task = mppi.task_ref();
    char err[1000];
    mjModel* m = mj_loadXML(task.model_path.c_str(), nullptr, err, sizeof(err));
    if (!m) { fprintf(stderr, "mj_loadXML: %s\n", err); return 1; }
    m->opt.timestep = task.dt;
    mjData* d = mj_makeData(m);

    // Resolve the base body (mirrors the controllers' base_bid_ resolution).
    const int base_bid = find_base_body(m);

    // set joint damping to match MPPI's internal model
    int qa[NUM_JOINTS], qv[NUM_JOINTS];
    joint_addresses(m, qa, qv);
    const double* damping = spec.joint_damping(mppi);
    for (int j = 0; j < NUM_JOINTS; ++j) m->dof_damping[qv[j]] = damping[j];

    // place robot feet on ground
    place_on_ground(m, d, qa, task.spawn_height_offset);

    // ── output files ─────────────────────────────────────────────────────────
    std::ofstream csv;
    if (!open_output(csv_path, csv)) return 1;
    write_csv_header_base(csv);
    spec.extra_header(csv);
    csv << "\n";

    const std::string qpos_path = qpos_path_for(csv_path);
    std::ofstream qpos_log(qpos_path);

    // ── stand-up phase (software PD) ─────────────────────────────────────────
    printf("Standing up (%.1f s)...\n", kStandupSecs + kHoldSecs);
    run_standup(m, d, qa, qv, task.dt);
    printf("Stand-up complete. Body height: %.3f m\n", d->qpos[2]);

    // ── MPPI loop ─────────────────────────────────────────────────────────────
    printf("Running MPPI for %d convergence solves then logging...\n",
           kConvergenceSolves);

    const double sim_duration = task.sim_duration;   // seconds of MPPI control to record
    double sim_t = 0.0;
    int solve_count = 0;
    double solve_sum_ms = 0.0;
    bool converged = false;

    while (sim_t < sim_duration) {
        // --- MPPI solve ---
        RobotState state = read_state(d, qa, qv);

        auto t0 = std::chrono::steady_clock::now();
        double tau[NUM_JOINTS] = {};
        mppi.advance_phase(state);
        mppi.update(state, tau);
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();

        solve_sum_ms += ms;
        ++solve_count;

        if (!converged && solve_count >= kConvergenceSolves) {
            printf("Converged (avg solve %.1f ms). Starting trajectory logging.\n",
                   solve_sum_ms / solve_count);
            converged = true;
        }

        // --- apply torques for one control step ---
        for (int j = 0; j < NUM_JOINTS; ++j)
            d->ctrl[j] = tau[j];
        mj_step(m, d);
        sim_t += task.dt;

        // --- log ---
        if (converged) {
            write_csv_row_base(csv, sim_t, spec.log_position(d, base_bid), d, qv);
            spec.extra_row(csv, mppi);
            csv << "\n";

            // save full qpos for GIF rendering
            write_qpos_row(qpos_log, m, d);
        }

        // --- safety: stop if robot falls ---
        if (has_fallen(d)) {
            printf("Robot fell at t=%.2f s — stopping.\n", sim_t);
            break;
        }

        // --- stop once the task's final phase has been reached and held ---
        // Both variants stop here, so every run of either controller records
        // the same span of the task (success, not the sim_duration clock).
        if (converged && mppi.task_success()) {
            printf("Task complete at t=%.2f s — stopping.\n", sim_t);
            break;
        }
    }

    printf("Done. Logged to %s and %s\n", csv_path.c_str(), qpos_path.c_str());
    printf("Avg MPPI solve: %.1f ms over %d solves\n",
           solve_sum_ms / solve_count, solve_count);

    // Flush before copying — a run that fell still gets its partial logs saved.
    csv.close();
    qpos_log.close();
    if (!trial_name.empty()) {
        const std::string trial_dir = trial_log::save(trial_name, {csv_path, qpos_path});
        if (!trial_dir.empty()) printf("Trial saved to %s\n", trial_dir.c_str());
    }

    mj_deleteData(d);
    mj_deleteModel(m);
    return 0;
}
