#pragma once

// Everything around the controller when running the robot: the stand-up
// procedure (used by the DDS controller and both sims), the MuJoCo sim
// harness helpers (model setup, spawn placement, state readout), and run_sim(),
// the standalone sim program both sims share. Its output files (CSV/qpos logs,
// rollout GIF) are written with log.h, and --save trial copying is trial_log.h.
//
// Shared on purpose: the stand-up (kStandDownPose/kStandUpPose, run_standup)
// and the sim loop are identical for the muscle and PD variants so their
// results stay comparable, and editing it changes both. If one variant needs
// different behavior, fork that piece into the variant's own folder first
// rather than changing it here (e.g. settle_standing() is muscle-only and PD
// never calls it; nominal_pose lives in PD's task config).

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

#include "activation_dynamics.h"   // filter_alpha() for the motor summary
#include "control_utils.h"
#include "log.h"
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
// running the stand-up and state readout.

// Fall, as Isaac Lab ends a Go2 velocity-task episode (base_contact:
// mdp.illegal_contact on body "base", 1 N; isaaclab_tasks
// manager_based/locomotion/velocity) with the head added, as robot_lab's SATA
// task does (head_contact on "Head.*"): the net contact force from outside the
// robot on the trunk box, Head_upper or Head_lower exceeds kFallForce. Isaac
// Lab's Go2 keeps those as three bodies; go2.xml merges them into the base
// body's three collision geoms, so each geom is checked on its own.
static constexpr double kFallForce = 1.0;   // N

// Collision geoms of the base body, and the robot's root body (contacts with
// the robot itself don't count; Isaac Lab's Go2 has self-collision off).
struct FallGeoms {
    std::vector<int> gid;
    int root = 0;
};

inline FallGeoms find_fall_geoms(const mjModel* m, int base_bid)
{
    FallGeoms fall;
    fall.root = m->body_rootid[base_bid];
    for (int g = 0; g < m->ngeom; ++g)
        if (m->geom_bodyid[g] == base_bid && (m->geom_contype[g] || m->geom_conaffinity[g]))
            fall.gid.push_back(g);
    return fall;
}

// Whether the step just taken pushed any fall geom with more than kFallForce of
// net contact force. Call right after mj_step() (see foot_contact_forces() in log.h).
inline bool has_fallen(const mjModel* m, const mjData* d, const FallGeoms& fall)
{
    for (int g : fall.gid) {
        mjtNum net[3] = {};
        for (int c = 0; c < d->ncon; ++c) {
            const mjContact& con = d->contact[c];
            const int side = con.geom[0] == g ? 0 : con.geom[1] == g ? 1 : -1;
            if (side < 0 || m->body_rootid[m->geom_bodyid[con.geom[1 - side]]] == fall.root) continue;

            // Contact-frame force in world axes: the force on geom[1] (the
            // frame's normal points from geom[0] to geom[1]).
            mjtNum f[6], fw[3];
            mj_contactForce(m, d, c, f);
            mju_mulMatTVec(fw, con.frame, f, 3, 3);
            const double sign = side == 1 ? 1.0 : -1.0;
            for (int k = 0; k < 3; ++k) net[k] += sign * fw[k];
        }
        if (mju_norm3(net) > kFallForce) return true;
    }
    return false;
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

// Stand-up physics step, whatever the run's own step (task motor.physics_dt),
// so every run hands over from the same standing state. At 2 ms the stand-up
// PD's explicit damping stays stable at any joint damping (kd·Δt/I = 0.43 for
// the calf in swing; 10 ms gives 2.1, past the explicit limit of 2).
static constexpr double kStandupDt = 0.002;

// Run the stand-up ramp plus hold (kStandupSecs + kHoldSecs) with software PD,
// stepping m at kStandupDt; m's own timestep is restored afterwards.
inline void run_standup(mjModel* m, mjData* d, const int qa[NUM_JOINTS], const int qv[NUM_JOINTS])
{
    const double run_dt = m->opt.timestep;
    m->opt.timestep = kStandupDt;

    const int steps = static_cast<int>(std::lround((kStandupSecs + kHoldSecs) / kStandupDt));
    for (int k = 0; k < steps; ++k) {
        double kp, q_des[NUM_JOINTS];
        standup_targets(k * kStandupDt, kp, q_des);
        for (int j = 0; j < NUM_JOINTS; ++j)
            d->ctrl[j] = unitree_pd_torque(
                kp, kStandupKd, q_des[j], d->qpos[qa[j]], /*dq_des=*/0.0, d->qvel[qv[j]], /*tau_ff=*/0.0);
        mj_step(m, d);
    }

    m->opt.timestep = run_dt;
}

// Settled standing state: joint angles, and the joint torques the actuators must
// supply to hold them (qfrc_bias − qfrc_constraint, so ground contact carries its
// share), in actuator order.
struct StandingEquilibrium {
    double q[NUM_JOINTS]   = {};
    double tau[NUM_JOINTS] = {};
};

// Stand the robot up on its own mjData for model m exactly as the sims do
// (place_on_ground + run_standup) and return the settled state. m's joint
// damping should already match the run's physical damping (e.g. muscle.kd_sim),
// so the result is the same state the sims hand over to MPPI.
inline StandingEquilibrium settle_standing(mjModel* m, double spawn_height_offset)
{
    mjData* d = mj_makeData(m);
    int qa[NUM_JOINTS], qv[NUM_JOINTS];
    joint_addresses(m, qa, qv);
    place_on_ground(m, d, qa, spawn_height_offset);
    run_standup(m, d, qa, qv);
    mj_forward(m, d);   // bias/constraint forces for the final state, stand-up PD still applied

    StandingEquilibrium eq;
    for (int j = 0; j < NUM_JOINTS; ++j) {
        eq.q[j]   = d->qpos[qa[j]];
        eq.tau[j] = d->qfrc_bias[qv[j]] - d->qfrc_constraint[qv[j]];
    }
    mj_deleteData(d);
    return eq;
}

// ============================================================================
// Standalone sim program
// ============================================================================

// What a standalone sim customises about run_sim(). Controller must provide
// task_ref() (with model_path, dt, motor, sim_duration, spawn_height_offset),
// advance_phase(), update(), actuate(), motor_damping() and task_success().
template <class Controller>
struct SimSpec {
    std::string default_yaml;   // task file when [yaml] isn't given
    std::string default_csv;    // output CSV when neither [output.csv] nor --name is given;
                                // its folder is where --name <run> writes <run>.csv

    // Joint damping for the simulated model, matching the controller's rollout model.
    std::function<const double*(const Controller&)> joint_damping;

    // Position logged as px/py/pz — whatever the controller's cost scores.
    // Called right after mj_step(), so derived quantities in d (xpos,
    // subtree_com, ...) are still at the pre-step state; d is mutable so a hook
    // can recompute what it reads (the next mj_step recomputes them anyway).
    std::function<const double*(const mjModel*, mjData*, int base_bid)> log_position;

    // Variant-specific CSV columns, each written with a leading ','. extra_row
    // also gets the state after the step its row records; it runs outside the
    // timed solve, so a hook may run rollouts without inflating solve_ms.
    std::function<void(std::ostream&)>                                 extra_header;
    std::function<void(std::ostream&, Controller&, const RobotState&)> extra_row;
};

// Standalone MuJoCo simulation around an MPPI controller. No DDS, no real-time
// constraint — MPPI runs as fast as possible against a local mjData
// simulation. Usage:
//   <sim> [task] [yaml] [output.csv] [--name <run>] [--save <name>] [--no-gif]
//
// Output files are all named after the output CSV: <out>.csv, <out>_qpos.csv
// and <out>.gif. By default that's spec.default_csv; --name <run> keeps the
// default folder but names the files <run>.*, and an explicit [output.csv]
// path sets it directly (a missing ".csv" is appended either way).
//
// Stands the robot up, runs kConvergenceSolves solves, then logs one CSV row
// (and one qpos row) per control step until sim_duration elapses, the robot
// falls, or the task's final phase is reached and held. Afterwards it deletes
// any existing <output>.gif and, unless --no-gif is given, renders the logged
// rollout to a fresh one; with --save the GIF is copied into the trial too. The
// console lines are parsed by run_trials.sh, so keep their wording stable.
template <class Controller>
int run_sim(int argc, char** argv, const SimSpec<Controller>& spec)
{
    // --save, --name and --no-gif are pulled out first so they can sit anywhere
    // on the command line without shifting the positional [task] [yaml]
    // [output.csv] arguments.
    std::string trial_name, run_name;
    bool make_gif = true;
    std::vector<std::string> args;
    const std::vector<std::string> raw = trial_log::parse_args(argc, argv, trial_name);
    for (size_t i = 0; i < raw.size(); ++i) {
        const std::string& arg = raw[i];
        if (i == 0) { args.push_back(arg); continue; }   // argv[0]
        if (arg == "--no-gif") {
            make_gif = false;
        } else if (arg == "--name") {
            if (i + 1 >= raw.size()) { fprintf(stderr, "--name needs a run name\n"); return 1; }
            run_name = raw[++i];
        } else if (arg.rfind("--name=", 0) == 0) {
            run_name = arg.substr(7);
        } else {
            args.push_back(arg);
        }
    }
    const size_t nargs = args.size();

    if (!run_name.empty() && nargs >= 4) {
        fprintf(stderr, "Give either an [output.csv] path or --name, not both.\n");
        return 1;
    }
    if (run_name.find("..") != std::string::npos
        || std::filesystem::path(run_name).is_absolute()) {
        fprintf(stderr, "--name '%s' must be a plain name (optionally a subfolder), "
                        "not a path outside the output folder.\n", run_name.c_str());
        return 1;
    }

    const std::string task_name = (nargs >= 2) ? args[1] : "walk";
    const std::string yaml_path = (nargs >= 3) ? args[2] : spec.default_yaml;
    const std::string csv_path  = with_csv_extension(
        (nargs >= 4)        ? args[3]
        : run_name.empty()  ? spec.default_csv
        : (std::filesystem::path(spec.default_csv).parent_path() / run_name).string());

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
    m->opt.timestep = task.motor.physics_dt;
    mjData* d = mj_makeData(m);

    // Resolve the base body (mirrors the controllers' base_bid_ resolution).
    const int base_bid = find_base_body(m);

    // Foot geoms, for the per-foot contact forces logged with every row.
    const FootGeoms feet = find_foot_geoms(m);

    // Base-body geoms, for the fall check (has_fallen()).
    const FallGeoms fall_geoms = find_fall_geoms(m, base_bid);
    if (fall_geoms.gid.empty()) {
        fprintf(stderr, "Base body has no collision geoms, so falls can't be detected.\n");
        return 1;
    }

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
    run_standup(m, d, qa, qv);
    printf("Stand-up complete. Body height: %.3f m\n", d->qpos[2]);

    // Hand over to the controller's motor (common/motor.h): from here on it
    // applies the torque limits, and its implicit damping share joins the
    // physical joint damping.
    for (int j = 0; j < NUM_JOINTS; ++j) {
        m->dof_damping[qv[j]] += mppi.motor_damping(j);
        m->actuator_ctrllimited[j] = 0;
    }
    const MotorParams& motor = task.motor;
    printf("Motor: bandwidth %.6g /s (cutoff %.6g Hz, alpha %.6g per step), %d physics steps of %.6g ms per control step, %s damping\n",
           motor.bandwidth, motor.bandwidth / (2.0 * M_PI), filter_alpha(motor.bandwidth, motor.physics_dt),
           motor.substeps, 1000.0 * motor.physics_dt,
           motor.damping == MotorDamping::Implicit ? "implicit" : "explicit");

    // ── MPPI loop ─────────────────────────────────────────────────────────────
    printf("Running MPPI for %d convergence solves then logging...\n",
           kConvergenceSolves);

    const double sim_duration = task.sim_duration;   // seconds of MPPI control to record
    double sim_t = 0.0;
    int solve_count = 0;
    double solve_sum_ms = 0.0;
    bool converged = false;
    int logged_rows = 0;

    while (sim_t < sim_duration) {
        // --- MPPI solve ---
        RobotState state = read_state(d, qa, qv);

        auto t0 = std::chrono::steady_clock::now();
        mppi.advance_phase(state);
        mppi.update(state);
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();

        solve_sum_ms += ms;
        ++solve_count;

        if (!converged && solve_count >= kConvergenceSolves) {
            printf("Converged (avg solve %.1f ms). Starting trajectory logging.\n",
                   solve_sum_ms / solve_count);
            converged = true;
        }

        // --- one control step: the actuators run every physics step on the
        //     command update() chose. Falls are checked every physics step and
        //     end the run after the control step, as in Isaac Lab ---
        bool fell = false;
        for (int i = 0; i < task.motor.substeps; ++i) {
            double ctrl[NUM_JOINTS];
            mppi.actuate(read_state(d, qa, qv), ctrl);
            for (int j = 0; j < NUM_JOINTS; ++j) d->ctrl[j] = ctrl[j];
            mj_step(m, d);
            if (has_fallen(m, d, fall_geoms)) fell = true;
        }
        sim_t += task.dt;

        // --- log ---
        if (converged) {
            write_csv_row_base(csv, sim_t, spec.log_position(m, d, base_bid),
                               m, d, qv, feet, ms);
            spec.extra_row(csv, mppi, read_state(d, qa, qv));
            csv << "\n";

            // save full qpos for GIF rendering
            write_qpos_row(qpos_log, m, d);
            ++logged_rows;
        }

        // --- stop if robot fell ---
        if (fell) {
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

    // Flush before rendering/copying — a run that fell still gets its partial logs saved.
    csv.close();
    qpos_log.close();

    // ── rollout GIF ──────────────────────────────────────────────────────────
    // Always clear this output's previous GIF first — also under --no-gif and
    // when there's nothing to render — so these CSVs are never left sitting next
    // to another run's GIF.
    const std::string gif_path = gif_path_for(csv_path);
    {
        std::error_code ec;
        std::filesystem::remove(gif_path, ec);
    }
    if (make_gif) {
        if (logged_rows == 0) {
            printf("No rollout logged — skipping GIF.\n");
        } else {
            printf("Rendering rollout GIF...\n");
            if (render_rollout_gif(qpos_path, gif_path, task_name, yaml_path))
                printf("GIF saved to %s\n", gif_path.c_str());
            else
                fprintf(stderr, "GIF rendering failed (set MUSCLE_MPPI_PYTHON to a Python with "
                                "mujoco, numpy, pyyaml and Pillow) — logs are unaffected.\n");
        }
    }

    if (!trial_name.empty()) {
        std::vector<std::string> files = {csv_path, qpos_path};
        if (make_gif) files.push_back(gif_path);   // trial_log::save skips files that don't exist
        const std::string trial_dir = trial_log::save(trial_name, files);
        if (!trial_dir.empty()) printf("Trial saved to %s\n", trial_dir.c_str());
    }

    mj_deleteData(d);
    mj_deleteModel(m);
    return 0;
}
