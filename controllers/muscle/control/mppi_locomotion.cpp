#include "mppi_locomotion.h"
#include "activation_gait.h"
#include "../../common/control_utils.h"
#include "../../common/harness.h"   // settle_standing() for the warm start

#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <yaml-cpp/yaml.h>

// ============================================================================
// Named gaits
// ============================================================================
//
// Mirrors RTWholeBodyMPPI's GAIT_*_PATH constants (mppi_locomotion.py): a fixed
// set of categorical gaits, each the activation version of one joint-space
// source gait in controllers/pd/gaits/. The activation files are generated into
// controllers/muscle/gaits/ at startup whenever they're missing or were built
// with different muscle parameters/stiffness (muscle/control/activation_gait.h).
// A phase selects a gait by name (TaskPhase::desired_gait) or, as an escape
// hatch, an explicit TSV path (TaskPhase::gait_path, used as-is) — see
// resolve_gait_key() in common/gait.h.
static const std::unordered_map<std::string, std::string> kNamedGaitSources = {
    {"in_place",  "FAST_0_0_10cm"},
    {"walk",      "MED_0_1_10cm"},
    {"walk_fast", "FAST_0_1_10cm"},
    {"trot",      "MED_0_5_15cm"},
};

static NamedGaitPaths named_activation_gaits()
{
    NamedGaitPaths named;
    for (const auto& kv : kNamedGaitSources) named[kv.first] = activation_gait_path(kv.second);
    return named;
}
static const NamedGaitPaths kNamedGaits = named_activation_gaits();

// ============================================================================
// Constructor
// ============================================================================

MPPILocomotion::MPPILocomotion(const std::string& task_name, const std::string& yaml_path)
    : BaseMPPI(load_task(task_name, yaml_path))
{
    muscle_ = task_.muscle;
    solve_gait_ref_.assign(task_.horizon * NUM_MUSCLES, 0.0);

    // BaseMPPI already initializes trajectory_, noise_, costs_ to the correct
    // sizes (horizon × NUM_MUSCLES). Activations are clamped to [0, 1] directly
    // in rollout() and update().

    base_bid_ = find_base_body(model_);

    {
        YAML::Node root = YAML::LoadFile(yaml_path);
        const YAML::Node& c = root[task_name]["cost"];
        cost_.pos_x       = c["pos_x"]       ? c["pos_x"].as<double>()       : 0.0;
        cost_.pos_y       = c["pos_y"]       ? c["pos_y"].as<double>()       : 0.0;
        cost_.pos_z       = c["pos_z"]       ? c["pos_z"].as<double>()       : 0.0;
        cost_.orientation = c["orientation"] ? c["orientation"].as<double>() : 0.0;
        cost_.vel_x       = c["vel_x"]       ? c["vel_x"].as<double>()       : 0.0;
        cost_.vel_y       = c["vel_y"]       ? c["vel_y"].as<double>()       : 0.0;
        cost_.vel_z       = c["vel_z"]       ? c["vel_z"].as<double>()       : 0.0;
        cost_.ang_vel     = c["ang_vel"]     ? c["ang_vel"].as<double>()     : 0.0;
        if (c["gait_ref_weights"]) {
            const auto& gw = c["gait_ref_weights"];
            for (int j = 0; j < NUM_JOINTS; ++j)
                cost_.gait_ref_weights[j] = gw[j].as<double>();
        }
    }

    // Make every named gait the phases use current for these muscle parameters
    // (regenerating it if needed), then load the task's gaits and activate
    // phase 0 (see PhaseSequencer::init()), then apply phase 0's noise override
    // against the YAML baseline.
    for (const TaskPhase& p : task_.phases)
        if (p.gait_path.empty())
            ensure_activation_gait(kNamedGaitSources.at(resolve_gait_key(p, kNamedGaits)), muscle_);
    std::memcpy(base_noise_sigma_act_, task_.noise_sigma_act, sizeof(base_noise_sigma_act_));
    phases_.init(task_.phases, kNamedGaits);
    apply_phase_noise();

    // Warm start: seed trajectory_ and real_act_ with activations that hold the
    // robot's standing pose. The planner model is stood up exactly as the sims
    // do (settle_standing(), common/harness.h — same stand-up, timestep, joint
    // damping and spawn height), so the pose and holding torques are the ones
    // MPPI actually takes over from. Each joint's activation pair is then the
    // point on its torque-balance line at muscle.stiffness (hill_invert_torque,
    // static so FV = 1 — the same inversion the gait-tracking cost uses).
    {
        const StandingEquilibrium stand = settle_standing(model_, task_.spawn_height_offset);
        double seed[NUM_MUSCLES] = {};
        for (int j = 0; j < NUM_JOINTS; ++j)
            hill_invert_torque(stand.q[j], /*dq=*/0.0, stand.tau[j], j, muscle_,
                               muscle_.stiffness, seed[2 * j], seed[2 * j + 1]);
        for (int t = 0; t < task_.horizon; ++t)
            for (int m = 0; m < NUM_MUSCLES; ++m)
                trajectory_[t * NUM_MUSCLES + m] = seed[m];
        std::memcpy(real_act_, seed, NUM_MUSCLES * sizeof(double));

        // Motor (common/motor.h): its velocity feedback is the driver's
        // damping, and it starts out applying the holding torque. From here on
        // it applies the torque limits, and its implicit share joins the joint
        // damping MuJoCo integrates.
        double lo[NUM_JOINTS], hi[NUM_JOINTS];
        for (int j = 0; j < NUM_JOINTS; ++j) {
            lo[j] = model_->actuator_ctrlrange[2 * j];
            hi[j] = model_->actuator_ctrlrange[2 * j + 1];
        }
        motor_.init(task_.motor, task_.motor.driver_kd, lo, hi);
        motor_.settle(stand.tau, real_motor_tau_);
        for (int j = 0; j < NUM_JOINTS; ++j) {
            model_->dof_damping[act_qvel_adr_[j]] += motor_.implicit_damping(j);
            model_->actuator_ctrllimited[j] = 0;
        }
    }
}

void MPPILocomotion::apply_phase_noise()
{
    // Per-phase noise_sigma_act override, falling back to the task-level
    // baseline — mirrors RTWholeBodyMPPI's next_goal(), which doubles thigh/
    // calf exploration noise specifically during trot phases.
    const TaskPhase* p = phases_.current_phase();
    if (p && p->has_noise_sigma_act)
        std::memcpy(task_.noise_sigma_act, p->noise_sigma_act, sizeof(task_.noise_sigma_act));
    else
        std::memcpy(task_.noise_sigma_act, base_noise_sigma_act_, sizeof(task_.noise_sigma_act));
}

// ============================================================================
// Rollout
// ============================================================================

double MPPILocomotion::rollout(int s, const RobotState& state)
{
    mjData* d = data_[s];
    set_mj_state(d, state);

    double activation[NUM_MUSCLES], motor_tau[NUM_JOINTS];
    std::memcpy(activation, real_act_, NUM_MUSCLES * sizeof(double));
    std::memcpy(motor_tau, real_motor_tau_, NUM_JOINTS * sizeof(double));

    const int stride  = task_.horizon * NUM_MUSCLES;
    double total_cost = 0.0;

    for (int t = 0; t < task_.horizon; ++t) {
        double act_cmd[NUM_MUSCLES];
        for (int m = 0; m < NUM_MUSCLES; ++m) {
            double noisy = trajectory_[t * NUM_MUSCLES + m]
                         + noise_[s * stride + t * NUM_MUSCLES + m];
            act_cmd[m] = std::clamp(noisy, 0.0, 1.0);
        }

        step_model(d, act_cmd, activation, motor_tau);

        double gait_ref[NUM_MUSCLES] = {};
        if (phases_.active_gait()) phases_.active_gait()->get_phase(t, gait_ref);
        total_cost += step_cost(d, gait_ref).total();
    }

    return std::isfinite(total_cost) ? total_cost : 1e6;
}

void MPPILocomotion::step_model(mjData* d, const double act_cmd[NUM_MUSCLES],
                                double activation[NUM_MUSCLES], double motor_tau[NUM_JOINTS]) const
{
    for (int i = 0; i < task_.motor.substeps; ++i) {
        double q[NUM_JOINTS], dq[NUM_JOINTS], tau[NUM_JOINTS], ctrl[NUM_JOINTS], tau_cmd[NUM_JOINTS];
        for (int j = 0; j < NUM_JOINTS; ++j) {
            q[j]  = d->qpos[act_qpos_adr_[j]];
            dq[j] = d->qvel[act_qvel_adr_[j]];
        }

        hill_compute_torques(act_cmd, q, dq, muscle_, task_.motor.physics_dt, activation, tau);
        motor_.command(tau, dq, motor_tau, ctrl, tau_cmd);

        for (int j = 0; j < model_->nu; ++j) d->ctrl[j] = 0.0;
        for (int j = 0; j < NUM_JOINTS; ++j) d->ctrl[j] = ctrl[j];

        mj_step(model_, d);

        for (int j = 0; j < NUM_JOINTS; ++j) dq[j] = d->qvel[act_qvel_adr_[j]];
        motor_.applied(ctrl, dq, motor_tau);
    }
}

void MPPILocomotion::actuate(const RobotState& state, double ctrl[NUM_JOINTS])
{
    settle_motor(state);

    double tau[NUM_JOINTS];
    hill_compute_torques(last_act_cmd_, state.q, state.dq, muscle_, task_.motor.physics_dt, real_act_, tau);
    motor_.command(tau, state.dq, real_motor_tau_, ctrl, last_tau_);

    std::memcpy(pending_ctrl_, ctrl, NUM_JOINTS * sizeof(double));
    pending_ = true;
}

void MPPILocomotion::settle_motor(const RobotState& state)
{
    if (!pending_) return;
    motor_.applied(pending_ctrl_, state.dq, real_motor_tau_);
    pending_ = false;
}

// ============================================================================
// Cost function
// ============================================================================

// mj_step() integrates qpos/qvel but leaves every derived quantity (subtree_com,
// xmat, cvel, qfrc_bias, ...) at the PRE-step state. Recompute what
// step_cost() reads for the new state — otherwise the base and gait terms would
// score the previous step while orientation and the joint terms (read straight
// from qpos/qvel) score this one:
//   mj_kinematics + mj_comPos   subtree_com, xmat (position, orientation)
//   mj_comVel + mj_subtreeVel   subtree_linvel (velocity)
//   mj_rne without acceleration qfrc_bias, gravity + Coriolis (gait); needs
//                               mj_comPos and mj_comVel first
// The next mj_step recomputes all of these anyway, so the dynamics are unaffected.
void MPPILocomotion::refresh_derived(mjData* d) const
{
    mj_kinematics(model_, d);
    mj_comPos(model_, d);
    mj_comVel(model_, d);
    mj_subtreeVel(model_, d);
    mj_rne(model_, d, 0, d->qfrc_bias);
}

// Whole-robot CoM position (world frame) and CoM velocity (body-frame axes).
//
// base_bid_ is the trunk (root) body, so its subtree is the entire robot
// (trunk + all legs): d->subtree_com is the mass-weighted CoM of that whole
// subtree and d->subtree_linvel its velocity, in world-aligned axes, rotated
// here into the body frame.
void MPPILocomotion::base_com_state(const mjData* d, double com_pos[3], double com_vel_body[3]) const
{
    const double* com = d->subtree_com + base_bid_ * 3;
    com_pos[0] = com[0];
    com_pos[1] = com[1];
    com_pos[2] = com[2];

    world_to_body(d->xmat + base_bid_ * 9, d->subtree_linvel + base_bid_ * 3, com_vel_body);
}

CostTerms MPPILocomotion::step_cost(mjData* d, const double gait_ref[NUM_MUSCLES]) const
{
    const CostWeights& w = cost_;
    CostTerms cost;

    refresh_derived(d);

    double pos[3], vel_body[3];
    base_com_state(d, pos, vel_body);

    cost.pos = w.pos_x * std::abs(pos[0] - command().goal_pos[0])
             + w.pos_y * std::abs(pos[1] - command().goal_pos[1])
             + w.pos_z * std::abs(pos[2] - command().goal_pos[2]);

    const double q_dot  = d->qpos[3]*goal_quat_[0] + d->qpos[4]*goal_quat_[1]
                         + d->qpos[5]*goal_quat_[2] + d->qpos[6]*goal_quat_[3];
    const double q_dist = 1.0 - std::abs(q_dot);
    cost.orient = w.orientation * q_dist * q_dist;

    if (w.vel_x > 0.0 || w.vel_y > 0.0 || w.vel_z > 0.0) {
        const double ex = vel_body[0] - command().vx;
        const double ey = vel_body[1] - command().vy;
        cost.vel = w.vel_x * ex*ex + w.vel_y * ey*ey + w.vel_z * vel_body[2]*vel_body[2];
    }

    if (w.ang_vel > 0.0) {
        const double wx = d->qvel[3], wy = d->qvel[4], wz = d->qvel[5];
        cost.ang_vel = w.ang_vel * (wx*wx + wy*wy + wz*wz);
    }

    for (int j = 0; j < NUM_JOINTS; ++j) {
        if (w.gait_ref_weights[j] == 0.0) continue;
        const double q_j   = d->qpos[act_qpos_adr_[j]];
        const double dq_j  = d->qvel[act_qvel_adr_[j]];
        const double tau_j = d->qfrc_bias[act_qvel_adr_[j]];
        double a1_imp, a2_imp;
        hill_invert_torque(q_j, dq_j, tau_j, j, muscle_, muscle_.stiffness, a1_imp, a2_imp);
        const double e1 = a1_imp - gait_ref[2 * j];
        const double e2 = a2_imp - gait_ref[2 * j + 1];
        cost.gait += w.gait_ref_weights[j] * (e1*e1 + e2*e2);
    }

    return cost;
}

CostTerms MPPILocomotion::executed_cost(const RobotState& state)
{
    mjData* d = data_[task_.n_samples];
    set_mj_state(d, state);
    return step_cost(d, solve_gait_ref_.data());
}

CostTerms MPPILocomotion::plan_cost()
{
    mjData* d = data_[task_.n_samples];
    set_mj_state(d, solve_state_);

    double activation[NUM_MUSCLES], motor_tau[NUM_JOINTS];
    std::memcpy(activation, solve_act_, NUM_MUSCLES * sizeof(double));
    std::memcpy(motor_tau, solve_motor_tau_, NUM_JOINTS * sizeof(double));

    CostTerms sum;
    for (int t = 0; t < task_.horizon; ++t) {
        step_model(d, &trajectory_[t * NUM_MUSCLES], activation, motor_tau);
        sum += step_cost(d, &solve_gait_ref_[t * NUM_MUSCLES]);
    }
    return sum;
}

// ============================================================================
// Main solve
// ============================================================================

void MPPILocomotion::update(const RobotState& state)
{
    const auto t_start = std::chrono::steady_clock::now();

    settle_motor(state);

    if (!state.valid) {
        std::fill(last_act_cmd_, last_act_cmd_ + NUM_MUSCLES, 0.0);
        return;
    }

    // Goal-facing orientation target for this tick's cost, held fixed across
    // the whole rollout batch below (see common/control_utils.h).
    goal_facing_quat(command().goal_pos, state.pos, phases_.dwelling(), goal_quat_);

    // What this solve plans from, for executed_cost() and plan_cost().
    solve_state_ = state;
    std::memcpy(solve_act_, real_act_, NUM_MUSCLES * sizeof(double));
    std::memcpy(solve_motor_tau_, real_motor_tau_, NUM_JOINTS * sizeof(double));
    for (int t = 0; t < task_.horizon; ++t) {
        double* ref = &solve_gait_ref_[t * NUM_MUSCLES];
        std::fill(ref, ref + NUM_MUSCLES, 0.0);
        if (phases_.active_gait()) phases_.active_gait()->get_phase(t, ref);
    }

    // Warm-start: shift trajectory_ forward by 1 step (see common/control_utils.h).
    const int stride = task_.horizon * NUM_MUSCLES;
    shift_trajectory(trajectory_, task_.horizon, NUM_MUSCLES);

    sample_noise();

    #pragma omp parallel for schedule(dynamic)
    for (int s = 0; s < task_.n_samples; ++s)
        costs_[s] = rollout(s, state);

    // Softmin weights (normalised); cmin and ESS are diagnostics only.
    std::vector<double> weights;
    const double cmin = softmin_weights(costs_, task_.lambda, weights);

    double w_sq = 0.0;
    for (double w : weights) w_sq += w * w;
    sample_min_ = cmin;
    ess_        = 1.0 / w_sq;

    // Weighted average update.
    std::vector<double> new_traj(stride, 0.0);
    for (int s = 0; s < task_.n_samples; ++s) {
        const double w = weights[s];
        for (int t = 0; t < task_.horizon; ++t)
            for (int m = 0; m < NUM_MUSCLES; ++m) {
                const int idx = t * NUM_MUSCLES + m;
                new_traj[idx] += w * std::clamp(
                    trajectory_[idx] + noise_[s * stride + idx], 0.0, 1.0);
            }
    }
    for (auto& v : new_traj) v = std::clamp(v, 0.0, 1.0);
    trajectory_ = std::move(new_traj);

    // Cost breakdown logging — every 50 updates (0.5 s of sim time at dt = 0.01).
    static constexpr int LOG_INTERVAL = 50;
    if (++log_counter_ % LOG_INTERVAL == 0) {
        const CostTerms c = plan_cost();
        std::printf("[cost] pos=%6.1f  orient=%6.1f  vel=%6.1f  gait=%6.1f  | sample_min=%6.1f  dt=%.1fms\n",
                    c.pos, c.orient, c.vel + c.ang_vel, c.gait, cmin, last_compute_ms_);
    }

    // Output: step 0 of the weighted-mean trajectory, held by actuate() until the next update().
    std::memcpy(last_act_cmd_, trajectory_.data(), NUM_MUSCLES * sizeof(double));

    if (phases_.active_gait()) phases_.active_gait()->advance();

    last_compute_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_start).count();
}

