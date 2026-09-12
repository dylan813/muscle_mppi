#include "mppi_locomotion.h"
#include "../../common/control_utils.h"

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
// set of categorical gaits, each backed by one pre-generated activation-gait TSV
// from the FAST/MED/SLOW library in controllers/muscle/gaits/. A phase selects a gait by name
// (TaskPhase::desired_gait) or, as an escape hatch, an explicit TSV path
// (TaskPhase::gait_path) — see resolve_gait_key() in common/gait.h. Paths
// are repo-relative and resolved with repo_path() at load time.
static const char* GAIT_INPLACE_PATH   = "controllers/muscle/gaits/FAST/activation_gait_FAST_0_0_10cm.tsv";
static const char* GAIT_WALK_PATH      = "controllers/muscle/gaits/MED/activation_gait_MED_0_1_10cm.tsv";
static const char* GAIT_WALK_FAST_PATH = "controllers/muscle/gaits/FAST/activation_gait_FAST_0_1_10cm.tsv";
static const char* GAIT_TROT_PATH      = "controllers/muscle/gaits/MED/activation_gait_MED_0_5_15cm.tsv";

static const NamedGaitPaths kNamedGaits = {
    {"in_place",  GAIT_INPLACE_PATH},
    {"walk",      GAIT_WALK_PATH},
    {"walk_fast", GAIT_WALK_FAST_PATH},
    {"trot",      GAIT_TROT_PATH},
};

// ============================================================================
// Constructor
// ============================================================================

MPPILocomotion::MPPILocomotion(const std::string& task_name, const std::string& yaml_path)
    : BaseMPPI(load_task(task_name, yaml_path))
{
    muscle_ = task_.muscle;

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
        gait_stiffness_ = c["gait_stiffness"] ? c["gait_stiffness"].as<double>() : 0.75;
    }

    // Load the task's gaits and activate phase 0 (see PhaseSequencer::init()),
    // then apply phase 0's noise override against the YAML baseline.
    std::memcpy(base_noise_sigma_act_, task_.noise_sigma_act, sizeof(base_noise_sigma_act_));
    phases_.init(task_.phases, kNamedGaits, task_.height_target);
    apply_phase_noise();

    // Seed trajectory_ and real_act_ with the posture constraint-line midpoint
    // (the co-contraction solution holding the nominal pose against gravity).
    // Only applied when posture geometry is provided (FL1 > 0 for at least one joint).
    {
        bool has_posture = false;
        for (int j = 0; j < NUM_JOINTS; ++j)
            if (task_.posture_FL1[j] > 1e-9) { has_posture = true; break; }

        if (has_posture) {
            double nominal[NUM_MUSCLES] = {};
            for (int j = 0; j < NUM_JOINTS; ++j) {
                const double FL1  = task_.posture_FL1[j];
                const double FL2  = task_.posture_FL2[j];
                const double bias = task_.posture_bias[j];
                const double a2_lo  = (FL2 > 1e-9) ? std::max(0.0, -bias / FL2)        : 0.0;
                const double a2_hi  = (FL2 > 1e-9) ? std::min(1.0, (FL1 - bias) / FL2) : 1.0;
                const double a2_mid = a2_lo + 0.75 * (a2_hi - a2_lo);
                const double a1_mid = std::clamp((bias + FL2 * a2_mid) / FL1, 0.0, 1.0);
                nominal[2 * j]     = a1_mid;
                nominal[2 * j + 1] = a2_mid;
            }
            for (int t = 0; t < task_.horizon; ++t)
                for (int m = 0; m < NUM_MUSCLES; ++m)
                    trajectory_[t * NUM_MUSCLES + m] = nominal[m];
            std::memcpy(real_act_, nominal, NUM_MUSCLES * sizeof(double));
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

    double activation[NUM_MUSCLES];
    std::memcpy(activation, real_act_, NUM_MUSCLES * sizeof(double));

    const int stride  = task_.horizon * NUM_MUSCLES;
    double total_cost = 0.0;

    for (int t = 0; t < task_.horizon; ++t) {
        double act_cmd[NUM_MUSCLES];
        for (int m = 0; m < NUM_MUSCLES; ++m) {
            double noisy = trajectory_[t * NUM_MUSCLES + m]
                         + noise_[s * stride + t * NUM_MUSCLES + m];
            act_cmd[m] = std::clamp(noisy, 0.0, 1.0);
        }

        double tau_out[NUM_JOINTS];
        double q_cur[NUM_JOINTS], dq_cur[NUM_JOINTS];
        for (int j = 0; j < NUM_JOINTS; ++j) {
            q_cur[j]  = d->qpos[act_qpos_adr_[j]];
            dq_cur[j] = d->qvel[act_qvel_adr_[j]];
        }

        hill_compute_torques(act_cmd, q_cur, dq_cur, muscle_, task_.dt, activation, tau_out);

        for (int j = 0; j < model_->nu; ++j) d->ctrl[j] = 0.0;
        for (int j = 0; j < NUM_JOINTS; ++j) d->ctrl[j] = tau_out[j];

        mj_step(model_, d);

        double gait_ref[NUM_MUSCLES] = {};
        if (phases_.active_gait()) phases_.active_gait()->get_phase(t, gait_ref);
        total_cost += step_cost(d, gait_ref);
    }

    return std::isfinite(total_cost) ? total_cost : 1e6;
}

// ============================================================================
// Cost function
// ============================================================================

// Whole-robot CoM position (world frame) and CoM velocity (body-frame axes).
//
// base_bid_ is the trunk (root) body, so its subtree is the entire robot
// (trunk + all legs). d->subtree_com is the mass-weighted CoM of that whole
// subtree and is already computed every step by mj_fwdPosition, so it's
// free. Its velocity (d->subtree_linvel) is a diagnostic quantity MuJoCo
// does NOT compute by default, so we call mj_subtreeVel() to populate it —
// this is an extra O(nbody) pass on top of the regular step, cheap for this
// model but not free. subtree_linvel comes out in world-aligned axes, so we
// rotate it into the body frame the same way the old code rotated qvel.
void MPPILocomotion::base_com_state(mjData* d, double com_pos[3], double com_vel_body[3]) const
{
    mj_subtreeVel(model_, d);

    const double* com = d->subtree_com + base_bid_ * 3;
    com_pos[0] = com[0];
    com_pos[1] = com[1];
    com_pos[2] = com[2];

    world_to_body(d->xmat + base_bid_ * 9, d->subtree_linvel + base_bid_ * 3, com_vel_body);
}

double MPPILocomotion::step_cost(mjData* d, const double gait_ref[NUM_MUSCLES])
{
    const CostWeights& w = cost_;
    double cost = 0.0;

    double pos[3], vel_body[3];
    base_com_state(d, pos, vel_body);

    cost += w.pos_x * std::abs(pos[0] - command().goal_pos[0]);
    cost += w.pos_y * std::abs(pos[1] - command().goal_pos[1]);
    cost += w.pos_z * std::abs(pos[2] - command().goal_pos[2]);

    const double q_dot  = d->qpos[3]*goal_quat_[0] + d->qpos[4]*goal_quat_[1]
                         + d->qpos[5]*goal_quat_[2] + d->qpos[6]*goal_quat_[3];
    const double q_dist = 1.0 - std::abs(q_dot);
    cost += w.orientation * q_dist * q_dist;

    if (w.vel_x > 0.0 || w.vel_y > 0.0 || w.vel_z > 0.0) {
        const double ex = vel_body[0] - command().vx;
        const double ey = vel_body[1] - command().vy;
        cost += w.vel_x * ex*ex + w.vel_y * ey*ey + w.vel_z * vel_body[2]*vel_body[2];
    }

    if (w.ang_vel > 0.0) {
        const double wx = d->qvel[3], wy = d->qvel[4], wz = d->qvel[5];
        cost += w.ang_vel * (wx*wx + wy*wy + wz*wz);
    }

    for (int j = 0; j < NUM_JOINTS; ++j) {
        if (w.gait_ref_weights[j] == 0.0) continue;
        const double q_j   = d->qpos[act_qpos_adr_[j]];
        const double dq_j  = d->qvel[act_qvel_adr_[j]];
        const double tau_j = d->qfrc_bias[act_qvel_adr_[j]];
        double a1_imp, a2_imp;
        hill_invert_torque(q_j, dq_j, tau_j, j, muscle_, gait_stiffness_, a1_imp, a2_imp);
        const double e1 = a1_imp - gait_ref[2 * j];
        const double e2 = a2_imp - gait_ref[2 * j + 1];
        cost += w.gait_ref_weights[j] * (e1*e1 + e2*e2);
    }

    return cost;
}

// ============================================================================
// Main solve
// ============================================================================

void MPPILocomotion::update(const RobotState& state, double tau_out[NUM_JOINTS])
{
    const auto t_start = std::chrono::steady_clock::now();

    if (!state.valid) {
        double act_cmd[NUM_MUSCLES] = {};
        hill_compute_torques(act_cmd, state.q, state.dq, muscle_, task_.dt, real_act_, tau_out);
        return;
    }

    // Goal-facing orientation target for this tick's cost, held fixed across
    // the whole rollout batch below (see common/control_utils.h).
    goal_facing_quat(command().goal_pos, state.pos, phases_.dwelling(), goal_quat_);

    // Warm-start: shift trajectory_ forward by 1 step (see common/control_utils.h).
    const int stride = task_.horizon * NUM_MUSCLES;
    shift_trajectory(trajectory_, task_.horizon, NUM_MUSCLES);

    sample_noise();

    #pragma omp parallel for schedule(dynamic)
    for (int s = 0; s < task_.n_samples; ++s)
        costs_[s] = rollout(s, state);

    // Softmin weights (normalised); cmin logged below as a diagnostic only.
    std::vector<double> weights;
    const double cmin = softmin_weights(costs_, task_.lambda, weights);

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
        mjData* dl = data_[task_.n_samples];
        set_mj_state(dl, state);
        double act_log[NUM_MUSCLES];
        std::memcpy(act_log, real_act_, NUM_MUSCLES * sizeof(double));

        double c_pos = 0, c_orient = 0, c_vel = 0, c_gait = 0;

        for (int t = 0; t < task_.horizon; ++t) {
            double cmd[NUM_MUSCLES];
            for (int m = 0; m < NUM_MUSCLES; ++m) cmd[m] = trajectory_[t * NUM_MUSCLES + m];

            double q_l[NUM_JOINTS], dq_l[NUM_JOINTS];
            for (int j = 0; j < NUM_JOINTS; ++j) {
                q_l[j]  = dl->qpos[act_qpos_adr_[j]];
                dq_l[j] = dl->qvel[act_qvel_adr_[j]];
            }
            double tau_l[NUM_JOINTS];
            hill_compute_torques(cmd, q_l, dq_l, muscle_, task_.dt, act_log, tau_l);
            for (int j = 0; j < model_->nu; ++j) dl->ctrl[j] = 0.0;
            for (int j = 0; j < NUM_JOINTS; ++j) dl->ctrl[j] = tau_l[j];
            mj_step(model_, dl);

            double lpos[3], lvel[3];
            base_com_state(dl, lpos, lvel);
            c_pos += cost_.pos_x * std::abs(lpos[0] - command().goal_pos[0]);
            c_pos += cost_.pos_y * std::abs(lpos[1] - command().goal_pos[1]);
            c_pos += cost_.pos_z * std::abs(lpos[2] - command().goal_pos[2]);

            const double q_dot_l  = dl->qpos[3]*goal_quat_[0] + dl->qpos[4]*goal_quat_[1]
                                   + dl->qpos[5]*goal_quat_[2] + dl->qpos[6]*goal_quat_[3];
            const double q_dist = 1.0 - std::abs(q_dot_l);
            c_orient += cost_.orientation * q_dist * q_dist;

            if (cost_.vel_x > 0.0 || cost_.vel_y > 0.0 || cost_.vel_z > 0.0) {
                c_vel += cost_.vel_x * (lvel[0] - command().vx)*(lvel[0] - command().vx)
                       + cost_.vel_y * (lvel[1] - command().vy)*(lvel[1] - command().vy)
                       + cost_.vel_z * lvel[2]*lvel[2];
            }

            if (cost_.ang_vel > 0.0) {
                const double wx = dl->qvel[3], wy = dl->qvel[4], wz = dl->qvel[5];
                c_vel += cost_.ang_vel * (wx*wx + wy*wy + wz*wz);
            }

            if (phases_.active_gait()) {
                double gref[NUM_MUSCLES] = {};
                phases_.active_gait()->get_phase(t, gref);
                for (int j = 0; j < NUM_JOINTS; ++j) {
                    if (cost_.gait_ref_weights[j] == 0.0) continue;
                    const double q_j   = dl->qpos[act_qpos_adr_[j]];
                    const double dq_j  = dl->qvel[act_qvel_adr_[j]];
                    const double tau_j = dl->qfrc_bias[act_qvel_adr_[j]];
                    double a1_imp, a2_imp;
                    hill_invert_torque(q_j, dq_j, tau_j, j, muscle_, gait_stiffness_, a1_imp, a2_imp);
                    const double e1 = a1_imp - gref[2 * j];
                    const double e2 = a2_imp - gref[2 * j + 1];
                    c_gait += cost_.gait_ref_weights[j] * (e1*e1 + e2*e2);
                }
            }
        }

        std::printf("[cost] pos=%6.1f  orient=%6.1f  vel=%6.1f  gait=%6.1f  | sample_min=%6.1f  dt=%.1fms\n",
                    c_pos, c_orient, c_vel, c_gait, cmin, last_compute_ms_);
    }

    // Output: execute step 0 of the weighted-mean trajectory from current state.
    double act_cmd[NUM_MUSCLES];
    for (int m = 0; m < NUM_MUSCLES; ++m) act_cmd[m] = trajectory_[m];
    hill_compute_torques(act_cmd, state.q, state.dq, muscle_, task_.dt, real_act_, tau_out);

    if (phases_.active_gait()) phases_.active_gait()->advance();

    last_compute_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_start).count();
}

