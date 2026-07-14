#include "mppi_locomotion.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

// ============================================================================
// Constructor
// ============================================================================

MPPILocomotion::MPPILocomotion(const std::string& task_name, const std::string& yaml_path)
    : BaseMPPI(load_task(task_name, yaml_path))
{
    muscle_ = task_.muscle;

    // BaseMPPI already initializes trajectory_, noise_, costs_
    // to the correct sizes (horizon × NUM_MUSCLES). Only action bounds need setting here.
    for (int m = 0; m < NUM_MUSCLES; ++m) {
        action_lo_[m] = 0.0;
        action_hi_[m] = 1.0;
    }

    // Find base body.
    for (const char* name : {"trunk", "base", "base_link"}) {
        int bid = mj_name2id(model_, mjOBJ_BODY, name);
        if (bid >= 0) { base_bid_ = bid; break; }
    }

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
        gait_stiffness_   = c["gait_stiffness"] ? c["gait_stiffness"].as<double>() : 0.75;
        if (c["vel_des"]) {
            cmd_.vx = c["vel_des"][0].as<double>();
            cmd_.vy = c["vel_des"][1].as<double>();
        }
        cmd_.goal_pos[2] = task_.height_target;  // z default; overridden below if set in YAML
        if (c["goal_pos"]) {
            cmd_.goal_pos[0] = c["goal_pos"][0].as<double>();
            cmd_.goal_pos[1] = c["goal_pos"][1].as<double>();
            cmd_.goal_pos[2] = c["goal_pos"][2].as<double>();
        }
    }

    if (!task_.gait_path.empty())
        gait_sched_.load(task_.gait_path);

    // Seed trajectory_, real_act_, predicted_activation_ with the constraint-line
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
        for (int j = 0; j < NUM_JOINTS; ++j) d->ctrl[JOINT_OFFSET + j] = tau_out[j];

        mj_step(model_, d);

        double gait_ref[NUM_MUSCLES] = {};
        if (gait_sched_.loaded()) gait_sched_.get_phase(t, gait_ref);
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

    const double* v_world = d->subtree_linvel + base_bid_ * 3;
    const double* xmat    = d->xmat + base_bid_ * 9;
    com_vel_body[0] = v_world[0]*xmat[0] + v_world[1]*xmat[3] + v_world[2]*xmat[6];
    com_vel_body[1] = v_world[0]*xmat[1] + v_world[1]*xmat[4] + v_world[2]*xmat[7];
    com_vel_body[2] = v_world[0]*xmat[2] + v_world[1]*xmat[5] + v_world[2]*xmat[8];
}

double MPPILocomotion::step_cost(mjData* d, const double gait_ref[NUM_MUSCLES])
{
    const CostWeights& w = cost_;
    double cost = 0.0;

    double pos[3], vel_body[3];
    base_com_state(d, pos, vel_body);

    cost += w.pos_x * std::abs(pos[0] - cmd_.goal_pos[0]);
    cost += w.pos_y * std::abs(pos[1] - cmd_.goal_pos[1]);
    cost += w.pos_z * std::abs(pos[2] - cmd_.goal_pos[2]);

    const double qw     = d->qpos[3];
    const double q_dist = 1.0 - std::abs(qw);
    cost += w.orientation * q_dist * q_dist;

    if (w.vel_x > 0.0 || w.vel_y > 0.0 || w.vel_z > 0.0) {
        const double ex = vel_body[0] - cmd_.vx;
        const double ey = vel_body[1] - cmd_.vy;
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

    // Warm-start: shift trajectory_ forward by 1 step (mirrors RTWholeBodyMPPI).
    const int stride = task_.horizon * NUM_MUSCLES;
    std::vector<double> shifted(stride);
    for (int t = 0; t < task_.horizon - 1; ++t)
        for (int m = 0; m < NUM_MUSCLES; ++m)
            shifted[t * NUM_MUSCLES + m] = trajectory_[(t + 1) * NUM_MUSCLES + m];
    for (int m = 0; m < NUM_MUSCLES; ++m)
        shifted[(task_.horizon - 1) * NUM_MUSCLES + m] =
            trajectory_[(task_.horizon - 1) * NUM_MUSCLES + m];
    trajectory_ = shifted;

    sample_noise();

    #pragma omp parallel for schedule(dynamic)
    for (int s = 0; s < task_.n_samples; ++s)
        costs_[s] = rollout(s, state);

    // Softmin weights.
    double cmin   = *std::min_element(costs_.begin(), costs_.end());
    double cmax   = *std::max_element(costs_.begin(), costs_.end());
    double crange = cmax - cmin;  // cmin logged below as a diagnostic only

    std::vector<double> weights(task_.n_samples);
    double wsum = 0.0;
    for (int s = 0; s < task_.n_samples; ++s) {
        double s_hat = (crange > 1e-12) ? (costs_[s] - cmin) / crange : 0.0;
        weights[s]   = std::exp(-s_hat / task_.lambda);
        wsum        += weights[s];
    }

    // Weighted average update.
    std::vector<double> new_traj(stride, 0.0);
    for (int s = 0; s < task_.n_samples; ++s) {
        const double w = weights[s] / wsum;
        for (int t = 0; t < task_.horizon; ++t)
            for (int m = 0; m < NUM_MUSCLES; ++m) {
                const int idx = t * NUM_MUSCLES + m;
                new_traj[idx] += w * std::clamp(
                    trajectory_[idx] + noise_[s * stride + idx], 0.0, 1.0);
            }
    }
    for (auto& v : new_traj) v = std::clamp(v, 0.0, 1.0);
    trajectory_ = std::move(new_traj);

    // Cost breakdown logging — runs once per second (every 50 updates at 50 Hz).
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
            for (int j = 0; j < NUM_JOINTS; ++j) dl->ctrl[JOINT_OFFSET + j] = tau_l[j];
            mj_step(model_, dl);

            double lpos[3], lvel[3];
            base_com_state(dl, lpos, lvel);
            c_pos += cost_.pos_x * std::abs(lpos[0] - cmd_.goal_pos[0]);
            c_pos += cost_.pos_y * std::abs(lpos[1] - cmd_.goal_pos[1]);
            c_pos += cost_.pos_z * std::abs(lpos[2] - cmd_.goal_pos[2]);

            const double qw     = dl->qpos[3];
            const double q_dist = 1.0 - std::abs(qw);
            c_orient += cost_.orientation * q_dist * q_dist;

            if (cost_.vel_x > 0.0 || cost_.vel_y > 0.0 || cost_.vel_z > 0.0) {
                c_vel += cost_.vel_x * (lvel[0] - cmd_.vx)*(lvel[0] - cmd_.vx)
                       + cost_.vel_y * (lvel[1] - cmd_.vy)*(lvel[1] - cmd_.vy)
                       + cost_.vel_z * lvel[2]*lvel[2];
            }

            if (cost_.ang_vel > 0.0) {
                const double wx = dl->qvel[3], wy = dl->qvel[4], wz = dl->qvel[5];
                c_vel += cost_.ang_vel * (wx*wx + wy*wy + wz*wz);
            }

            if (gait_sched_.loaded()) {
                double gref[NUM_MUSCLES] = {};
                gait_sched_.get_phase(t, gref);
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

    gait_sched_.advance();

    last_compute_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_start).count();
}

