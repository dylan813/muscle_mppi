#include "mppi_locomotion.h"

#include <cmath>
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

    // BaseMPPI already initializes trajectory_, best_traj_, noise_, costs_
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
        cost_.height        = c["height"]        ? c["height"].as<double>()        : 0.0;
        cost_.orientation   = c["orientation"]   ? c["orientation"].as<double>()   : 0.0;
        cost_.vel           = c["vel"]           ? c["vel"].as<double>()           : 0.0;
        if (c["vel_des"]) {
            cmd_.vx = c["vel_des"][0].as<double>();
            cmd_.vy = c["vel_des"][1].as<double>();
        }
    }

    if (!task_.gait_path.empty())
        gait_sched_.load(task_.gait_path);

    // Seed trajectory_, best_traj_, real_act_, predicted_activation_ with the
    // constraint-line midpoint activations — same role as RTWholeBodyMPPI's sampling_init.
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
                const double a2_mid = 0.5 * (a2_lo + a2_hi);
                const double a1_mid = std::clamp((bias + FL2 * a2_mid) / FL1, 0.0, 1.0);
                nominal[2 * j]     = a1_mid;
                nominal[2 * j + 1] = a2_mid;
            }
            for (int t = 0; t < task_.horizon; ++t)
                for (int m = 0; m < NUM_MUSCLES; ++m) {
                    trajectory_[t * NUM_MUSCLES + m] = nominal[m];
                    best_traj_[t * NUM_MUSCLES + m]  = nominal[m];
                }
            std::memcpy(real_act_,             nominal, NUM_MUSCLES * sizeof(double));
            std::memcpy(predicted_activation_, nominal, NUM_MUSCLES * sizeof(double));
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
    std::memcpy(activation, predicted_activation_, NUM_MUSCLES * sizeof(double));

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
        for (int sub = 0; sub < task_.substeps; ++sub) {
            double q_cur[NUM_JOINTS], dq_cur[NUM_JOINTS];
            for (int j = 0; j < NUM_JOINTS; ++j) {
                q_cur[j]  = d->qpos[act_qpos_adr_[j]];
                dq_cur[j] = d->qvel[act_qvel_adr_[j]];
            }

            hill_compute_torques(act_cmd, q_cur, dq_cur, muscle_, task_.dt, activation, tau_out);

            for (int j = 0; j < model_->nu; ++j) d->ctrl[j] = 0.0;
            for (int j = 0; j < NUM_JOINTS; ++j) d->ctrl[JOINT_OFFSET + j] = tau_out[j];

            mj_step(model_, d);

            if (!std::isfinite(d->qpos[2]) || d->qpos[2] < -1.0)
                return 1e6;
        }

        double gait_ref[NUM_MUSCLES] = {};
        if (gait_sched_.loaded()) gait_sched_.get_phase(t, gait_ref);
        total_cost += step_cost(d, act_cmd, gait_ref);
    }

    return std::isfinite(total_cost) ? total_cost : 1e6;
}

// ============================================================================
// Cost function
// ============================================================================

double MPPILocomotion::step_cost(const mjData* d, const double act_cmd[NUM_MUSCLES],
                                  const double gait_ref[NUM_MUSCLES])
{
    const CostWeights& w = cost_;
    double cost = 0.0;

    cost += w.height * std::abs(d->xpos[base_bid_ * 3 + 2] - height_target_);

    const double qw    = d->qpos[3];
    const double angle = 2.0 * std::acos(std::clamp(std::abs(qw), 0.0, 1.0));
    cost += w.orientation * angle * angle;

    if (w.vel > 0.0) {
        // Project world-frame CoM velocity onto body x and y axes (R^T * v_world).
        // xmat is row-major: columns are body axes expressed in world frame.
        const double* xmat = d->xmat + base_bid_ * 9;
        const double vx_body = d->qvel[0]*xmat[0] + d->qvel[1]*xmat[3] + d->qvel[2]*xmat[6];
        const double vy_body = d->qvel[0]*xmat[1] + d->qvel[1]*xmat[4] + d->qvel[2]*xmat[7];
        const double ex = vx_body - cmd_.vx;
        const double ey = vy_body - cmd_.vy;
        cost += w.vel * (ex*ex + ey*ey);
    }

    // contact_vel: iterate over all active contacts, penalize velocity at each contact point.
    // gait_ref: L1 deviation of activation commands from the cyclic gait reference.
    if (w.gait_ref > 0.0) {
        for (int m = 0; m < NUM_MUSCLES; ++m) {
            const double e = act_cmd[m] - gait_ref[m];
            cost += w.gait_ref * e * e;
        }
    }

    return cost;
}

// ============================================================================
// Predict state (latency compensation)
// ============================================================================

RobotState MPPILocomotion::predict_state(const RobotState& state, int n_steps)
{
    mjData* d = data_[task_.n_samples];
    set_mj_state(d, state);

    double activation[NUM_MUSCLES];
    std::memcpy(activation, real_act_, NUM_MUSCLES * sizeof(double));

    for (int t = 0; t < n_steps; ++t) {
        double act_cmd[NUM_MUSCLES];
        for (int m = 0; m < NUM_MUSCLES; ++m)
            act_cmd[m] = best_traj_[t * NUM_MUSCLES + m];

        double tau_out[NUM_JOINTS];
        for (int sub = 0; sub < task_.substeps; ++sub) {
            double q_cur[NUM_JOINTS], dq_cur[NUM_JOINTS];
            for (int j = 0; j < NUM_JOINTS; ++j) {
                q_cur[j]  = d->qpos[act_qpos_adr_[j]];
                dq_cur[j] = d->qvel[act_qvel_adr_[j]];
            }
            hill_compute_torques(act_cmd, q_cur, dq_cur, muscle_, task_.dt, activation, tau_out);
            for (int j = 0; j < model_->nu; ++j) d->ctrl[j] = 0.0;
            for (int j = 0; j < NUM_JOINTS; ++j) d->ctrl[JOINT_OFFSET + j] = tau_out[j];
            mj_step(model_, d);
        }
    }

    RobotState predicted;
    if (has_freejoint_) {
        predicted.pos[0]  = d->qpos[0]; predicted.pos[1]  = d->qpos[1]; predicted.pos[2]  = d->qpos[2];
        predicted.quat[0] = d->qpos[3]; predicted.quat[1] = d->qpos[4];
        predicted.quat[2] = d->qpos[5]; predicted.quat[3] = d->qpos[6];
        predicted.vel[0]  = d->qvel[0]; predicted.vel[1]  = d->qvel[1]; predicted.vel[2]  = d->qvel[2];
        predicted.gyro[0] = d->qvel[3]; predicted.gyro[1] = d->qvel[4]; predicted.gyro[2] = d->qvel[5];
    }
    for (int j = 0; j < NUM_JOINTS; ++j) {
        predicted.q[j]  = d->qpos[act_qpos_adr_[j]];
        predicted.dq[j] = d->qvel[act_qvel_adr_[j]];
    }
    predicted.valid = true;
    std::memcpy(predicted_activation_, activation, NUM_MUSCLES * sizeof(double));
    return predicted;
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

    const double dt_step = task_.substeps * task_.dt;
    const int n_skip = std::clamp(
        static_cast<int>(std::round(last_compute_ms_ * 1e-3 / dt_step)),
        1, task_.horizon / 2);

    RobotState predicted = predict_state(state, n_skip);

    // Warm-start: shift best_traj_ forward by n_skip steps.
    const int skip   = std::min(n_skip, task_.horizon - 1);
    const int stride = task_.horizon * NUM_MUSCLES;
    std::vector<double> shifted(stride);
    for (int t = 0; t < task_.horizon - skip; ++t)
        for (int m = 0; m < NUM_MUSCLES; ++m)
            shifted[t * NUM_MUSCLES + m] = best_traj_[(t + skip) * NUM_MUSCLES + m];
    for (int t = task_.horizon - skip; t < task_.horizon; ++t)
        for (int m = 0; m < NUM_MUSCLES; ++m)
            shifted[t * NUM_MUSCLES + m] = best_traj_[(task_.horizon - 1) * NUM_MUSCLES + m];
    trajectory_ = shifted;

    const int N = task_.n_iterations;
    best_cost_ = 1e9;

    for (int iter = 0; iter < N; ++iter) {
        sample_noise(iter);

        #pragma omp parallel for schedule(dynamic)
        for (int s = 0; s < task_.n_samples; ++s)
            costs_[s] = rollout(s, predicted);

        // Track best sample.
        for (int s = 0; s < task_.n_samples; ++s) {
            if (costs_[s] < best_cost_) {
                best_cost_ = costs_[s];
                for (int t = 0; t < task_.horizon; ++t)
                    for (int m = 0; m < NUM_MUSCLES; ++m) {
                        const int idx = t * NUM_MUSCLES + m;
                        best_traj_[idx] = std::clamp(
                            trajectory_[idx] + noise_[s * stride + idx], 0.0, 1.0);
                    }
            }
        }

        // Softmin weights.
        double cmin   = *std::min_element(costs_.begin(), costs_.end());
        double cmax   = *std::max_element(costs_.begin(), costs_.end());
        double crange = cmax - cmin;

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
    }

    // Output: execute step 0 of best trajectory from predicted state.
    double act_cmd[NUM_MUSCLES];
    for (int m = 0; m < NUM_MUSCLES; ++m) act_cmd[m] = best_traj_[m];
    hill_compute_torques(act_cmd, predicted.q, predicted.dq, muscle_, task_.dt,
                         predicted_activation_, tau_out);
    std::memcpy(real_act_, predicted_activation_, NUM_MUSCLES * sizeof(double));

    gait_sched_.advance();

    last_compute_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_start).count();
}

