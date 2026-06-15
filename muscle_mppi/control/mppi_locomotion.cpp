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

    // Find feet.
    const char* foot_names[] = {"FL_foot", "FR_foot", "RL_foot", "RR_foot"};
    n_feet_ = 0;
    for (int i = 0; i < 4; ++i) {
        int bid = mj_name2id(model_, mjOBJ_BODY, foot_names[i]);
        if (bid >= 0) foot_body_ids_[n_feet_++] = bid;
    }
    if (n_feet_ == 0)
        throw std::runtime_error("MPPILocomotion: no foot bodies found in model");

    {
        YAML::Node root = YAML::LoadFile(yaml_path);
        const YAML::Node& c = root[task_name]["cost"];
        cost_.height        = c["height"]        ? c["height"].as<double>()        : 0.0;
        cost_.orientation   = c["orientation"]   ? c["orientation"].as<double>()   : 0.0;
        cost_.posture       = c["posture"]       ? c["posture"].as<double>()       : 0.0;
        cost_.contact_vel   = c["contact_vel"]   ? c["contact_vel"].as<double>()   : 0.0;
        cost_.contact_force = c["contact_force"] ? c["contact_force"].as<double>() : 0.0;
        cost_.terminal      = c["terminal"]      ? c["terminal"].as<double>()      : 0.0;
        if (c["vel_des"]) {
            cmd_.vx = c["vel_des"][0].as<double>();
            cmd_.vy = c["vel_des"][1].as<double>();
        }
    }

    std::copy(task_.posture_bias, task_.posture_bias + NUM_JOINTS, posture_bias_);
    std::copy(task_.posture_FL1,  task_.posture_FL1  + NUM_JOINTS, posture_FL1_);
    std::copy(task_.posture_FL2,  task_.posture_FL2  + NUM_JOINTS, posture_FL2_);
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

        total_cost += step_cost(d, act_cmd);
    }

    total_cost += terminal_cost(d);

    return std::isfinite(total_cost) ? total_cost : 1e6;
}

// ============================================================================
// Cost function
// ============================================================================

double MPPILocomotion::step_cost(const mjData* d, const double act_cmd[NUM_MUSCLES])
{
    const CostWeights& w = cost_;
    double cost = 0.0;

    cost += w.height * std::abs(d->xpos[base_bid_ * 3 + 2] - height_target_);

    const double qw    = d->qpos[3];
    const double angle = 2.0 * std::acos(std::clamp(std::abs(qw), 0.0, 1.0));
    cost += w.orientation * angle * angle;

    if (w.posture > 0.0) {
        for (int j = 0; j < NUM_JOINTS; ++j) {
            const double res = posture_FL1_[j] * act_cmd[2*j]
                             - posture_FL2_[j] * act_cmd[2*j+1]
                             - posture_bias_[j];
            cost += w.posture * res * res;
        }
    }

    // contact_vel: iterate over all active contacts, penalize velocity at each contact point.
    // This matches the paper's "vc contact velocities" — any body touching anything, not just feet.
    if (w.contact_vel > 0.0) {
        for (int ci = 0; ci < d->ncon; ++ci) {
            const mjContact& con = d->contact[ci];
            const int bid1 = model_->geom_bodyid[con.geom1];
            const int bid2 = model_->geom_bodyid[con.geom2];
            // penalize the non-world body (worldbody = 0)
            const int bid = (bid1 != 0) ? bid1 : bid2;
            if (bid == 0) continue;

            mjtNum vel6[6];
            mj_objectVelocity(model_, d, mjOBJ_BODY, bid, vel6, 0);
            // vel6[0..2] = angular, vel6[3..5] = linear at body origin; correct to contact point
            const double dx = con.pos[0] - d->xpos[3*bid + 0];
            const double dy = con.pos[1] - d->xpos[3*bid + 1];
            const double dz = con.pos[2] - d->xpos[3*bid + 2];
            const double vx = vel6[3] + vel6[1]*dz - vel6[2]*dy;
            const double vy = vel6[4] + vel6[2]*dx - vel6[0]*dz;
            const double vz = vel6[5] + vel6[0]*dy - vel6[1]*dx;
            cost += w.contact_vel * (std::abs(vx) + std::abs(vy) + std::abs(vz));
        }
    }

    // contact_force: per-foot aggregate force vs. reference (Eq. 17: ||fc - f0||_1)
    if (w.contact_force > 0.0) {
        for (int fi = 0; fi < n_feet_; ++fi) {
            const int bid = foot_body_ids_[fi];
            const double fx = d->cfrc_ext[6*bid + 3];
            const double fy = d->cfrc_ext[6*bid + 4];
            const double fz = d->cfrc_ext[6*bid + 5];
            cost += w.contact_force * (std::abs(fx - f0_[fi][0])
                                     + std::abs(fy - f0_[fi][1])
                                     + std::abs(fz - f0_[fi][2]));
        }
    }

    return cost;
}

double MPPILocomotion::terminal_cost(const mjData* d)
{
    const double horizon_secs = task_.horizon * task_.substeps * task_.dt;
    const double px_target = start_pos_[0] + cmd_.vx * horizon_secs;
    const double py_target = start_pos_[1] + cmd_.vy * horizon_secs;
    return cost_.terminal * (std::abs(d->qpos[0] - px_target)
                                 + std::abs(d->qpos[1] - py_target));
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

    start_pos_[0] = predicted.pos[0];
    start_pos_[1] = predicted.pos[1];

    // Capture f0 once from the first predicted state as the reference contact forces.
    if (!f0_captured_) {
        const mjData* dp = data_[task_.n_samples];
        for (int fi = 0; fi < n_feet_; ++fi) {
            const int bid = foot_body_ids_[fi];
            f0_[fi][0] = dp->cfrc_ext[6*bid + 3];
            f0_[fi][1] = dp->cfrc_ext[6*bid + 4];
            f0_[fi][2] = dp->cfrc_ext[6*bid + 5];
        }
        f0_captured_ = true;
    }

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

    last_compute_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_start).count();
}

