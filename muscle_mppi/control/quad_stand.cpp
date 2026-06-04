#include "quad_stand.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <stdexcept>
#include <vector>
#include <yaml-cpp/yaml.h>

// ============================================================================
// Constructor
// ============================================================================

QuadStand::QuadStand(const std::string& task_name, const std::string& yaml_path)
    : BaseMPPI(load_task(task_name, yaml_path))
{
    muscle_ = task_.muscle;

    for (int m = 0; m < NUM_MUSCLES; ++m) {
        action_lo_[m] = 0.0;
        action_hi_[m] = 1.0;
    }

    // Find base body.
    for (const char* name : {"trunk", "base", "base_link"}) {
        int bid = mj_name2id(model_, mjOBJ_BODY, name);
        if (bid >= 0) { base_bid_ = bid; break; }
    }
    if (base_bid_ < 0)
        throw std::runtime_error("QuadStand: no base body found in model");

    // Find feet.
    const char* foot_names[] = {"FL_foot", "FR_foot", "RL_foot", "RR_foot"};
    n_feet_ = 0;
    for (int i = 0; i < 4; ++i) {
        int bid = mj_name2id(model_, mjOBJ_BODY, foot_names[i]);
        if (bid >= 0) foot_body_ids_[n_feet_++] = bid;
    }
    if (n_feet_ == 0)
        throw std::runtime_error("QuadStand: no foot bodies found in model");

    // Load cost weights; all default to 0 if absent.
    {
        YAML::Node root = YAML::LoadFile(yaml_path);
        const YAML::Node& c = root[task_name]["cost"];
        cost_.height      = c["height"]      ? c["height"].as<double>()      : 0.0;
        cost_.orientation = c["orientation"] ? c["orientation"].as<double>() : 0.0;
        cost_.posture     = c["posture"]     ? c["posture"].as<double>()     : 0.0;
        cost_.joint_vel   = c["joint_vel"]   ? c["joint_vel"].as<double>()   : 0.0;
        cost_.terminal    = c["terminal"]    ? c["terminal"].as<double>()    : 0.0;
    }
}

// ============================================================================
// Rollout
// ============================================================================

double QuadStand::rollout(int s, const RobotState& state)
{
    mjData* d = data_[s];
    set_mj_state(d, state);

    double activation[NUM_MUSCLES];
    std::memcpy(activation, rollout_act_, NUM_MUSCLES * sizeof(double));

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
            for (int j = 0; j < NUM_JOINTS; ++j)
                d->ctrl[JOINT_OFFSET + j] = tau_out[j];

            mj_step(model_, d);

            if (!std::isfinite(d->qpos[2]) || d->qpos[2] < -1.0)
                return 1e6;
        }

        total_cost += step_cost(d);
    }

    total_cost += terminal_cost(d);
    return std::isfinite(total_cost) ? total_cost : 1e6;
}

// ============================================================================
// Cost — stubs, fill in to define the standing objective
// ============================================================================

double QuadStand::step_cost(const mjData* /*d*/)
{
    return 0.0;
}

double QuadStand::terminal_cost(const mjData* /*d*/)
{
    return 0.0;
}

// ============================================================================
// Solve
// ============================================================================

void QuadStand::update(const RobotState& state, double tau_out[NUM_JOINTS])
{
    if (!state.valid) {
        double act_cmd[NUM_MUSCLES] = {};
        hill_compute_torques(act_cmd, state.q, state.dq, muscle_, task_.dt, real_act_, tau_out);
        return;
    }

    std::memcpy(rollout_act_, real_act_, NUM_MUSCLES * sizeof(double));

    warm_start(1);

    best_cost_ = 1e9;
    const int stride = task_.horizon * NUM_MUSCLES;

    for (int iter = 0; iter < task_.n_iterations; ++iter) {
        sample_noise(iter);

        #pragma omp parallel for schedule(dynamic)
        for (int s = 0; s < task_.n_samples; ++s)
            costs_[s] = rollout(s, state);

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

        // Softmin weights over min-max normalised costs.
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

    double act_cmd[NUM_MUSCLES];
    for (int m = 0; m < NUM_MUSCLES; ++m) act_cmd[m] = best_traj_[m];
    hill_compute_torques(act_cmd, state.q, state.dq, muscle_, task_.dt, real_act_, tau_out);
}
