#include "single_leg_torque.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <stdexcept>
#include <iostream>

SingleLegTorque::SingleLegTorque(const std::string& task_name,
                                   const std::string& yaml_path)
    : BaseMPPI(load_task(task_name, yaml_path))
{
    best_traj_.assign(task_.horizon * NUM_JOINTS, 0.0);

    foot_body_id_ = mj_name2id(model_, mjOBJ_BODY, "FL_foot");
    if (foot_body_id_ < 0)
        throw std::runtime_error("Body not found: FL_foot");
}

// -----------------------------------------------------------------------------
// Cost
// -----------------------------------------------------------------------------

double SingleLegTorque::step_cost(const mjData* d,
                                   const double tau_cmd[NUM_JOINTS],
                                   int /*horizon_step*/)
{
    double cost = 0.0;

    // Foot position tracking error.
    const double* fp = d->xpos + 3 * foot_body_id_;
    double dx = fp[0] - task_.foot_target[0];
    double dy = fp[1] - task_.foot_target[1];
    double dz = fp[2] - task_.foot_target[2];
    cost += task_.cost.foot_pos * (dx*dx + dy*dy + dz*dz);

    // Joint velocity — penalise oscillation and overshoot.
    if (task_.cost.joint_vel > 0.0) {
        for (int j = 0; j < NUM_JOINTS; ++j) {
            double v = d->qvel[act_qvel_adr_[j]];
            cost += task_.cost.joint_vel * v * v;
        }
    }

    // Torque effort — keep commands small when not needed.
    if (task_.cost.act_effort > 0.0) {
        for (int j = 0; j < NUM_JOINTS; ++j)
            cost += task_.cost.act_effort * tau_cmd[j] * tau_cmd[j];
    }

    return cost;
}

double SingleLegTorque::terminal_cost(const mjData* d)
{
    const double* fp = d->xpos + 3 * foot_body_id_;
    double dx = fp[0] - task_.foot_target[0];
    double dy = fp[1] - task_.foot_target[1];
    double dz = fp[2] - task_.foot_target[2];
    return task_.cost.terminal * (dx*dx + dy*dy + dz*dz);
}

// -----------------------------------------------------------------------------
// Rollout
// -----------------------------------------------------------------------------

double SingleLegTorque::rollout(int s, const RobotState& state)
{
    mjData* d = data_[s];
    set_mj_state(d, state);

    double total_cost = 0.0;

    for (int t = 0; t < task_.horizon; ++t) {
        double tau_cmd[NUM_JOINTS];
        for (int j = 0; j < NUM_JOINTS; ++j) {
            double noisy = trajectory_[t * NUM_JOINTS + j]
                         + noise_[s * task_.horizon * NUM_JOINTS + t * NUM_JOINTS + j];
            tau_cmd[j] = std::clamp(noisy, -task_.tau_max[j], task_.tau_max[j]);
        }

        for (int j = 0; j < model_->nu; ++j) d->ctrl[j] = 0.0;
        for (int j = 0; j < NUM_JOINTS; ++j)
            d->ctrl[JOINT_OFFSET + j] = tau_cmd[j];

        mj_step(model_, d);

        // Fixed-base scene: no freejoint, so only check for NaN divergence.
        if (!std::isfinite(d->qpos[act_qpos_adr_[0]]))
            return 1e6;

        total_cost += step_cost(d, tau_cmd, t);
    }

    total_cost += terminal_cost(d);
    return std::isfinite(total_cost) ? total_cost : 1e6;
}

// -----------------------------------------------------------------------------
// Solve
// -----------------------------------------------------------------------------

void SingleLegTorque::update(const RobotState& state, double tau_out[NUM_JOINTS])
{
    if (!state.valid) {
        for (int j = 0; j < NUM_JOINTS; ++j) tau_out[j] = best_traj_[j];
        return;
    }

    // Warm-start: shift trajectory forward one step, hold tail.
    for (int t = 0; t < task_.horizon - 1; ++t)
        for (int j = 0; j < NUM_JOINTS; ++j)
            trajectory_[t * NUM_JOINTS + j] = best_traj_[(t + 1) * NUM_JOINTS + j];
    for (int j = 0; j < NUM_JOINTS; ++j)
        trajectory_[(task_.horizon - 1) * NUM_JOINTS + j] =
            best_traj_[(task_.horizon - 1) * NUM_JOINTS + j];

    best_cost_ = 1e9;

    for (int iter = 0; iter < task_.n_iterations; ++iter) {
        sample_noise(iter, task_.n_iterations);

        #pragma omp parallel for schedule(dynamic)
        for (int s = 0; s < task_.n_samples; ++s)
            costs_[s] = rollout(s, state);

        // Track best sample.
        for (int s = 0; s < task_.n_samples; ++s) {
            if (costs_[s] < best_cost_) {
                best_cost_ = costs_[s];
                for (int t = 0; t < task_.horizon; ++t)
                    for (int j = 0; j < NUM_JOINTS; ++j) {
                        int idx = t * NUM_JOINTS + j;
                        best_traj_[idx] = std::clamp(
                            trajectory_[idx]
                                + noise_[s * task_.horizon * NUM_JOINTS + idx],
                            -task_.tau_max[j], task_.tau_max[j]);
                    }
            }
        }

        // Softmin weights over min-max normalised costs.
        double cmin = *std::min_element(costs_.begin(), costs_.end());
        double cmax = *std::max_element(costs_.begin(), costs_.end());
        double crange = cmax - cmin;

        std::vector<double> weights(task_.n_samples);
        double wsum = 0.0;
        for (int s = 0; s < task_.n_samples; ++s) {
            double s_hat = (crange > 1e-12) ? (costs_[s] - cmin) / crange : 0.0;
            weights[s] = std::exp(-s_hat / task_.lambda);
            wsum += weights[s];
        }

        std::vector<double> new_traj(task_.horizon * NUM_JOINTS, 0.0);
        for (int s = 0; s < task_.n_samples; ++s) {
            double w = weights[s] / wsum;
            for (int t = 0; t < task_.horizon; ++t)
                for (int j = 0; j < NUM_JOINTS; ++j) {
                    int idx = t * NUM_JOINTS + j;
                    new_traj[idx] += w * (trajectory_[idx]
                        + noise_[s * task_.horizon * NUM_JOINTS + idx]);
                }
        }
        for (int t = 0; t < task_.horizon; ++t)
            for (int j = 0; j < NUM_JOINTS; ++j) {
                int idx = t * NUM_JOINTS + j;
                new_traj[idx] = std::clamp(new_traj[idx],
                                           -task_.tau_max[j], task_.tau_max[j]);
            }

        trajectory_ = std::move(new_traj);
    }

    for (int j = 0; j < NUM_JOINTS; ++j)
        tau_out[j] = best_traj_[j];
}
