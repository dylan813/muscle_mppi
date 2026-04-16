#include "mppi_locomotion.h"
#include "../utils/tasks.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <future>

MPPILocomotion::MPPILocomotion(const std::string& task_name)
    : BaseMPPI(get_task(task_name))
{
    muscle_ = task_.muscle;
    std::copy(task_.nominal_pose, task_.nominal_pose + NUM_JOINTS, joint_ref_);

    // -----------------------------------------------------------------------
    // Warm-start: compute gravity-compensation activation at nominal pose.
    // MuJoCo qfrc_bias[6+j] is the gravity torque at joint j when dq=0.
    // act_init = tau_grav / tau_max gives the activation to hold the pose.
    // This prevents rollouts from all collapsing immediately (zero activation
    // = zero torque = robot in free-fall on every sample).
    // -----------------------------------------------------------------------
    mjData* d0 = mj_makeData(model_);
    mj_resetData(model_, d0);
    d0->qpos[2] = task_.height_target;
    d0->qpos[3] = 1.0;  // upright (quat w=1)
    for (int j = 0; j < NUM_JOINTS; ++j)
        d0->qpos[7 + j] = task_.nominal_pose[j];
    mj_forward(model_, d0);

    for (int t = 0; t < task_.horizon; ++t) {
        for (int j = 0; j < NUM_JOINTS; ++j) {
            double tau_grav = d0->qfrc_bias[6 + j];
            double act_init = std::clamp(tau_grav / muscle_.tau_max[j], ACT_MIN, ACT_MAX);
            trajectory_[t * NUM_JOINTS + j] = act_init;
        }
    }
    mj_deleteData(d0);
}

// -----------------------------------------------------------------------
// Cost — based on simulated state, not on action (no joint_track of action)
// -----------------------------------------------------------------------
double MPPILocomotion::step_cost(const mjData* d) {
    const CostWeights& w = task_.cost;
    double cost = 0.0;

    double hz = d->qpos[2] - height_target_;
    cost += w.height * hz * hz;

    double tilt = 1.0 - d->qpos[3];
    cost += w.orientation * tilt * tilt;

    for (int j = 0; j < 3; ++j)
        cost += w.lin_vel * d->qvel[j] * d->qvel[j];

    for (int j = 3; j < 6; ++j)
        cost += w.ang_vel * d->qvel[j] * d->qvel[j];

    // Penalise actual joint positions deviating from reference
    for (int j = 0; j < NUM_LEG_JOINTS; ++j) {
        double dq = d->qpos[7 + j] - joint_ref_[j];
        cost += w.joint_deviation * dq * dq;
    }

    return cost;
}

double MPPILocomotion::terminal_cost(const mjData* d) {
    const CostWeights& w = task_.cost;
    double hz   = d->qpos[2] - height_target_;
    double tilt = 1.0 - d->qpos[3];
    double cost = w.height * hz * hz + w.orientation * tilt * tilt;
    for (int j = 0; j < NUM_LEG_JOINTS; ++j) {
        double dq = d->qpos[7 + j] - joint_ref_[j];
        cost += w.terminal_joint * dq * dq;
    }
    return cost;
}

// -----------------------------------------------------------------------
// Rollout — pure muscle model, no PD
// -----------------------------------------------------------------------
double MPPILocomotion::rollout(int s, const RobotState& state) {
    mjData* d = data_[s];
    set_mj_state(d, state);

    double activation[NUM_JOINTS];
    std::memcpy(activation, muscle_state_.activation, sizeof(activation));

    double total_cost = 0.0;

    for (int t = 0; t < task_.horizon; ++t) {
        // Perturbed activation command, clamped to [-1, 1]
        double act_cmd[NUM_JOINTS];
        for (int j = 0; j < NUM_JOINTS; ++j) {
            double nom   = trajectory_[t * NUM_JOINTS + j];
            double noisy = nom + noise_[s * task_.horizon * NUM_JOINTS + t * NUM_JOINTS + j];
            act_cmd[j] = std::clamp(noisy, ACT_MIN, ACT_MAX);
        }

        // Hold act_cmd constant over all substeps (one command per control step).
        // hill_compute_torques updates activation[] and fatigue[] per substep.
        double tau_out[NUM_JOINTS];
        for (int sub = 0; sub < task_.substeps; ++sub) {
            double dq_cur[NUM_JOINTS];
            for (int j = 0; j < NUM_JOINTS; ++j)
                dq_cur[j] = d->qvel[6 + j];

            hill_compute_torques(act_cmd, dq_cur, muscle_, activation, tau_out);

            for (int j = 0; j < NUM_JOINTS; ++j)
                d->ctrl[j] = tau_out[j];

            mj_step(model_, d);

            if (!std::isfinite(d->qpos[2]) || d->qpos[2] < -1.0)
                return 1e6;
        }

        total_cost += step_cost(d);
    }

    total_cost += terminal_cost(d);
    return std::isfinite(total_cost) ? total_cost : 1e6;
}

// -----------------------------------------------------------------------
// MPPI update — trajectory is activation commands ∈ [-1, 1]
// -----------------------------------------------------------------------
void MPPILocomotion::update(const RobotState& state, double activations_out[NUM_JOINTS]) {
    if (!state.valid) {
        for (int j = 0; j < NUM_JOINTS; ++j)
            activations_out[j] = trajectory_[j];
        return;
    }

    sample_noise();

    std::vector<std::future<double>> futures;
    futures.reserve(task_.n_samples);
    for (int s = 0; s < task_.n_samples; ++s)
        futures.push_back(std::async(std::launch::async,
            [this, s, &state]() { return rollout(s, state); }));

    for (int s = 0; s < task_.n_samples; ++s)
        costs_[s] = futures[s].get();

    double cost_min = *std::min_element(costs_.begin(), costs_.end());
    std::vector<double> weights(task_.n_samples);
    double weight_sum = 0.0;
    for (int s = 0; s < task_.n_samples; ++s) {
        weights[s]  = std::exp(-1.0 / task_.lambda * (costs_[s] - cost_min));
        weight_sum += weights[s];
    }

    std::vector<double> new_traj(task_.horizon * NUM_JOINTS, 0.0);
    for (int s = 0; s < task_.n_samples; ++s) {
        double w = weights[s] / weight_sum;
        for (int t = 0; t < task_.horizon; ++t)
            for (int j = 0; j < NUM_JOINTS; ++j) {
                int idx = t * NUM_JOINTS + j;
                new_traj[idx] += w * (trajectory_[idx]
                    + noise_[s * task_.horizon * NUM_JOINTS + idx]);
            }
    }

    for (auto& v : new_traj)
        v = std::clamp(v, ACT_MIN, ACT_MAX);

    for (int j = 0; j < NUM_JOINTS; ++j)
        activations_out[j] = new_traj[j];

    std::copy(new_traj.begin() + NUM_JOINTS, new_traj.end(), trajectory_.begin());
    std::copy(new_traj.end() - NUM_JOINTS,   new_traj.end(), trajectory_.end() - NUM_JOINTS);
}

// -----------------------------------------------------------------------
// Hill model for the real robot — advances persistent muscle_state_
// -----------------------------------------------------------------------
void MPPILocomotion::compute_real_torques(const RobotState& state,
                                          const double activations[NUM_JOINTS],
                                          double tau_out[NUM_JOINTS])
{
    const double ctrl_dt = task_.dt * task_.substeps;
    hill_compute_torques(activations, state.dq, muscle_,
                         muscle_state_.activation, tau_out);
}
