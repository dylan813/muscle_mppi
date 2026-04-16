#include "mppi_locomotion.h"
#include "../utils/tasks.h"

#include <cmath>
#include <algorithm>
#include <future>

MPPILocomotion::MPPILocomotion(const std::string& task_name)
    : BaseMPPI(get_task(task_name))
{
    // Warm-start all horizon steps at the nominal pose
    for (int t = 0; t < task_.horizon; ++t)
        for (int j = 0; j < NUM_JOINTS; ++j)
            trajectory_[t * NUM_JOINTS + j] = task_.nominal_pose[j];

    // Cost function tracks deviation from nominal pose
    std::copy(task_.nominal_pose, task_.nominal_pose + NUM_JOINTS, joint_ref_);
}

// -----------------------------------------------------------------------
// Step cost
// -----------------------------------------------------------------------
double MPPILocomotion::step_cost(const mjData* d, const double q_des[NUM_JOINTS]) {
    const CostWeights& w = task_.cost;
    double cost = 0.0;

    // Height: keep base at measured standing height
    double hz   = d->qpos[2] - height_target_;
    cost += w.height * hz * hz;

    // Orientation: keep upright (quat w ≈ 1)
    double tilt = 1.0 - d->qpos[3];
    cost += w.orientation * tilt * tilt;

    // Base linear velocity: penalise drifting
    for (int j = 0; j < 3; ++j)
        cost += w.lin_vel * d->qvel[j] * d->qvel[j];

    // Base angular velocity: penalise tipping
    for (int j = 3; j < 6; ++j)
        cost += w.ang_vel * d->qvel[j] * d->qvel[j];

    // Joint tracking: penalise target deviation from reference pose
    for (int j = 0; j < NUM_JOINTS; ++j) {
        double dq = q_des[j] - joint_ref_[j];
        cost += w.joint_track * dq * dq;
    }

    // PD effort: penalise how hard we're working
    for (int j = 0; j < NUM_JOINTS; ++j) {
        double tau = d->actuator_force[j];
        cost += w.effort * tau * tau;
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
// Rollout: simulate horizon control steps, each held for substeps physics steps
// -----------------------------------------------------------------------
double MPPILocomotion::rollout(int s, const RobotState& state) {
    mjData* d = data_[s];
    set_mj_state(d, state);

    double total_cost = 0.0;

    for (int t = 0; t < task_.horizon; ++t) {
        // Perturbed position targets, clamped to joint limits
        double q_des[NUM_JOINTS];
        for (int j = 0; j < NUM_JOINTS; ++j) {
            double nom   = trajectory_[t * NUM_JOINTS + j];
            double noisy = nom + noise_[s * task_.horizon * NUM_JOINTS + t * NUM_JOINTS + j];
            q_des[j] = std::clamp(noisy, q_min_[j], q_max_[j]);
        }

        // Hold target for substeps physics steps — matches the real 50 Hz control loop
        for (int sub = 0; sub < task_.substeps; ++sub) {
            for (int j = 0; j < NUM_JOINTS; ++j) {
                double q_act  = d->qpos[7 + j];
                double dq_act = d->qvel[6 + j];
                d->ctrl[j] = task_.kp * (q_des[j] - q_act)
                           + task_.kd * (0.0      - dq_act);
            }
            mj_step(model_, d);

            if (!std::isfinite(d->qpos[2]) || d->qpos[2] < -1.0)
                return 1e6;
        }

        total_cost += step_cost(d, q_des);
    }

    total_cost += terminal_cost(d);
    return std::isfinite(total_cost) ? total_cost : 1e6;
}

// -----------------------------------------------------------------------
// MPPI update — analogous to MPPI_quad's mppi_locomotion.py update()
// -----------------------------------------------------------------------
void MPPILocomotion::update(const RobotState& state, double q_targets_out[NUM_JOINTS]) {
    if (!state.valid) {
        for (int j = 0; j < NUM_JOINTS; ++j)
            q_targets_out[j] = trajectory_[j];
        return;
    }

    sample_noise();

    // Parallel rollouts
    std::vector<std::future<double>> futures;
    futures.reserve(task_.n_samples);
    for (int s = 0; s < task_.n_samples; ++s)
        futures.push_back(std::async(std::launch::async,
            [this, s, &state]() { return rollout(s, state); }));

    for (int s = 0; s < task_.n_samples; ++s)
        costs_[s] = futures[s].get();

    // MPPI weights: exp(-1/λ * (cost - min_cost))
    double cost_min = *std::min_element(costs_.begin(), costs_.end());
    std::vector<double> weights(task_.n_samples);
    double weight_sum = 0.0;
    for (int s = 0; s < task_.n_samples; ++s) {
        weights[s]  = std::exp(-1.0 / task_.lambda * (costs_[s] - cost_min));
        weight_sum += weights[s];
    }

    // Weighted average update
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

    // Clamp to joint limits
    for (int t = 0; t < task_.horizon; ++t)
        for (int j = 0; j < NUM_JOINTS; ++j) {
            int idx = t * NUM_JOINTS + j;
            new_traj[idx] = std::clamp(new_traj[idx], q_min_[j], q_max_[j]);
        }

    // Output first control step
    for (int j = 0; j < NUM_JOINTS; ++j)
        q_targets_out[j] = new_traj[j];

    // Shift trajectory (receding horizon)
    std::copy(new_traj.begin() + NUM_JOINTS, new_traj.end(), trajectory_.begin());
    std::copy(new_traj.end() - NUM_JOINTS,   new_traj.end(), trajectory_.end() - NUM_JOINTS);
}
