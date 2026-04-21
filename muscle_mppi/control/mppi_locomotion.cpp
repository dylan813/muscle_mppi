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
    // Trajectory and activation state initialised to zero.
    // MPPI optimises from this cold start on the first update.
}

// -----------------------------------------------------------------------
// step_cost — reference-free running cost
//
// Penalises: height deviation, orientation tilt, leg posture vs nominal,
//            and activation rate-of-change (smoothness).
// No gait reference is used — locomotion emerges from the dynamics.
// -----------------------------------------------------------------------
double MPPILocomotion::step_cost(const mjData* d,
                                  const double act_cmd[NUM_JOINTS],
                                  const double act_prev[NUM_JOINTS])
{
    const CostWeights& w = task_.cost;
    double cost = 0.0;

    // -- Body height --
    double hz = d->qpos[2] - height_target_;
    cost += w.height * hz * hz;

    // -- Orientation: penalise tilt from upright on all three rotation axes.
    // qx²+qy²+qz² ≈ θ²/4 for small tilt (includes yaw drift).
    double qx = d->qpos[4], qy = d->qpos[5], qz = d->qpos[6];
    cost += w.orientation * (qx*qx + qy*qy + qz*qz);

    // -- Leg posture vs nominal standing pose --
    for (int j = 0; j < NUM_LEG_JOINTS; ++j) {
        double dq = d->qpos[7 + LS_TO_QPOS[j]] - task_.nominal_pose[j];
        cost += w.posture * dq * dq;
    }

    // -- Activation smoothness (rate of change) --
    for (int j = 0; j < NUM_JOINTS; ++j) {
        double da = act_cmd[j] - act_prev[j];
        cost += w.act_smooth * da * da;
    }

    return cost;
}

// -----------------------------------------------------------------------
// terminal_cost — L1 displacement error + height/orientation at horizon end.
//
// Drives the robot toward (start_pos + v_cmd * H * dt) without prescribing
// how to get there — gait and contact emerge from Hill model + physics.
// -----------------------------------------------------------------------
double MPPILocomotion::terminal_cost(const mjData* d)
{
    const CostWeights& w = task_.cost;

    const double dt_step   = task_.dt * task_.substeps;
    const double px_target = start_pos_[0] + cmd_.vx * task_.horizon * dt_step;
    const double py_target = start_pos_[1] + cmd_.vy * task_.horizon * dt_step;

    double cost = w.terminal * (std::abs(d->qpos[0] - px_target)
                               + std::abs(d->qpos[1] - py_target));

    double hz = d->qpos[2] - height_target_;
    double qx = d->qpos[4], qy = d->qpos[5], qz = d->qpos[6];
    cost += w.height * hz * hz + w.orientation * (qx*qx + qy*qy + qz*qz);

    return cost;
}

// -----------------------------------------------------------------------
// Rollout — forward simulate horizon steps, accumulate running + terminal cost
// -----------------------------------------------------------------------
double MPPILocomotion::rollout(int s, const RobotState& state)
{
    mjData* d = data_[s];
    set_mj_state(d, state);

    double activation[NUM_JOINTS];
    std::memcpy(activation, muscle_state_.activation, sizeof(activation));

    double total_cost = 0.0;

    double act_prev[NUM_JOINTS];
    std::memcpy(act_prev, muscle_state_.activation, sizeof(act_prev));

    for (int t = 0; t < task_.horizon; ++t) {
        double act_cmd[NUM_JOINTS];
        for (int j = 0; j < NUM_JOINTS; ++j) {
            double nom   = trajectory_[t * NUM_JOINTS + j];
            double noisy = nom + noise_[s * task_.horizon * NUM_JOINTS
                                        + t * NUM_JOINTS + j];
            act_cmd[j] = std::clamp(noisy, ACT_MIN, ACT_MAX);
        }

        double tau_out[NUM_JOINTS];
        for (int sub = 0; sub < task_.substeps; ++sub) {
            double q_cur[NUM_JOINTS], dq_cur[NUM_JOINTS];
            for (int j = 0; j < NUM_JOINTS; ++j) {
                q_cur[j]  = d->qpos[7 + LS_TO_QPOS[j]];
                dq_cur[j] = d->qvel[6 + LS_TO_QPOS[j]];
            }
            hill_compute_torques(act_cmd, q_cur, dq_cur, muscle_, activation, tau_out);

            for (int j = 0; j < NUM_JOINTS; ++j)
                d->ctrl[j] = tau_out[j] - muscle_.kd_sim[j] * dq_cur[j];

            mj_step(model_, d);

            if (!std::isfinite(d->qpos[2]) || d->qpos[2] < -1.0)
                return 1e6;
        }

        total_cost += step_cost(d, act_cmd, act_prev);
        std::memcpy(act_prev, act_cmd, sizeof(act_cmd));
    }

    total_cost += terminal_cost(d);
    return std::isfinite(total_cost) ? total_cost : 1e6;
}

// -----------------------------------------------------------------------
// MPPI update — trajectory is activation commands ∈ [-1, 1]
// -----------------------------------------------------------------------
void MPPILocomotion::update(const RobotState& state, double activations_out[NUM_JOINTS])
{
    if (!state.valid) {
        for (int j = 0; j < NUM_JOINTS; ++j)
            activations_out[j] = trajectory_[j];
        return;
    }

    // Capture start position for terminal displacement cost — shared by all rollouts.
    start_pos_[0] = state.pos[0];
    start_pos_[1] = state.pos[1];
    start_pos_[2] = state.pos[2];

    sample_noise();

    std::vector<std::future<double>> futures;
    futures.reserve(task_.n_samples);
    for (int s = 0; s < task_.n_samples; ++s)
        futures.push_back(std::async(std::launch::async,
            [this, s, &state]() { return rollout(s, state); }));

    for (int s = 0; s < task_.n_samples; ++s)
        costs_[s] = futures[s].get();

    // Range-normalise costs before exponentiation so λ is scale-invariant.
    double cost_min = *std::min_element(costs_.begin(), costs_.end());
    double cost_max = *std::max_element(costs_.begin(), costs_.end());
    double cost_range = cost_max - cost_min;

    std::vector<double> weights(task_.n_samples);
    double weight_sum = 0.0;
    for (int s = 0; s < task_.n_samples; ++s) {
        double s_hat = (cost_range > 1e-12) ? (costs_[s] - cost_min) / cost_range : 0.0;
        weights[s]  = std::exp(-s_hat / task_.lambda);
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

    // Shift trajectory: drop step 0, repeat last step
    std::copy(new_traj.begin() + NUM_JOINTS, new_traj.end(), trajectory_.begin());
    std::copy(new_traj.end() - NUM_JOINTS,   new_traj.end(), trajectory_.end() - NUM_JOINTS);
}

// -----------------------------------------------------------------------
// Cost breakdown — zero-noise rollout using current nominal trajectory
// -----------------------------------------------------------------------
MPPILocomotion::CostBreakdown
MPPILocomotion::diagnose_cost(const RobotState& state)
{
    mjData* d = data_[0];
    set_mj_state(d, state);

    double activation[NUM_JOINTS];
    std::memcpy(activation, muscle_state_.activation, sizeof(activation));

    double act_prev[NUM_JOINTS];
    std::memcpy(act_prev, muscle_state_.activation, sizeof(act_prev));

    CostBreakdown bd;
    const CostWeights& w = task_.cost;

    for (int t = 0; t < task_.horizon; ++t) {
        double act_cmd[NUM_JOINTS];
        for (int j = 0; j < NUM_JOINTS; ++j)
            act_cmd[j] = trajectory_[t * NUM_JOINTS + j];

        double tau_out[NUM_JOINTS];
        for (int sub = 0; sub < task_.substeps; ++sub) {
            double q_cur[NUM_JOINTS], dq_cur[NUM_JOINTS];
            for (int j = 0; j < NUM_JOINTS; ++j) {
                q_cur[j]  = d->qpos[7 + LS_TO_QPOS[j]];
                dq_cur[j] = d->qvel[6 + LS_TO_QPOS[j]];
            }
            hill_compute_torques(act_cmd, q_cur, dq_cur, muscle_, activation, tau_out);
            for (int j = 0; j < NUM_JOINTS; ++j)
                d->ctrl[j] = tau_out[j] - muscle_.kd_sim[j] * dq_cur[j];
            mj_step(model_, d);
        }

        double hz = d->qpos[2] - height_target_;
        bd.height += w.height * hz * hz;

        double qx = d->qpos[4], qy = d->qpos[5], qz = d->qpos[6];
        bd.orientation += w.orientation * (qx*qx + qy*qy + qz*qz);

        for (int j = 0; j < NUM_LEG_JOINTS; ++j) {
            double dq = d->qpos[7 + LS_TO_QPOS[j]] - task_.nominal_pose[j];
            bd.posture += w.posture * dq * dq;
        }

        for (int j = 0; j < NUM_JOINTS; ++j) {
            double da = act_cmd[j] - act_prev[j];
            bd.act_smooth += w.act_smooth * da * da;
        }
        std::memcpy(act_prev, act_cmd, sizeof(act_cmd));
    }

    // Terminal
    const double dt_step   = task_.dt * task_.substeps;
    const double px_target = start_pos_[0] + cmd_.vx * task_.horizon * dt_step;
    const double py_target = start_pos_[1] + cmd_.vy * task_.horizon * dt_step;
    bd.terminal = w.terminal * (std::abs(d->qpos[0] - px_target)
                                + std::abs(d->qpos[1] - py_target));
    double hz = d->qpos[2] - height_target_;
    double qx = d->qpos[4], qy = d->qpos[5], qz = d->qpos[6];
    bd.terminal += w.height * hz * hz + w.orientation * (qx*qx + qy*qy + qz*qz);

    return bd;
}

// -----------------------------------------------------------------------
// Real-robot Hill model — advances persistent muscle_state_
// -----------------------------------------------------------------------
void MPPILocomotion::compute_real_torques(const RobotState& state,
                                          const double activations[NUM_JOINTS],
                                          double tau_out[NUM_JOINTS])
{
    hill_compute_torques(activations, state.q, state.dq, muscle_,
                         muscle_state_.activation, tau_out);
}
