#include "single_leg_torque.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

SingleLegTorque::SingleLegTorque(const std::string& task_name,
                                   const std::string& yaml_path)
    : BaseMPPI(load_task(task_name, yaml_path))
{
    for (int j = 0; j < NUM_JOINTS; ++j) {
        action_lo_[j] = -task_.tau_max[j];
        action_hi_[j] =  task_.tau_max[j];
    }

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

    warm_start(1);
    run_iterations(state);

    for (int j = 0; j < NUM_JOINTS; ++j)
        tau_out[j] = best_traj_[j];
}
