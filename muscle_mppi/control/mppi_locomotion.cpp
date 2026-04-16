#include "mppi_locomotion.h"
#include "../utils/tasks.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <future>

MPPILocomotion::MPPILocomotion(const std::string& task_name)
    : BaseMPPI(get_task(task_name))
    , gait_sched_(task_.nominal_pose)
{
    muscle_ = task_.muscle;

    // ------------------------------------------------------------------
    // Gravity-compensation warm-start.
    //
    // Correct grounded torque: tau = qfrc_bias - qfrc_constraint
    //
    // qfrc_bias   = gravitational load (free-floating)
    // qfrc_constraint = ground-contact contribution in generalised coords
    //
    // Using qfrc_bias alone over-torques the legs because ground contact
    // already carries part of the load.  The excess torque creates net
    // forces that accelerate the body horizontally.
    //
    // Hill steady-state: tau = tanh(act_cmd) * tau_max
    //   → act_cmd = atanh(tau / tau_max)
    // ------------------------------------------------------------------
    mjData* d0 = mj_makeData(model_);
    mj_resetData(model_, d0);
    d0->qpos[2] = task_.height_target;
    d0->qpos[3] = 1.0;
    for (int j = 0; j < NUM_JOINTS; ++j)
        d0->qpos[7 + LS_TO_QPOS[j]] = task_.nominal_pose[j];
    mj_forward(model_, d0);   // computes contacts → populates qfrc_constraint

    for (int j = 0; j < NUM_JOINTS; ++j) {
        double act_cmd;

        if (j < NUM_LEG_JOINTS) {
            // Grounded equilibrium torque: bias minus contact contribution.
            int    dof      = 6 + LS_TO_QPOS[j];
            double tau_grav = d0->qfrc_bias[dof] - d0->qfrc_constraint[dof];
            double ratio    = std::clamp(tau_grav / muscle_.tau_max[j], -0.99, 0.99);
            act_cmd         = std::clamp(std::atanh(ratio), ACT_MIN, ACT_MAX);
        } else {
            // Wheel joints: zero warm-start.
            // Free-floating qfrc_bias for wheel DOFs reflects load absorbed by ground
            // contact; applying it spins the wheels and translates the body.
            act_cmd = 0.0;
        }

        for (int t = 0; t < task_.horizon; ++t)
            trajectory_[t * NUM_JOINTS + j] = act_cmd;

        muscle_state_.activation[j] = std::tanh(act_cmd);
    }
    mj_deleteData(d0);
}

// -----------------------------------------------------------------------
// step_cost — deviation from GaitReference + activation regularisation
// -----------------------------------------------------------------------
double MPPILocomotion::step_cost(const mjData* d,
                                  const StepReference& ref,
                                  const double act_cmd[NUM_JOINTS],
                                  const double act_prev[NUM_JOINTS])
{
    const CostWeights& w = task_.cost;
    double cost = 0.0;

    // -- Body height --
    double hz = d->qpos[2] - ref.body_pos[2];
    cost += w.height * hz * hz;

    // -- Body orientation (quaternion tilt from upright)
    // Use pitch+roll components of quaternion vector: qx²+qy² ≈ θ²/4 for small tilt.
    // (1-qw)² ≈ θ⁴/64 — quartic, nearly zero for small tilts; qx²+qy² is quadratic.
    double qx = d->qpos[4], qy = d->qpos[5];
    cost += w.orientation * (qx*qx + qy*qy);

    // -- Body linear velocity vs reference --
    for (int k = 0; k < 3; ++k) {
        double dv = d->qvel[k] - ref.body_vel[k];
        cost += w.lin_vel * dv * dv;
    }

    // -- Body angular velocity vs reference --
    for (int k = 0; k < 3; ++k) {
        double dw = d->qvel[3 + k] - ref.body_omega[k];
        cost += w.ang_vel * dw * dw;
    }

    // -- Leg joint positions vs gait-scheduled targets --
    for (int j = 0; j < NUM_LEG_JOINTS; ++j) {
        double dq = d->qpos[7 + LS_TO_QPOS[j]] - ref.joint_pos[j];
        cost += w.joint_track * dq * dq;
    }

    // -- Wheel angular velocity vs rolling target --
    for (int w_idx = 0; w_idx < NUM_WHEELS; ++w_idx) {
        double dv = d->qvel[6 + LS_TO_QPOS[NUM_LEG_JOINTS + w_idx]] - ref.wheel_vel[w_idx];
        cost += w.wheel_vel * dv * dv;
    }

    // -- Activation smoothness (rate of change of commanded activation) --
    for (int j = 0; j < NUM_JOINTS; ++j) {
        double da = act_cmd[j] - act_prev[j];
        cost += w.act_smooth * da * da;
    }

    return cost;
}

double MPPILocomotion::terminal_cost(const mjData* d, const StepReference& ref)
{
    const CostWeights& w = task_.cost;

    double hz   = d->qpos[2] - ref.body_pos[2];
    double qx   = d->qpos[4], qy = d->qpos[5];
    double cost = w.height * hz * hz + w.orientation * (qx*qx + qy*qy);

    for (int j = 0; j < NUM_LEG_JOINTS; ++j) {
        double dq = d->qpos[7 + LS_TO_QPOS[j]] - ref.joint_pos[j];
        cost += w.terminal_joint * dq * dq;
    }
    return cost;
}

// -----------------------------------------------------------------------
// Rollout — generate gait reference first, then track it step-by-step
// -----------------------------------------------------------------------
double MPPILocomotion::rollout(int s, const RobotState& state)
{
    mjData* d = data_[s];
    set_mj_state(d, state);

    // Local copy of activation state for this rollout sample
    double activation[NUM_JOINTS];
    std::memcpy(activation, muscle_state_.activation, sizeof(activation));

    // Use the reference pre-computed in update() — same for all samples
    const GaitReference& ref = gait_ref_;

    double total_cost = 0.0;

    // Track previous activation command for smoothness cost
    double act_prev[NUM_JOINTS];
    std::memcpy(act_prev, muscle_state_.activation, sizeof(act_prev));

    for (int t = 0; t < task_.horizon; ++t) {
        // Perturbed activation command, clamped to [-1, 1]
        double act_cmd[NUM_JOINTS];
        for (int j = 0; j < NUM_JOINTS; ++j) {
            double nom   = trajectory_[t * NUM_JOINTS + j];
            double noisy = nom + noise_[s * task_.horizon * NUM_JOINTS
                                        + t * NUM_JOINTS + j];
            act_cmd[j] = std::clamp(noisy, ACT_MIN, ACT_MAX);
        }

        // Hold act_cmd over all substeps; Hill model updates activation[] each substep
        double tau_out[NUM_JOINTS];
        for (int sub = 0; sub < task_.substeps; ++sub) {
            double q_cur[NUM_JOINTS], dq_cur[NUM_JOINTS];
            for (int j = 0; j < NUM_JOINTS; ++j) {
                q_cur[j]  = d->qpos[7 + LS_TO_QPOS[j]];
                dq_cur[j] = d->qvel[6 + LS_TO_QPOS[j]];
            }
            hill_compute_torques(act_cmd, q_cur, dq_cur, muscle_, activation, tau_out);

            // Mirror real-robot control: Hill feedforward + hardware kd toward zero velocity.
            // Without this the rollout is underdamped vs the real robot, causing MPPI to
            // optimise activations for a dynamics model that doesn't match deployment.
            for (int j = 0; j < NUM_JOINTS; ++j)
                d->ctrl[j] = tau_out[j] - muscle_.kd_sim[j] * dq_cur[j];

            mj_step(model_, d);

            if (!std::isfinite(d->qpos[2]) || d->qpos[2] < -1.0)
                return 1e6;
        }

        // Cost: deviation from gait reference + activation smoothness
        total_cost += step_cost(d, ref.steps[t], act_cmd, act_prev);

        // Advance act_prev for smoothness cost at next step
        std::memcpy(act_prev, act_cmd, sizeof(act_cmd));
    }

    total_cost += terminal_cost(d, ref.steps[task_.horizon - 1]);
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

    // Generate gait reference once — all rollout threads read the same reference.
    // This ensures fair comparison across samples and avoids 32x redundant work.
    gait_sched_.generate(state, task_.horizon, task_.dt * task_.substeps, gait_ref_);

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
        // Nominal trajectory — no noise
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

        const StepReference& ref = gait_ref_.steps[t];

        double hz = d->qpos[2] - ref.body_pos[2];
        bd.height += w.height * hz * hz;

        double qx = d->qpos[4], qy = d->qpos[5];
        bd.orientation += w.orientation * (qx*qx + qy*qy);

        for (int k = 0; k < 3; ++k) {
            double dv = d->qvel[k] - ref.body_vel[k];
            bd.lin_vel += w.lin_vel * dv * dv;
        }
        for (int k = 0; k < 3; ++k) {
            double dw = d->qvel[3 + k] - ref.body_omega[k];
            bd.ang_vel += w.ang_vel * dw * dw;
        }
        for (int j = 0; j < NUM_LEG_JOINTS; ++j) {
            double dq = d->qpos[7 + LS_TO_QPOS[j]] - ref.joint_pos[j];
            bd.joint_track += w.joint_track * dq * dq;
        }
        for (int w_idx = 0; w_idx < NUM_WHEELS; ++w_idx) {
            double dv = d->qvel[6 + LS_TO_QPOS[NUM_LEG_JOINTS + w_idx]] - ref.wheel_vel[w_idx];
            bd.wheel_vel += w.wheel_vel * dv * dv;
        }
        for (int j = 0; j < NUM_JOINTS; ++j) {
            double da = act_cmd[j] - act_prev[j];
            bd.act_smooth += w.act_smooth * da * da;
        }
        std::memcpy(act_prev, act_cmd, sizeof(act_cmd));
    }

    // Terminal
    const StepReference& last = gait_ref_.steps[task_.horizon - 1];
    double hz  = d->qpos[2] - last.body_pos[2];
    double qx  = d->qpos[4], qy = d->qpos[5];
    bd.terminal = w.height * hz * hz + w.orientation * (qx*qx + qy*qy);
    for (int j = 0; j < NUM_LEG_JOINTS; ++j) {
        double dq = d->qpos[7 + LS_TO_QPOS[j]] - last.joint_pos[j];
        bd.terminal += w.terminal_joint * dq * dq;
    }

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
