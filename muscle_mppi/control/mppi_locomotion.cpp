#include "mppi_locomotion.h"
#include "../utils/tasks.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <stdexcept>

MPPILocomotion::MPPILocomotion(const std::string& task_name)
    : BaseMPPI(get_task(task_name))
{
    muscle_ = task_.muscle;
    best_trajectory_.assign(task_.horizon * NUM_JOINTS, 0.0);

    // Warm-start: initialize leg activations to approximate standing values so
    // rollouts don't begin with zero torques and immediate collapse.
    // Hip≈0 (minimal abduction load), thigh≈0.3, calf≈0.2, wheels≈0.
    static constexpr double STAND_ACT[NUM_JOINTS] = {
        0.0, 0.15, 0.15,   // FR hip/thigh/calf
        0.0, 0.15, 0.15,   // FL
        0.0, 0.15, 0.15,   // RR
        0.0, 0.15, 0.15,   // RL
        0.0, 0.0,  0.0, 0.0  // wheels
    };
    for (int t = 0; t < task_.horizon; ++t)
        for (int j = 0; j < NUM_JOINTS; ++j)
            best_trajectory_[t * NUM_JOINTS + j] = STAND_ACT[j];
    std::memcpy(predicted_activation_, STAND_ACT, sizeof(STAND_ACT));
    // Seed real-robot activation state so predict_state() starts with correct torques.
    std::memcpy(muscle_state_.activation, STAND_ACT, sizeof(STAND_ACT));

    // Base body ID
    for (const char* name : {"trunk", "base", "base_link"}) {
        int bid = mj_name2id(model_, mjOBJ_BODY, name);
        if (bid >= 0) { base_bid_ = bid; break; }
    }

    // Wheel body IDs and spin-axle directions — FR, FL, RR, RL
    static const char* WHEEL_BODY_NAMES[4] = {
        "FR_wheel_link", "FL_wheel_link", "RR_wheel_link", "RL_wheel_link"
    };
    for (int k = 0; k < 4; ++k) {
        wheel_body_ids_[k] = mj_name2id(model_, mjOBJ_BODY, WHEEL_BODY_NAMES[k]);
        if (wheel_body_ids_[k] < 0)
            throw std::runtime_error(std::string("Body not found: ") + WHEEL_BODY_NAMES[k]);
        // Actuator NUM_LEG_JOINTS+k is the wheel joint; read its axis from the model
        const int jid = model_->actuator_trnid[2 * (NUM_LEG_JOINTS + k)];
        wheel_axle_[k][0] = model_->jnt_axis[3 * jid + 0];
        wheel_axle_[k][1] = model_->jnt_axis[3 * jid + 1];
        wheel_axle_[k][2] = model_->jnt_axis[3 * jid + 2];
    }

    // Nominal contact force per wheel: total_mass * |g| / 4
    double total_mass = 0.0;
    for (int i = 1; i < model_->nbody; ++i)
        total_mass += model_->body_mass[i];
    f_nominal_ = total_mass * std::abs(model_->opt.gravity[2]) / 4.0;
}

// -----------------------------------------------------------------------
// step_cost — paper Eq. (17), Table II walking weights
//
// c_t = w_h|Δz| + w_orient*θ² + w_q*||q-q0||² + w_cv*||v_c||₁ + w_cf*||f_c-f0||₁
// -----------------------------------------------------------------------
double MPPILocomotion::step_cost(const mjData* d,
                                  const double act_cmd[NUM_JOINTS],
                                  const double act_prev[NUM_JOINTS])
{
    const CostWeights& w = task_.cost;
    double cost = 0.0;

    // -- Height (L1) at a point 0.1 m ahead of trunk in body frame --
    // xmat column 0 = body x-axis in world; z-component = xmat[6] (R[2][0])
    static constexpr double kHeightFwdOffset = 0.1;
    const mjtNum* xmat  = d->xmat + 9 * base_bid_;
    const double  z_fwd = d->xpos[base_bid_ * 3 + 2] + kHeightFwdOffset * xmat[6];
    cost += w.height * std::abs(z_fwd - height_target_);

    // -- Orientation: geodesic angle² from upright.
    // ||log(R)||² = θ² where θ = 2*acos(|qw|).  Exact for all tilt magnitudes.
    double qw    = d->qpos[3];
    double angle = 2.0 * std::acos(std::clamp(std::abs(qw), 0.0, 1.0));
    cost += w.orientation * angle * angle;

    // -- Posture: leg joints near nominal (w_q=0 disables for walking) --
    if (w.posture > 0.0) {
        for (int j = 0; j < NUM_LEG_JOINTS; ++j) {
            double dq = d->qpos[act_qpos_adr_[j]] - task_.nominal_pose[j];
            cost += w.posture * dq * dq;
        }
    }

    // -- Contact velocity: lateral slip only (velocity along wheel axle).
    // Rolling (perpendicular to axle) is free; only sideways slip is penalised.
    if (w.contact_vel > 0.0) {
        for (int k = 0; k < 4; ++k) {
            mjtNum vel6[6];
            mj_objectVelocity(model_, d, mjOBJ_BODY, wheel_body_ids_[k], vel6, 0);
            // vel6[3..5] = linear velocity in world frame
            const mjtNum* wmat = d->xmat + 9 * wheel_body_ids_[k];
            // Rotate axle from body frame to world frame (column 0 of rotation matrix)
            const double ax = wmat[0]*wheel_axle_[k][0] + wmat[1]*wheel_axle_[k][1] + wmat[2]*wheel_axle_[k][2];
            const double ay = wmat[3]*wheel_axle_[k][0] + wmat[4]*wheel_axle_[k][1] + wmat[5]*wheel_axle_[k][2];
            const double az = wmat[6]*wheel_axle_[k][0] + wmat[7]*wheel_axle_[k][1] + wmat[8]*wheel_axle_[k][2];
            const double slip = ax*vel6[3] + ay*vel6[4] + az*vel6[5];
            cost += w.contact_vel * std::abs(slip);
        }
    }

    // -- Contact force: L1 deviation of wheel body force from nominal.
    // cfrc_ext[6*bid + 3..5] = external contact force on body (world frame).
    if (w.contact_force > 0.0) {
        for (int k = 0; k < 4; ++k) {
            int bid = wheel_body_ids_[k];
            double fx = d->cfrc_ext[6*bid + 3];
            double fy = d->cfrc_ext[6*bid + 4];
            double fz = d->cfrc_ext[6*bid + 5];
            // fx/fy should be ~0; fz should be ~f_nominal_
            cost += w.contact_force * (std::abs(fx) + std::abs(fy)
                                      + std::abs(fz - f_nominal_));
        }
    }

    // -- Activation smoothness --
    for (int j = 0; j < NUM_JOINTS; ++j) {
        double da = act_cmd[j] - act_prev[j];
        cost += w.act_smooth * da * da;
    }

    return cost;
}

// -----------------------------------------------------------------------
// terminal_cost — paper Eq. (18)
// c_T = w_H * ||p_base(xH) - p_target||₁,  p_target = p_0 + v_des*H*dt
// -----------------------------------------------------------------------
double MPPILocomotion::terminal_cost(const mjData* d)
{
    const CostWeights& w = task_.cost;

    const double dt_step   = task_.dt * task_.substeps;
    const double px_target = start_pos_[0] + cmd_.vx * task_.horizon * dt_step;
    const double py_target = start_pos_[1] + cmd_.vy * task_.horizon * dt_step;

    return w.terminal * (std::abs(d->qpos[0] - px_target)
                        + std::abs(d->qpos[1] - py_target));
}

// -----------------------------------------------------------------------
// Rollout
// -----------------------------------------------------------------------
double MPPILocomotion::rollout(int s, const RobotState& state)
{
    mjData* d = data_[s];
    set_mj_state(d, state);

    double activation[NUM_JOINTS];
    std::memcpy(activation, predicted_activation_, sizeof(activation));

    double total_cost = 0.0;

    double act_prev[NUM_JOINTS];
    std::memcpy(act_prev, predicted_activation_, sizeof(act_prev));

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
                q_cur[j]  = d->qpos[act_qpos_adr_[j]];
                dq_cur[j] = d->qvel[act_qvel_adr_[j]];
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
// State prediction (Sec. III-E, Eq. 14-15): simulate n_steps ahead using
// the prefix of best_trajectory_ to compensate for compute latency.
// Uses the dedicated slot data_[task_.n_samples] — isolated from rollout slots.
// -----------------------------------------------------------------------
RobotState MPPILocomotion::predict_state(const RobotState& state, int n_steps)
{
    mjData* d = data_[task_.n_samples];
    set_mj_state(d, state);

    double activation[NUM_JOINTS];
    std::memcpy(activation, muscle_state_.activation, sizeof(activation));

    for (int t = 0; t < n_steps; ++t) {
        double act_cmd[NUM_JOINTS];
        for (int j = 0; j < NUM_JOINTS; ++j)
            act_cmd[j] = best_trajectory_[t * NUM_JOINTS + j];

        double tau_out[NUM_JOINTS];
        for (int sub = 0; sub < task_.substeps; ++sub) {
            double q_cur[NUM_JOINTS], dq_cur[NUM_JOINTS];
            for (int j = 0; j < NUM_JOINTS; ++j) {
                q_cur[j]  = d->qpos[act_qpos_adr_[j]];
                dq_cur[j] = d->qvel[act_qvel_adr_[j]];
            }
            hill_compute_torques(act_cmd, q_cur, dq_cur, muscle_, activation, tau_out);
            for (int j = 0; j < NUM_JOINTS; ++j)
                d->ctrl[j] = tau_out[j] - muscle_.kd_sim[j] * dq_cur[j];
            mj_step(model_, d);
        }
    }

    RobotState predicted;
    predicted.pos[0]  = d->qpos[0];  predicted.pos[1]  = d->qpos[1];  predicted.pos[2]  = d->qpos[2];
    predicted.quat[0] = d->qpos[3];  predicted.quat[1] = d->qpos[4];
    predicted.quat[2] = d->qpos[5];  predicted.quat[3] = d->qpos[6];
    predicted.vel[0]  = d->qvel[0];  predicted.vel[1]  = d->qvel[1];  predicted.vel[2]  = d->qvel[2];
    predicted.gyro[0] = d->qvel[3];  predicted.gyro[1] = d->qvel[4];  predicted.gyro[2] = d->qvel[5];
    for (int j = 0; j < NUM_JOINTS; ++j) {
        predicted.q[j]  = d->qpos[act_qpos_adr_[j]];
        predicted.dq[j] = d->qvel[act_qvel_adr_[j]];
    }
    predicted.valid = true;
    std::memcpy(predicted_activation_, activation, sizeof(activation));
    return predicted;
}

// -----------------------------------------------------------------------
// MPPI update — state prediction + multiple planning iterations with noise
// annealing and best-trajectory tracking (Sec. III-D/E, Eq. 8).
// -----------------------------------------------------------------------
void MPPILocomotion::update(const RobotState& state, double activations_out[NUM_JOINTS])
{
    auto t_start = std::chrono::steady_clock::now();

    if (!state.valid) {
        for (int j = 0; j < NUM_JOINTS; ++j)
            activations_out[j] = best_trajectory_[j];
        return;
    }

    // Predict state n_skip MPC steps ahead to compensate for compute latency.
    // n_skip = round(last_compute_ms / dt_step), clamped to [1, horizon/2].
    const double dt_step = task_.substeps * task_.dt;
    const int n_skip = std::clamp(
        static_cast<int>(std::round(last_compute_ms_ * 1e-3 / dt_step)),
        1, task_.horizon / 2);

    RobotState predicted = predict_state(state, n_skip);

    start_pos_[0] = predicted.pos[0];
    start_pos_[1] = predicted.pos[1];
    start_pos_[2] = predicted.pos[2];

    // Warm-start: shift best_trajectory_ forward by n_skip steps, pad tail.
    for (int t = 0; t < task_.horizon - n_skip; ++t)
        for (int j = 0; j < NUM_JOINTS; ++j)
            trajectory_[t * NUM_JOINTS + j] =
                best_trajectory_[(t + n_skip) * NUM_JOINTS + j];
    for (int t = task_.horizon - n_skip; t < task_.horizon; ++t)
        for (int j = 0; j < NUM_JOINTS; ++j)
            trajectory_[t * NUM_JOINTS + j] =
                best_trajectory_[(task_.horizon - 1) * NUM_JOINTS + j];

    const int N = task_.n_iterations;
    best_cost_ = 1e9;

    for (int iter = 0; iter < N; ++iter) {
        sample_noise(iter, N);

        #pragma omp parallel for schedule(dynamic)
        for (int s = 0; s < task_.n_samples; ++s)
            costs_[s] = rollout(s, predicted);

        // Track best rollout seen across all iterations
        for (int s = 0; s < task_.n_samples; ++s) {
            if (costs_[s] < best_cost_) {
                best_cost_ = costs_[s];
                for (int t = 0; t < task_.horizon; ++t)
                    for (int j = 0; j < NUM_JOINTS; ++j) {
                        int idx = t * NUM_JOINTS + j;
                        best_trajectory_[idx] = std::clamp(
                            trajectory_[idx] + noise_[s * task_.horizon * NUM_JOINTS + idx],
                            ACT_MIN, ACT_MAX);
                    }
            }
        }

        // Weighted average → update trajectory_ for next iteration
        double cost_min   = *std::min_element(costs_.begin(), costs_.end());
        double cost_max   = *std::max_element(costs_.begin(), costs_.end());
        double cost_range = cost_max - cost_min;

        std::vector<double> weights(task_.n_samples);
        double weight_sum = 0.0;
        for (int s = 0; s < task_.n_samples; ++s) {
            double s_hat = (cost_range > 1e-12) ? (costs_[s] - cost_min) / cost_range : 0.0;
            weights[s]   = std::exp(-s_hat / task_.lambda);
            weight_sum  += weights[s];
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

        trajectory_ = std::move(new_traj);
    }

    // Execute from best_trajectory_ (not weighted average)
    for (int j = 0; j < NUM_JOINTS; ++j)
        activations_out[j] = best_trajectory_[j];

    last_compute_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_start).count();
}

// -----------------------------------------------------------------------
// Cost breakdown (zero-noise nominal rollout)
// -----------------------------------------------------------------------
MPPILocomotion::CostBreakdown
MPPILocomotion::diagnose_cost(const RobotState& state)
{
    mjData* d = data_[task_.n_samples];
    set_mj_state(d, state);

    double activation[NUM_JOINTS];
    std::memcpy(activation, predicted_activation_, sizeof(activation));

    double act_prev[NUM_JOINTS];
    std::memcpy(act_prev, predicted_activation_, sizeof(act_prev));

    CostBreakdown bd;
    const CostWeights& w = task_.cost;

    for (int t = 0; t < task_.horizon; ++t) {
        double act_cmd[NUM_JOINTS];
        for (int j = 0; j < NUM_JOINTS; ++j)
            act_cmd[j] = best_trajectory_[t * NUM_JOINTS + j];

        double tau_out[NUM_JOINTS];
        for (int sub = 0; sub < task_.substeps; ++sub) {
            double q_cur[NUM_JOINTS], dq_cur[NUM_JOINTS];
            for (int j = 0; j < NUM_JOINTS; ++j) {
                q_cur[j]  = d->qpos[act_qpos_adr_[j]];
                dq_cur[j] = d->qvel[act_qvel_adr_[j]];
            }
            hill_compute_torques(act_cmd, q_cur, dq_cur, muscle_, activation, tau_out);
            for (int j = 0; j < NUM_JOINTS; ++j)
                d->ctrl[j] = tau_out[j] - muscle_.kd_sim[j] * dq_cur[j];
            mj_step(model_, d);
        }

        const mjtNum* dxmat  = d->xmat + 9 * base_bid_;
        const double  dz_fwd = d->xpos[base_bid_ * 3 + 2] + 0.1 * dxmat[6];
        bd.height += w.height * std::abs(dz_fwd - height_target_);

        double qw    = d->qpos[3];
        double angle = 2.0 * std::acos(std::clamp(std::abs(qw), 0.0, 1.0));
        bd.orientation += w.orientation * angle * angle;

        if (w.posture > 0.0) {
            for (int j = 0; j < NUM_LEG_JOINTS; ++j) {
                double dq = d->qpos[act_qpos_adr_[j]] - task_.nominal_pose[j];
                bd.posture += w.posture * dq * dq;
            }
        }

        if (w.contact_vel > 0.0) {
            for (int k = 0; k < 4; ++k) {
                mjtNum vel6[6];
                mj_objectVelocity(model_, d, mjOBJ_BODY, wheel_body_ids_[k], vel6, 0);
                const mjtNum* wmat = d->xmat + 9 * wheel_body_ids_[k];
                const double ax = wmat[0]*wheel_axle_[k][0] + wmat[1]*wheel_axle_[k][1] + wmat[2]*wheel_axle_[k][2];
                const double ay = wmat[3]*wheel_axle_[k][0] + wmat[4]*wheel_axle_[k][1] + wmat[5]*wheel_axle_[k][2];
                const double az = wmat[6]*wheel_axle_[k][0] + wmat[7]*wheel_axle_[k][1] + wmat[8]*wheel_axle_[k][2];
                bd.contact_vel += w.contact_vel * std::abs(ax*vel6[3] + ay*vel6[4] + az*vel6[5]);
            }
        }

        if (w.contact_force > 0.0) {
            for (int k = 0; k < 4; ++k) {
                int bid = wheel_body_ids_[k];
                double fx = d->cfrc_ext[6*bid + 3];
                double fy = d->cfrc_ext[6*bid + 4];
                double fz = d->cfrc_ext[6*bid + 5];
                bd.contact_force += w.contact_force
                    * (std::abs(fx) + std::abs(fy) + std::abs(fz - f_nominal_));
            }
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

    return bd;
}

// -----------------------------------------------------------------------
// Real-robot Hill model
// -----------------------------------------------------------------------
void MPPILocomotion::compute_real_torques(const RobotState& state,
                                          const double activations[NUM_JOINTS],
                                          double tau_out[NUM_JOINTS])
{
    hill_compute_torques(activations, state.q, state.dq, muscle_,
                         muscle_state_.activation, tau_out);
}
