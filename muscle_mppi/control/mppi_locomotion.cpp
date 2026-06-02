#include "mppi_locomotion.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <omp.h>
#include <stdexcept>
#include <iostream>

// ============================================================================
// Constructor
// ============================================================================

MPPILocomotion::MPPILocomotion(const std::string& task_name, const std::string& yaml_path,
                                const std::string& log_dir)
    : BaseMPPI(load_task(task_name, yaml_path))
{
    muscle_ = task_.muscle;
    n_nodes_ = task_.n_nodes;

    // Δt between spline nodes: total horizon time / (K-1)
    const double horizon_secs = task_.horizon * task_.substeps * task_.dt;
    dt_node_ = horizon_secs / std::max(n_nodes_ - 1, 1);

    // Resize base-class arrays for spline node layout.
    // trajectory_ and best_trajectory_: [K × 2 × NUM_JOINTS]
    // noise_:                            [N_samples × K × 2 × NUM_JOINTS]
    // noise_sched_:                      [N_iterations × K]
    const int stride = n_nodes_ * 2 * NUM_JOINTS;
    trajectory_.assign(stride, 0.0);
    best_trajectory_.assign(stride, 0.0);
    noise_.assign(task_.n_samples * stride, 0.0);

    const int N = task_.n_iterations;
    noise_sched_.assign(N * n_nodes_, 0.0);
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < n_nodes_; ++k)
            noise_sched_[i * n_nodes_ + k] = std::exp(-0.5 * (
                static_cast<double>(i) / (task_.beta1 * N)
                + static_cast<double>(n_nodes_ - k) / (task_.beta2 * n_nodes_)));

    // Initialise spline nodes at nominal pose, zero velocity.
    for (int k = 0; k < n_nodes_; ++k)
        for (int j = 0; j < NUM_JOINTS; ++j) {
            trajectory_[k * 2 * NUM_JOINTS + j]              = task_.nominal_pose[j];
            trajectory_[k * 2 * NUM_JOINTS + NUM_JOINTS + j] = 0.0;
        }
    best_trajectory_ = trajectory_;

    // Find base body.
    for (const char* name : {"trunk", "base", "base_link"}) {
        int bid = mj_name2id(model_, mjOBJ_BODY, name);
        if (bid >= 0) { base_bid_ = bid; break; }
    }

    // Find feet — try all four, fall back gracefully.
    const char* foot_names[] = {"FL_foot", "FR_foot", "RL_foot", "RR_foot"};
    n_feet_ = 0;
    for (int i = 0; i < 4; ++i) {
        int bid = mj_name2id(model_, mjOBJ_BODY, foot_names[i]);
        if (bid >= 0) foot_body_ids_[n_feet_++] = bid;
    }
    if (n_feet_ == 0)
        throw std::runtime_error("MPPILocomotion: no foot bodies found in model");

    double total_mass = 0.0;
    for (int i = 1; i < model_->nbody; ++i)
        total_mass += model_->body_mass[i];
    f_nominal_ = total_mass * std::abs(model_->opt.gravity[2]) / 4.0;

    cmd_.vx = task_.cost.vel_des[0];
    cmd_.vy = task_.cost.vel_des[1];
    cmd_.wz = task_.cost.vel_des[2];

    std::filesystem::create_directories(log_dir);
    lat_log_.open(log_dir + "/walk_latency.csv", std::ios::out | std::ios::trunc);
    lat_log_ << "call,total_ms,predict_ms,warmstart_ms,iterations_ms,"
                "avg_hill_ms,avg_mjstep_ms,avg_cost_ms\n";
}

// ============================================================================
// Hermite spline evaluation (Sec. III-A, Eq. 2–4)
// ============================================================================

void MPPILocomotion::hermite_eval(const double* traj, double t_sec,
                                   double q_des[NUM_JOINTS],
                                   double dq_des[NUM_JOINTS]) const
{
    if (n_nodes_ <= 1) {
        for (int j = 0; j < NUM_JOINTS; ++j) {
            q_des[j]  = traj[j];
            dq_des[j] = traj[NUM_JOINTS + j];
        }
        return;
    }

    const double k_float = t_sec / dt_node_;
    int k = static_cast<int>(k_float);
    k = std::min(k, n_nodes_ - 2);
    const double s = std::clamp(k_float - k, 0.0, 1.0);

    // Cubic Hermite basis (Eq. 3, 4).
    const double s2 = s * s, s3 = s2 * s;
    const double h00 =  2*s3 - 3*s2 + 1;
    const double h10 =    s3 - 2*s2 + s;
    const double h01 = -2*s3 + 3*s2;
    const double h11 =    s3 -   s2;
    // Derivatives w.r.t. s (for velocity reconstruction).
    const double dh00 =  6*s2 - 6*s;
    const double dh10 =  3*s2 - 4*s + 1;
    const double dh01 = -6*s2 + 6*s;
    const double dh11 =  3*s2 - 2*s;

    const double* nk  = traj + k       * 2 * NUM_JOINTS;
    const double* nk1 = traj + (k + 1) * 2 * NUM_JOINTS;

    for (int j = 0; j < NUM_JOINTS; ++j) {
        const double q0 = nk[j],              v0 = nk[NUM_JOINTS + j];
        const double q1 = nk1[j],             v1 = nk1[NUM_JOINTS + j];
        q_des[j]  = h00*q0 + h10*dt_node_*v0 + h01*q1 + h11*dt_node_*v1;
        dq_des[j] = (dh00*q0 + dh10*dt_node_*v0 + dh01*q1 + dh11*dt_node_*v1) / dt_node_;
    }
}

// ============================================================================
// Muscle inverse — replaces PD controller (Sec. III-A, Eq. 13 analogue)
//
// Given desired joint target (q_des, dq_des), computes the agonist/antagonist
// activations that produce the required impedance torque via the Hill model.
// ============================================================================

void MPPILocomotion::activations_from_target(int j,
                                              double q_des, double dq_des,
                                              double q_cur, double dq_cur,
                                              double& act1_out, double& act2_out) const
{
    static constexpr double eps = 1e-6;
    const MuscleParams& p = muscle_;

    // Moment arm (matches muscle.h).
    const double r1 = (p.lce_max[j] - p.lce_min[j] + eps)
                    / (p.phi_max[j]  - p.phi_min[j]  + eps);

    // Fiber lengths and velocities at current state.
    const double lce1     =  q_cur * r1 + (p.lce_min[j] - r1 * p.phi_min[j]);
    const double lce2     = -q_cur * r1 + (p.lce_min[j] + r1 * p.phi_max[j]);
    const double lce_dot1 =  r1 * dq_cur;
    const double lce_dot2 = -r1 * dq_cur;

    // Active force-length (including secondary shoulder — matches muscle.h).
    const double FL1 = active_force_length(lce1, p.lce_min[j], 1.0, p.lce_max[j])
                     + 0.15 * active_force_length(lce1, p.lce_min[j],
                                                   0.5*(p.lce_min[j] + 0.95), 0.95);
    const double FL2 = active_force_length(lce2, p.lce_min[j], 1.0, p.lce_max[j])
                     + 0.15 * active_force_length(lce2, p.lce_min[j],
                                                   0.5*(p.lce_min[j] + 0.95), 0.95);

    // Force-velocity.
    const double c   = p.FVmax[j] - 1.0;
    const double FV1 = force_vel(lce_dot1, c, p.vmax[j], p.FVmax[j]);
    const double FV2 = force_vel(lce_dot2, c, p.vmax[j], p.FVmax[j]);

    // Passive parallel elasticity.
    const double b    = 0.5 * (p.lce_max[j] + 1.0);
    const double pFL1 = passive_force_length(lce1, p.pFLmax[j], b);
    const double pFL2 = passive_force_length(lce2, p.pFLmax[j], b);

    // Desired impedance torque (replaces Kp*(q_des - q_t) + Kd*(dq_des - v_t)).
    const double tau_des = task_.kp_muscle[j] * (q_des - q_cur)
                         + task_.kd_muscle[j] * (dq_des - dq_cur);

    // Passive torque already produced: τ_passive = r1*(pFL2 - pFL1)*peak_force
    const double tau_passive = r1 * (pFL2 - pFL1) * p.peak_force[j];

    // Active torque required: τ_active = r1*(act2*FL2*FV2 - act1*FL1*FV1)*peak_force
    const double tau_active = tau_des - tau_passive;

    if (tau_active >= 0.0) {
        // Antagonist (muscle 2) contracts.
        act1_out = 0.0;
        act2_out = std::clamp(tau_active / (r1 * FL2 * FV2 * p.peak_force[j] + eps),
                              0.0, 1.0);
    } else {
        // Agonist (muscle 1) contracts.
        act2_out = 0.0;
        act1_out = std::clamp(-tau_active / (r1 * FL1 * FV1 * p.peak_force[j] + eps),
                              0.0, 1.0);
    }
}

// ============================================================================
// Clamp spline nodes (Eq. 5 — derivative constraint)
// ============================================================================

void MPPILocomotion::clamp_spline_nodes(std::vector<double>& nodes) const
{
    for (int k = 0; k < n_nodes_; ++k) {
        for (int j = 0; j < NUM_JOINTS; ++j) {
            double& q = nodes[k * 2 * NUM_JOINTS + j];
            double& v = nodes[k * 2 * NUM_JOINTS + NUM_JOINTS + j];

            q = std::clamp(q, task_.muscle.phi_min[j], task_.muscle.phi_max[j]);

            // |v| ≤ min(q_max − q, q − q_min) / (Δt_node / 2)  (Eq. 5)
            const double margin = std::min(task_.muscle.phi_max[j] - q,
                                           q - task_.muscle.phi_min[j]);
            const double v_lim  = std::max(margin / (dt_node_ * 0.5), 0.0);
            v = std::clamp(v, -v_lim, v_lim);
        }
    }
}

// ============================================================================
// Spline-aware noise sampling
// ============================================================================

void MPPILocomotion::sample_spline_noise(int iter)
{
    const int NU = NUM_JOINTS;
    const int stride = n_nodes_ * 2 * NU;
    for (int s = 0; s < task_.n_samples; ++s) {
        for (int k = 0; k < n_nodes_; ++k) {
            const double anneal = noise_sched_[iter * n_nodes_ + k];
            for (int j = 0; j < NU; ++j) {
                noise_[s * stride + k * 2 * NU + j]      =
                    task_.noise_sigma_q[j] * anneal * normal_(rng_);
                noise_[s * stride + k * 2 * NU + NU + j] =
                    task_.noise_sigma_v[j] * anneal * normal_(rng_);
            }
        }
    }
}

// ============================================================================
// Rollout
// ============================================================================

double MPPILocomotion::rollout(int s, const RobotState& state)
{
    using Clock = std::chrono::steady_clock;
    using Us    = std::chrono::microseconds;

    mjData* d = data_[s];
    set_mj_state(d, state);

    double activation[NUM_MUSCLES];
    std::memcpy(activation, predicted_activation_, NUM_MUSCLES * sizeof(double));

    // Build noisy spline for this sample.
    const int NU = NUM_JOINTS;
    const int stride = n_nodes_ * 2 * NU;
    std::vector<double> noisy(stride);
    for (int k = 0; k < n_nodes_; ++k) {
        for (int j = 0; j < NU; ++j) {
            noisy[k * 2 * NU + j]      = trajectory_[k * 2 * NU + j]
                                        + noise_[s * stride + k * 2 * NU + j];
            noisy[k * 2 * NU + NU + j] = trajectory_[k * 2 * NU + NU + j]
                                        + noise_[s * stride + k * 2 * NU + NU + j];
        }
    }
    clamp_spline_nodes(noisy);

    double total_cost = 0.0;

    for (int t = 0; t < task_.horizon; ++t) {
        const double t_sec = t * task_.substeps * task_.dt;
        double q_des[NUM_JOINTS], dq_des[NUM_JOINTS];
        hermite_eval(noisy.data(), t_sec, q_des, dq_des);

        double tau_out[NUM_JOINTS];
        for (int sub = 0; sub < task_.substeps; ++sub) {
            double q_cur[NUM_JOINTS], dq_cur[NUM_JOINTS];
            for (int j = 0; j < NUM_JOINTS; ++j) {
                q_cur[j]  = d->qpos[act_qpos_adr_[j]];
                dq_cur[j] = d->qvel[act_qvel_adr_[j]];
            }

            auto t_h = Clock::now();
            double act_cmd[NUM_MUSCLES];
            for (int j = 0; j < NUM_JOINTS; ++j)
                activations_from_target(j, q_des[j], dq_des[j],
                                         q_cur[j], dq_cur[j],
                                         act_cmd[2*j], act_cmd[2*j+1]);
            hill_compute_torques(act_cmd, q_cur, dq_cur, muscle_, task_.dt, activation, tau_out);
            lat_hill_us_.fetch_add(
                std::chrono::duration_cast<Us>(Clock::now() - t_h).count(),
                std::memory_order_relaxed);

            for (int j = 0; j < model_->nu; ++j)
                d->ctrl[j] = 0.0;
            for (int j = 0; j < NUM_JOINTS; ++j)
                d->ctrl[JOINT_OFFSET + j] = tau_out[j];

            auto t_mj = Clock::now();
            mj_step(model_, d);
            lat_mjstep_us_.fetch_add(
                std::chrono::duration_cast<Us>(Clock::now() - t_mj).count(),
                std::memory_order_relaxed);

            if (!std::isfinite(d->qpos[2]) || d->qpos[2] < -1.0)
                return 1e6;
        }

        auto t_c = Clock::now();
        total_cost += step_cost(d, t);
        lat_cost_us_.fetch_add(
            std::chrono::duration_cast<Us>(Clock::now() - t_c).count(),
            std::memory_order_relaxed);
    }

    auto t_tc = Clock::now();
    total_cost += terminal_cost(d);
    lat_cost_us_.fetch_add(
        std::chrono::duration_cast<Us>(Clock::now() - t_tc).count(),
        std::memory_order_relaxed);

    return std::isfinite(total_cost) ? total_cost : 1e6;
}

// ============================================================================
// Cost function (Eq. 17, 18)
// ============================================================================

double MPPILocomotion::step_cost(const mjData* d, int /*horizon_step*/)
{
    const CostWeights& w = task_.cost;
    double cost = 0.0;

    // Base height.
    cost += w.height * std::abs(d->xpos[base_bid_ * 3 + 2] - height_target_);

    // Orientation: squared tilt angle from upright.
    const double qw    = d->qpos[3];
    const double angle = 2.0 * std::acos(std::clamp(std::abs(qw), 0.0, 1.0));
    cost += w.orientation * angle * angle;

    // Posture deviation from nominal.
    if (w.posture > 0.0) {
        for (int j = 0; j < NUM_JOINTS; ++j) {
            double dq = d->qpos[act_qpos_adr_[j]] - task_.nominal_pose[j];
            cost += w.posture * dq * dq;
        }
    }

    // Contact velocity and force over tracked feet.
    if (w.contact_vel > 0.0 || w.contact_force > 0.0) {
        for (int fi = 0; fi < n_feet_; ++fi) {
            const int bid = foot_body_ids_[fi];

            if (w.contact_vel > 0.0) {
                mjtNum vel6[6];
                mj_objectVelocity(model_, d, mjOBJ_BODY, bid, vel6, 0);
                const double speed = std::sqrt(vel6[3]*vel6[3] + vel6[4]*vel6[4] + vel6[5]*vel6[5]);
                cost += w.contact_vel * speed;
            }

            if (w.contact_force > 0.0) {
                const double fx = d->cfrc_ext[6*bid + 3];
                const double fy = d->cfrc_ext[6*bid + 4];
                const double fz = d->cfrc_ext[6*bid + 5];
                cost += w.contact_force * (std::abs(fx) + std::abs(fy)
                                          + std::abs(fz - f_nominal_));
            }
        }
    }

    // Velocity tracking.
    if (w.vel_cmd > 0.0) {
        const double dvx = d->qvel[0] - w.vel_des[0];
        const double dvy = d->qvel[1] - w.vel_des[1];
        cost += w.vel_cmd * (dvx*dvx + dvy*dvy);
    }

    return cost;
}

// Terminal cost: base displacement from velocity-integrated target (Eq. 18).
double MPPILocomotion::terminal_cost(const mjData* d)
{
    const double horizon_secs = task_.horizon * task_.substeps * task_.dt;
    const double px_target = start_pos_[0] + cmd_.vx * horizon_secs;
    const double py_target = start_pos_[1] + cmd_.vy * horizon_secs;
    return task_.cost.terminal * (std::abs(d->qpos[0] - px_target)
                                 + std::abs(d->qpos[1] - py_target));
}

// ============================================================================
// Predict state (latency compensation — Sec. III-E, Eq. 14)
// ============================================================================

RobotState MPPILocomotion::predict_state(const RobotState& state, int n_steps)
{
    mjData* d = data_[task_.n_samples];
    set_mj_state(d, state);

    double activation[NUM_MUSCLES];
    std::memcpy(activation, muscle_state_.activation, NUM_MUSCLES * sizeof(double));

    for (int t = 0; t < n_steps; ++t) {
        const double t_sec = t * task_.substeps * task_.dt;
        double q_des[NUM_JOINTS], dq_des[NUM_JOINTS];
        hermite_eval(best_trajectory_.data(), t_sec, q_des, dq_des);

        double tau_out[NUM_JOINTS];
        for (int sub = 0; sub < task_.substeps; ++sub) {
            double q_cur[NUM_JOINTS], dq_cur[NUM_JOINTS];
            for (int j = 0; j < NUM_JOINTS; ++j) {
                q_cur[j]  = d->qpos[act_qpos_adr_[j]];
                dq_cur[j] = d->qvel[act_qvel_adr_[j]];
            }
            double act_cmd[NUM_MUSCLES];
            for (int j = 0; j < NUM_JOINTS; ++j)
                activations_from_target(j, q_des[j], dq_des[j],
                                         q_cur[j], dq_cur[j],
                                         act_cmd[2*j], act_cmd[2*j+1]);
            hill_compute_torques(act_cmd, q_cur, dq_cur, muscle_, task_.dt, activation, tau_out);
            for (int j = 0; j < model_->nu; ++j)
                d->ctrl[j] = 0.0;
            for (int j = 0; j < NUM_JOINTS; ++j)
                d->ctrl[JOINT_OFFSET + j] = tau_out[j];
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
// Main solve (Algorithm 1)
// ============================================================================

void MPPILocomotion::update(const RobotState& state,
                             double q_des_out[NUM_JOINTS],
                             double dq_des_out[NUM_JOINTS])
{
    auto t_start = std::chrono::steady_clock::now();
    using Clock = std::chrono::steady_clock;
    using Us    = std::chrono::microseconds;
    auto elapsed_us = [](auto t0) {
        return std::chrono::duration_cast<Us>(Clock::now() - t0).count();
    };

    if (!state.valid) {
        for (int j = 0; j < NUM_JOINTS; ++j) {
            q_des_out[j]  = best_trajectory_[j];
            dq_des_out[j] = best_trajectory_[NUM_JOINTS + j];
        }
        return;
    }

    lat_hill_us_.store(0,   std::memory_order_relaxed);
    lat_mjstep_us_.store(0, std::memory_order_relaxed);
    lat_cost_us_.store(0,   std::memory_order_relaxed);

    // Latency compensation: n_skip in horizon steps (Eq. 14).
    const double dt_step = task_.substeps * task_.dt;
    const int n_skip = std::clamp(
        static_cast<int>(std::round(last_compute_ms_ * 1e-3 / dt_step)),
        1, task_.horizon / 2);

    auto t_pred = Clock::now();
    RobotState predicted = predict_state(state, n_skip);
    long long pred_us = elapsed_us(t_pred);

    start_pos_[0] = predicted.pos[0];
    start_pos_[1] = predicted.pos[1];
    start_pos_[2] = predicted.pos[2];

    // Warm-start: shift spline nodes forward by n_skip steps (Eq. 15).
    // Convert horizon-step skip to node skip.
    auto t_ws = Clock::now();
    const int n_skip_nodes = std::clamp(
        static_cast<int>(std::round(n_skip * dt_step / dt_node_)),
        1, n_nodes_ - 1);

    const int NU     = NUM_JOINTS;
    const int stride = n_nodes_ * 2 * NU;
    std::vector<double> shifted(stride);
    for (int k = 0; k < n_nodes_ - n_skip_nodes; ++k)
        for (int d = 0; d < 2 * NU; ++d)
            shifted[k * 2 * NU + d] =
                best_trajectory_[(k + n_skip_nodes) * 2 * NU + d];
    // Pad tail: hold last position, zero velocity.
    for (int k = n_nodes_ - n_skip_nodes; k < n_nodes_; ++k)
        for (int j = 0; j < NU; ++j) {
            shifted[k * 2 * NU + j]      = best_trajectory_[(n_nodes_-1) * 2 * NU + j];
            shifted[k * 2 * NU + NU + j] = 0.0;
        }
    trajectory_ = shifted;
    long long ws_us = elapsed_us(t_ws);

    // MPPI iterations.
    const int N = task_.n_iterations;
    best_cost_ = 1e9;

    auto t_iter = Clock::now();
    for (int iter = 0; iter < N; ++iter) {
        sample_spline_noise(iter);

        #pragma omp parallel for schedule(dynamic)
        for (int s = 0; s < task_.n_samples; ++s)
            costs_[s] = rollout(s, predicted);

        // Track best sample (Sec. III-D).
        for (int s = 0; s < task_.n_samples; ++s) {
            if (costs_[s] < best_cost_) {
                best_cost_ = costs_[s];
                for (int k = 0; k < n_nodes_; ++k) {
                    for (int j = 0; j < NU; ++j) {
                        double q_noisy = trajectory_[k*2*NU+j]    + noise_[s*stride + k*2*NU+j];
                        double v_noisy = trajectory_[k*2*NU+NU+j] + noise_[s*stride + k*2*NU+NU+j];
                        best_trajectory_[k*2*NU+j]    = std::clamp(q_noisy,
                            task_.muscle.phi_min[j], task_.muscle.phi_max[j]);
                        best_trajectory_[k*2*NU+NU+j] = v_noisy;  // clamped below
                    }
                }
                clamp_spline_nodes(best_trajectory_);
            }
        }

        // Softmin weights over min-max normalised costs (Eq. 10, 11).
        double cost_min   = *std::min_element(costs_.begin(), costs_.end());
        double cost_max   = *std::max_element(costs_.begin(), costs_.end());
        double cost_range = cost_max - cost_min;

        std::vector<double> weights(task_.n_samples);
        double weight_sum = 0.0;
        for (int s = 0; s < task_.n_samples; ++s) {
            double s_hat  = (cost_range > 1e-12) ? (costs_[s] - cost_min) / cost_range : 0.0;
            weights[s]    = std::exp(-s_hat / task_.lambda);
            weight_sum   += weights[s];
        }

        // Update nominal trajectory as weighted average (Eq. 12).
        std::vector<double> new_traj(stride, 0.0);
        for (int s = 0; s < task_.n_samples; ++s) {
            const double w = weights[s] / weight_sum;
            for (int k = 0; k < n_nodes_; ++k)
                for (int d = 0; d < 2 * NU; ++d) {
                    int idx = k * 2 * NU + d;
                    new_traj[idx] += w * (trajectory_[idx] + noise_[s * stride + idx]);
                }
        }
        clamp_spline_nodes(new_traj);
        trajectory_ = std::move(new_traj);
    }

    long long iter_us = elapsed_us(t_iter);

    // Return first node of best trajectory as joint targets (Eq. 13 analogue).
    for (int j = 0; j < NUM_JOINTS; ++j) {
        q_des_out[j]  = best_trajectory_[j];
        dq_des_out[j] = best_trajectory_[NUM_JOINTS + j];
    }

    last_compute_ms_ = std::chrono::duration<double, std::milli>(
        Clock::now() - t_start).count();

    if (lat_log_.is_open()) {
        const long long n_rollouts = (long long)task_.n_iterations * task_.n_samples;
        const double avg_hill_ms   = lat_hill_us_.load(std::memory_order_relaxed)   * 1e-3 / n_rollouts;
        const double avg_mjstep_ms = lat_mjstep_us_.load(std::memory_order_relaxed) * 1e-3 / n_rollouts;
        const double avg_cost_ms   = lat_cost_us_.load(std::memory_order_relaxed)   * 1e-3 / n_rollouts;

        lat_log_ << ++lat_call_count_  << ","
                 << last_compute_ms_   << ","
                 << pred_us  * 1e-3    << ","
                 << ws_us    * 1e-3    << ","
                 << iter_us  * 1e-3    << ","
                 << avg_hill_ms        << ","
                 << avg_mjstep_ms      << ","
                 << avg_cost_ms        << "\n";
        lat_log_.flush();
    }
}

// ============================================================================
// Compute real torques (muscle inverse + Hill model)
// ============================================================================

void MPPILocomotion::compute_real_torques(const RobotState& state,
                                           const double q_des[NUM_JOINTS],
                                           const double dq_des[NUM_JOINTS],
                                           double tau_out[NUM_JOINTS])
{
    double act_cmd[NUM_MUSCLES];
    for (int j = 0; j < NUM_JOINTS; ++j)
        activations_from_target(j, q_des[j], dq_des[j],
                                 state.q[j], state.dq[j],
                                 act_cmd[2*j], act_cmd[2*j+1]);

    hill_compute_torques(act_cmd, state.q, state.dq, muscle_, task_.dt,
                         muscle_state_.activation, tau_out);
}

// ============================================================================
// Cost diagnosis (for debugging — mirrors rollout cost on best trajectory)
// ============================================================================

MPPILocomotion::CostBreakdown
MPPILocomotion::diagnose_cost(const RobotState& state)
{
    mjData* d = data_[task_.n_samples];
    set_mj_state(d, state);

    double activation[NUM_MUSCLES];
    std::memcpy(activation, predicted_activation_, NUM_MUSCLES * sizeof(double));

    CostBreakdown bd;
    const CostWeights& w = task_.cost;

    for (int t = 0; t < task_.horizon; ++t) {
        const double t_sec = t * task_.substeps * task_.dt;
        double q_des[NUM_JOINTS], dq_des[NUM_JOINTS];
        hermite_eval(best_trajectory_.data(), t_sec, q_des, dq_des);

        double tau_out[NUM_JOINTS];
        for (int sub = 0; sub < task_.substeps; ++sub) {
            double q_cur[NUM_JOINTS], dq_cur[NUM_JOINTS];
            for (int j = 0; j < NUM_JOINTS; ++j) {
                q_cur[j]  = d->qpos[act_qpos_adr_[j]];
                dq_cur[j] = d->qvel[act_qvel_adr_[j]];
            }
            double act_cmd[NUM_MUSCLES];
            for (int j = 0; j < NUM_JOINTS; ++j)
                activations_from_target(j, q_des[j], dq_des[j],
                                         q_cur[j], dq_cur[j],
                                         act_cmd[2*j], act_cmd[2*j+1]);
            hill_compute_torques(act_cmd, q_cur, dq_cur, muscle_, task_.dt, activation, tau_out);
            for (int j = 0; j < model_->nu; ++j)
                d->ctrl[j] = 0.0;
            for (int j = 0; j < NUM_JOINTS; ++j)
                d->ctrl[JOINT_OFFSET + j] = tau_out[j];
            mj_step(model_, d);
        }

        bd.height      += w.height * std::abs(d->xpos[base_bid_*3+2] - height_target_);
        const double qw = d->qpos[3];
        const double ang = 2.0 * std::acos(std::clamp(std::abs(qw), 0.0, 1.0));
        bd.orientation += w.orientation * ang * ang;
        if (w.posture > 0.0)
            for (int j = 0; j < NUM_JOINTS; ++j) {
                double dq = d->qpos[act_qpos_adr_[j]] - task_.nominal_pose[j];
                bd.posture += w.posture * dq * dq;
            }
        for (int fi = 0; fi < n_feet_; ++fi) {
            const int bid = foot_body_ids_[fi];
            if (w.contact_vel > 0.0) {
                mjtNum vel6[6];
                mj_objectVelocity(model_, d, mjOBJ_BODY, bid, vel6, 0);
                bd.contact_vel += w.contact_vel * std::sqrt(
                    vel6[3]*vel6[3] + vel6[4]*vel6[4] + vel6[5]*vel6[5]);
            }
            if (w.contact_force > 0.0) {
                double fx = d->cfrc_ext[6*bid+3], fy = d->cfrc_ext[6*bid+4], fz = d->cfrc_ext[6*bid+5];
                bd.contact_force += w.contact_force * (std::abs(fx) + std::abs(fy) + std::abs(fz - f_nominal_));
            }
        }
        if (w.vel_cmd > 0.0) {
            double dvx = d->qvel[0] - w.vel_des[0], dvy = d->qvel[1] - w.vel_des[1];
            bd.vel_tracking += w.vel_cmd * (dvx*dvx + dvy*dvy);
        }
    }

    const double horizon_secs = task_.horizon * task_.substeps * task_.dt;
    bd.terminal = task_.cost.terminal * (
        std::abs(d->qpos[0] - (start_pos_[0] + cmd_.vx * horizon_secs))
      + std::abs(d->qpos[1] - (start_pos_[1] + cmd_.vy * horizon_secs)));

    return bd;
}
