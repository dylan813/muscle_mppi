#include "mppi_locomotion.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <stdexcept>
#include <iostream>

MPPILocomotion::MPPILocomotion(const std::string& task_name, const std::string& yaml_path)
    : BaseMPPI(load_task(task_name, yaml_path))
{
    muscle_ = task_.muscle;
    best_trajectory_.assign(task_.horizon * NUM_MUSCLES, 0.0);

    for (const char* name : {"trunk", "base", "base_link"}) {
        int bid = mj_name2id(model_, mjOBJ_BODY, name);
        if (bid >= 0) { base_bid_ = bid; break; }
    }

    foot_body_ids_[0] = mj_name2id(model_, mjOBJ_BODY, "FL_foot");
    if (foot_body_ids_[0] < 0)
        throw std::runtime_error("Body not found: FL_foot");

    double total_mass = 0.0;
    for (int i = 1; i < model_->nbody; ++i)
        total_mass += model_->body_mass[i];
    f_nominal_ = total_mass * std::abs(model_->opt.gravity[2]) / 4.0;

    cmd_.vx = task_.cost.vel_des[0];
    cmd_.vy = task_.cost.vel_des[1];
    cmd_.wz = task_.cost.vel_des[2];
}


double MPPILocomotion::step_cost(const mjData* d,
                                  const double act_cmd[NUM_MUSCLES],
                                  int horizon_step)
{
    const CostWeights& w = task_.cost;
    double cost = 0.0;

    cost += w.height * std::abs(d->xpos[base_bid_ * 3 + 2] - height_target_);

    double qw    = d->qpos[3];
    double angle = 2.0 * std::acos(std::clamp(std::abs(qw), 0.0, 1.0));
    cost += w.orientation * angle * angle;

    if (w.posture > 0.0) {
        for (int j = 0; j < NUM_JOINTS; ++j) {
            double dq = d->qpos[act_qpos_adr_[j]] - task_.nominal_pose[j];
            cost += w.posture * dq * dq;
        }
    }

    if (w.foot_pos > 0.0) {
        const double* fp = d->xpos + 3 * foot_body_ids_[0];
        double dx = fp[0] - task_.foot_target[0];
        double dy = fp[1] - task_.foot_target[1];
        double dz = fp[2] - task_.foot_target[2];
        cost += w.foot_pos * std::sqrt(dx*dx + dy*dy + dz*dz);
    }

    if (w.contact_vel > 0.0) {
        mjtNum vel6[6];
        mj_objectVelocity(model_, d, mjOBJ_BODY, foot_body_ids_[0], vel6, 0);
        double speed = std::sqrt(vel6[3]*vel6[3] + vel6[4]*vel6[4] + vel6[5]*vel6[5]);
        cost += w.contact_vel * speed;
    }

    if (w.contact_force > 0.0) {
        int bid = foot_body_ids_[0];
        double fx = d->cfrc_ext[6*bid + 3];
        double fy = d->cfrc_ext[6*bid + 4];
        double fz = d->cfrc_ext[6*bid + 5];
        cost += w.contact_force * (std::abs(fx) + std::abs(fy) + std::abs(fz - f_nominal_));
    }

    if (w.vel_cmd > 0.0) {
        double dvx = d->qvel[0] - w.vel_des[0];
        double dvy = d->qvel[1] - w.vel_des[1];
        cost += w.vel_cmd * (dvx * dvx + dvy * dvy);
    }

    return cost;
}

double MPPILocomotion::terminal_cost(const mjData* d)
{
    const double* fp = d->xpos + 3 * foot_body_ids_[0];
    double dx = fp[0] - task_.foot_target[0];
    double dy = fp[1] - task_.foot_target[1];
    double dz = fp[2] - task_.foot_target[2];
    return task_.cost.terminal * std::sqrt(dx*dx + dy*dy + dz*dz);
}

double MPPILocomotion::rollout(int s, const RobotState& state)
{
    mjData* d = data_[s];
    set_mj_state(d, state);

    double activation[NUM_MUSCLES];
    std::memcpy(activation, predicted_activation_, NUM_MUSCLES * sizeof(double));

    double total_cost = 0.0;

    for (int t = 0; t < task_.horizon; ++t) {
        double act_cmd[NUM_MUSCLES];
        for (int m = 0; m < NUM_MUSCLES; ++m) {
            double nom   = trajectory_[t * NUM_MUSCLES + m];
            double noisy = nom + noise_[s * task_.horizon * NUM_MUSCLES
                                        + t * NUM_MUSCLES + m];
            act_cmd[m] = std::clamp(noisy, ACT_MIN, ACT_MAX);
        }

        double tau_out[NUM_JOINTS];
        double effort_accum = 0.0;
        for (int sub = 0; sub < task_.substeps; ++sub) {
            double q_cur[NUM_JOINTS], dq_cur[NUM_JOINTS];
            for (int j = 0; j < NUM_JOINTS; ++j) {
                q_cur[j]  = d->qpos[act_qpos_adr_[j]];
                dq_cur[j] = d->qvel[act_qvel_adr_[j]];
            }
            hill_compute_torques(act_cmd, q_cur, dq_cur, muscle_, task_.dt, activation, tau_out);

            if (task_.cost.act_effort > 0.0) {
                for (int j = 0; j < NUM_JOINTS; ++j)
                    effort_accum += (activation[2*j]   * activation[2*j]
                                   + activation[2*j+1] * activation[2*j+1])
                                   * muscle_.peak_force[j];
            }

            for (int j = 0; j < model_->nu; ++j)
                d->ctrl[j] = 0.0;
            for (int j = 0; j < NUM_JOINTS; ++j)
                d->ctrl[JOINT_OFFSET + j] = tau_out[j];

            mj_step(model_, d);

            if (!std::isfinite(d->qpos[2]) || d->qpos[2] < -1.0)
                return 1e6;
        }

        total_cost += task_.cost.act_effort * effort_accum / task_.substeps;
        total_cost += step_cost(d, act_cmd, t);
    }

    total_cost += terminal_cost(d);
    return std::isfinite(total_cost) ? total_cost : 1e6;
}

RobotState MPPILocomotion::predict_state(const RobotState& state, int n_steps)
{
    mjData* d = data_[task_.n_samples];
    set_mj_state(d, state);

    double activation[NUM_MUSCLES];
    std::memcpy(activation, muscle_state_.activation, NUM_MUSCLES * sizeof(double));

    for (int t = 0; t < n_steps; ++t) {
        double act_cmd[NUM_MUSCLES];
        for (int m = 0; m < NUM_MUSCLES; ++m)
            act_cmd[m] = best_trajectory_[t * NUM_MUSCLES + m];

        double tau_out[NUM_JOINTS];
        for (int sub = 0; sub < task_.substeps; ++sub) {
            double q_cur[NUM_JOINTS], dq_cur[NUM_JOINTS];
            for (int j = 0; j < NUM_JOINTS; ++j) {
                q_cur[j]  = d->qpos[act_qpos_adr_[j]];
                dq_cur[j] = d->qvel[act_qvel_adr_[j]];
            }
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
        predicted.pos[0]  = d->qpos[0];  predicted.pos[1]  = d->qpos[1];  predicted.pos[2]  = d->qpos[2];
        predicted.quat[0] = d->qpos[3];  predicted.quat[1] = d->qpos[4];
        predicted.quat[2] = d->qpos[5];  predicted.quat[3] = d->qpos[6];
        predicted.vel[0]  = d->qvel[0];  predicted.vel[1]  = d->qvel[1];  predicted.vel[2]  = d->qvel[2];
        predicted.gyro[0] = d->qvel[3];  predicted.gyro[1] = d->qvel[4];  predicted.gyro[2] = d->qvel[5];
    }
    for (int j = 0; j < NUM_JOINTS; ++j) {
        predicted.q[j]  = d->qpos[act_qpos_adr_[j]];
        predicted.dq[j] = d->qvel[act_qvel_adr_[j]];
    }
    predicted.valid = true;
    std::memcpy(predicted_activation_, activation, NUM_MUSCLES * sizeof(double));
    return predicted;
}

void MPPILocomotion::update(const RobotState& state, double activations_out[NUM_MUSCLES])
{
    auto t_start = std::chrono::steady_clock::now();

    if (!state.valid) {
        for (int m = 0; m < NUM_MUSCLES; ++m)
            activations_out[m] = best_trajectory_[m];
        return;
    }

    const double dt_step = task_.substeps * task_.dt;
    const int n_skip = std::clamp(
        static_cast<int>(std::round(last_compute_ms_ * 1e-3 / dt_step)),
        1, task_.horizon / 2);

    RobotState predicted = predict_state(state, n_skip);

    start_pos_[0] = predicted.pos[0];
    start_pos_[1] = predicted.pos[1];
    start_pos_[2] = predicted.pos[2];

    for (int t = 0; t < task_.horizon - n_skip; ++t)
        for (int m = 0; m < NUM_MUSCLES; ++m)
            trajectory_[t * NUM_MUSCLES + m] =
                best_trajectory_[(t + n_skip) * NUM_MUSCLES + m];
    for (int t = task_.horizon - n_skip; t < task_.horizon; ++t)
        for (int m = 0; m < NUM_MUSCLES; ++m)
            trajectory_[t * NUM_MUSCLES + m] =
                best_trajectory_[(task_.horizon - 1) * NUM_MUSCLES + m];

    const int N = task_.n_iterations;
    best_cost_ = 1e9;

    for (int iter = 0; iter < N; ++iter) {
        sample_noise(iter, N);

        #pragma omp parallel for schedule(dynamic)
        for (int s = 0; s < task_.n_samples; ++s)
            costs_[s] = rollout(s, predicted);

        for (int s = 0; s < task_.n_samples; ++s) {
            if (costs_[s] < best_cost_) {
                best_cost_ = costs_[s];
                for (int t = 0; t < task_.horizon; ++t)
                    for (int m = 0; m < NUM_MUSCLES; ++m) {
                        int idx = t * NUM_MUSCLES + m;
                        best_trajectory_[idx] = std::clamp(
                            trajectory_[idx] + noise_[s * task_.horizon * NUM_MUSCLES + idx],
                            ACT_MIN, ACT_MAX);
                    }
            }
        }

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

        std::vector<double> new_traj(task_.horizon * NUM_MUSCLES, 0.0);
        for (int s = 0; s < task_.n_samples; ++s) {
            double w = weights[s] / weight_sum;
            for (int t = 0; t < task_.horizon; ++t)
                for (int m = 0; m < NUM_MUSCLES; ++m) {
                    int idx = t * NUM_MUSCLES + m;
                    new_traj[idx] += w * (trajectory_[idx]
                        + noise_[s * task_.horizon * NUM_MUSCLES + idx]);
                }
        }
        for (auto& v : new_traj)
            v = std::clamp(v, ACT_MIN, ACT_MAX);

        trajectory_ = std::move(new_traj);
    }

    for (int m = 0; m < NUM_MUSCLES; ++m)
        activations_out[m] = best_trajectory_[m];

    last_compute_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_start).count();
}

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
        double act_cmd[NUM_MUSCLES];
        for (int m = 0; m < NUM_MUSCLES; ++m)
            act_cmd[m] = best_trajectory_[t * NUM_MUSCLES + m];

        double tau_out[NUM_JOINTS];
        double effort_accum = 0.0;
        for (int sub = 0; sub < task_.substeps; ++sub) {
            double q_cur[NUM_JOINTS], dq_cur[NUM_JOINTS];
            for (int j = 0; j < NUM_JOINTS; ++j) {
                q_cur[j]  = d->qpos[act_qpos_adr_[j]];
                dq_cur[j] = d->qvel[act_qvel_adr_[j]];
            }
            hill_compute_torques(act_cmd, q_cur, dq_cur, muscle_, task_.dt, activation, tau_out);
            if (w.act_effort > 0.0) {
                for (int j = 0; j < NUM_JOINTS; ++j)
                    effort_accum += (activation[2*j]   * activation[2*j]
                                   + activation[2*j+1] * activation[2*j+1])
                                   * muscle_.peak_force[j];
            }
            for (int j = 0; j < model_->nu; ++j)
                d->ctrl[j] = 0.0;
            for (int j = 0; j < NUM_JOINTS; ++j)
                d->ctrl[JOINT_OFFSET + j] = tau_out[j];
            mj_step(model_, d);
        }
        bd.act_effort += w.act_effort * effort_accum / task_.substeps;

        bd.height += w.height * std::abs(d->xpos[base_bid_ * 3 + 2] - height_target_);

        double qw    = d->qpos[3];
        double angle = 2.0 * std::acos(std::clamp(std::abs(qw), 0.0, 1.0));
        bd.orientation += w.orientation * angle * angle;

        if (w.posture > 0.0) {
            for (int j = 0; j < NUM_JOINTS; ++j) {
                double dq = d->qpos[act_qpos_adr_[j]] - task_.nominal_pose[j];
                bd.posture += w.posture * dq * dq;
            }
        }

        if (w.contact_vel > 0.0) {
            mjtNum vel6[6];
            mj_objectVelocity(model_, d, mjOBJ_BODY, foot_body_ids_[0], vel6, 0);
            double speed = std::sqrt(vel6[3]*vel6[3] + vel6[4]*vel6[4] + vel6[5]*vel6[5]);
            bd.contact_vel += w.contact_vel * speed;
        }

        if (w.contact_force > 0.0) {
            int bid = foot_body_ids_[0];
            double fx = d->cfrc_ext[6*bid + 3];
            double fy = d->cfrc_ext[6*bid + 4];
            double fz = d->cfrc_ext[6*bid + 5];
            bd.contact_force += w.contact_force
                * (std::abs(fx) + std::abs(fy) + std::abs(fz - f_nominal_));
        }

        if (w.vel_cmd > 0.0) {
            double dvx = d->qvel[0] - w.vel_des[0];
            double dvy = d->qvel[1] - w.vel_des[1];
            bd.vel_tracking += w.vel_cmd * (dvx * dvx + dvy * dvy);
        }
    }

    const double dt_step   = task_.dt * task_.substeps;
    const double px_target = start_pos_[0] + cmd_.vx * task_.horizon * dt_step;
    const double py_target = start_pos_[1] + cmd_.vy * task_.horizon * dt_step;
    bd.terminal = w.terminal * (std::abs(d->qpos[0] - px_target)
                                + std::abs(d->qpos[1] - py_target));

    return bd;
}

void MPPILocomotion::compute_real_torques(const RobotState& state,
                                          const double activations[NUM_MUSCLES],
                                          double tau_out[NUM_JOINTS])
{
    hill_compute_torques(activations, state.q, state.dq, muscle_, task_.dt,
                         muscle_state_.activation, tau_out);
}
