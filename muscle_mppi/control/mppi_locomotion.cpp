#include "mppi_locomotion.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <iostream>

MPPILocomotion::MPPILocomotion(const std::string& task_name, const std::string& yaml_path)
    : BaseMPPI(load_task(task_name, yaml_path))
{
    muscle_ = task_.muscle;
    best_trajectory_.assign(task_.horizon * NUM_JOINTS, 0.0);

    for (const char* name : {"trunk", "base", "base_link"}) {
        int bid = mj_name2id(model_, mjOBJ_BODY, name);
        if (bid >= 0) { base_bid_ = bid; break; }
    }

    static const char* WHEEL_BODY_NAMES[4] = {
        "FR_wheel_link", "FL_wheel_link", "RR_wheel_link", "RL_wheel_link"
    };
    for (int k = 0; k < 4; ++k) {
        wheel_body_ids_[k] = mj_name2id(model_, mjOBJ_BODY, WHEEL_BODY_NAMES[k]);
        if (wheel_body_ids_[k] < 0)
            throw std::runtime_error(std::string("Body not found: ") + WHEEL_BODY_NAMES[k]);
        const int jid = model_->actuator_trnid[2 * (NUM_LEG_JOINTS + k)];
        wheel_axle_[k][0] = model_->jnt_axis[3 * jid + 0];
        wheel_axle_[k][1] = model_->jnt_axis[3 * jid + 1];
        wheel_axle_[k][2] = model_->jnt_axis[3 * jid + 2];
    }

    double total_mass = 0.0;
    for (int i = 1; i < model_->nbody; ++i)
        total_mass += model_->body_mass[i];
    f_nominal_ = total_mass * std::abs(model_->opt.gravity[2]) / 4.0;

    cmd_.vx = task_.cost.vel_des[0];
    cmd_.vy = task_.cost.vel_des[1];
    cmd_.wz = task_.cost.vel_des[2];
}

void MPPILocomotion::load_reference(const std::string& csv_path, double ref_dt)
{
    std::ifstream f(csv_path);
    if (!f.is_open())
        throw std::runtime_error("load_reference: cannot open " + csv_path);

    reference_.clear();
    ref_steps_    = 0;
    ref_n_joints_ = 0;
    ref_dt_       = ref_dt;

    std::string line;
    if (std::getline(f, line)) {
        if (!line.empty() && (std::isalpha(line[0]) || line[0] == 'j')) {
            // skip header
        } else {
            // first data line — parse it and detect column count
            std::istringstream ss(line);
            std::string tok;
            while (std::getline(ss, tok, ','))
                reference_.push_back(std::stod(tok));
            ref_n_joints_ = static_cast<int>(reference_.size());
            ++ref_steps_;
        }
    }

    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ','))
            reference_.push_back(std::stod(tok));
        ++ref_steps_;
    }

    if (ref_steps_ == 0)
        throw std::runtime_error("load_reference: empty file " + csv_path);
    if (ref_n_joints_ < NUM_LEG_JOINTS)
        throw std::runtime_error("load_reference: CSV has only " + std::to_string(ref_n_joints_)
                                 + " columns, need at least " + std::to_string(NUM_LEG_JOINTS));

    std::cout << "Loaded activation reference: " << ref_steps_
              << " steps × " << ref_n_joints_ << " joints from " << csv_path << "\n";
}

double MPPILocomotion::step_cost(const mjData* d,
                                  const double act_cmd[NUM_JOINTS],
                                  int horizon_step)
{
    const CostWeights& w = task_.cost;
    double cost = 0.0;

    cost += w.height * std::abs(d->xpos[base_bid_ * 3 + 2] - height_target_);

    double qw    = d->qpos[3];
    double angle = 2.0 * std::acos(std::clamp(std::abs(qw), 0.0, 1.0));
    cost += w.orientation * angle * angle;

    if (w.posture > 0.0) {
        for (int j = 0; j < NUM_LEG_JOINTS; ++j) {
            double dq = d->qpos[act_qpos_adr_[j]] - task_.nominal_pose[j];
            cost += w.posture * dq * dq;
        }
    }

    // lateral slip: penalise velocity along wheel axle (rolling is free)
    if (w.contact_vel > 0.0) {
        for (int k = 0; k < 4; ++k) {
            mjtNum vel6[6];
            mj_objectVelocity(model_, d, mjOBJ_BODY, wheel_body_ids_[k], vel6, 0);
            const mjtNum* wmat = d->xmat + 9 * wheel_body_ids_[k];
            const double ax = wmat[0]*wheel_axle_[k][0] + wmat[1]*wheel_axle_[k][1] + wmat[2]*wheel_axle_[k][2];
            const double ay = wmat[3]*wheel_axle_[k][0] + wmat[4]*wheel_axle_[k][1] + wmat[5]*wheel_axle_[k][2];
            const double az = wmat[6]*wheel_axle_[k][0] + wmat[7]*wheel_axle_[k][1] + wmat[8]*wheel_axle_[k][2];
            const double slip = ax*vel6[3] + ay*vel6[4] + az*vel6[5];
            cost += w.contact_vel * std::abs(slip);
        }
    }

    if (w.contact_force > 0.0) {
        for (int k = 0; k < 4; ++k) {
            int bid = wheel_body_ids_[k];
            double fx = d->cfrc_ext[6*bid + 3];
            double fy = d->cfrc_ext[6*bid + 4];
            double fz = d->cfrc_ext[6*bid + 5];
            cost += w.contact_force * (std::abs(fx) + std::abs(fy)
                                      + std::abs(fz - f_nominal_));
        }
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
    const CostWeights& w = task_.cost;

    const double dt_step   = task_.dt * task_.substeps;
    const double px_target = start_pos_[0] + cmd_.vx * task_.horizon * dt_step;
    const double py_target = start_pos_[1] + cmd_.vy * task_.horizon * dt_step;

    return w.terminal * (std::abs(d->qpos[0] - px_target)
                        + std::abs(d->qpos[1] - py_target));
}

double MPPILocomotion::rollout(int s, const RobotState& state)
{
    mjData* d = data_[s];
    set_mj_state(d, state);

    double activation[NUM_JOINTS];
    std::memcpy(activation, predicted_activation_, sizeof(activation));

    double total_cost = 0.0;

    for (int t = 0; t < task_.horizon; ++t) {
        double act_cmd[NUM_JOINTS];
        for (int j = 0; j < NUM_LEG_JOINTS; ++j) {
            double nom   = trajectory_[t * NUM_JOINTS + j];
            double noisy = nom + noise_[s * task_.horizon * NUM_JOINTS
                                        + t * NUM_JOINTS + j];
            act_cmd[j] = std::clamp(noisy, ACT_MIN, ACT_MAX);
        }
        for (int j = NUM_LEG_JOINTS; j < NUM_JOINTS; ++j)
            act_cmd[j] = 0.0;

        double tau_out[NUM_JOINTS];
        double effort_accum  = 0.0;
        double ref_accum     = 0.0;
        const double* ref    = reference_at(t);
        for (int sub = 0; sub < task_.substeps; ++sub) {
            double q_cur[NUM_JOINTS], dq_cur[NUM_JOINTS];
            for (int j = 0; j < NUM_JOINTS; ++j) {
                q_cur[j]  = d->qpos[act_qpos_adr_[j]];
                dq_cur[j] = d->qvel[act_qvel_adr_[j]];
            }
            hill_compute_torques(act_cmd, q_cur, dq_cur, muscle_, activation, tau_out);

            if (task_.cost.act_effort > 0.0) {
                for (int j = 0; j < NUM_LEG_JOINTS; ++j)
                    effort_accum += activation[j] * activation[j] * muscle_.tau_max[j];
            }

            if (task_.cost.act_reference > 0.0 && ref) {
                for (int j = 0; j < NUM_LEG_JOINTS; ++j) {
                    double dt = (tau_out[j] - ref[j]) / muscle_.tau_max[j];
                    ref_accum += dt * dt;
                }
            }

            for (int j = 0; j < NUM_JOINTS; ++j)
                d->ctrl[j] = tau_out[j];

            mj_step(model_, d);

            if (!std::isfinite(d->qpos[2]) || d->qpos[2] < -1.0)
                return 1e6;
        }

        total_cost += task_.cost.act_effort    * effort_accum / task_.substeps;
        total_cost += task_.cost.act_reference * ref_accum    / task_.substeps;
        total_cost += step_cost(d, act_cmd, t);
    }

    total_cost += terminal_cost(d);
    return std::isfinite(total_cost) ? total_cost : 1e6;
}

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
                d->ctrl[j] = tau_out[j];
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

void MPPILocomotion::update(const RobotState& state, double activations_out[NUM_JOINTS])
{
    auto t_start = std::chrono::steady_clock::now();

    if (!state.valid) {
        for (int j = 0; j < NUM_JOINTS; ++j)
            activations_out[j] = best_trajectory_[j];
        return;
    }

    const double dt_step = task_.substeps * task_.dt;
    const int n_skip = std::clamp(
        static_cast<int>(std::round(last_compute_ms_ * 1e-3 / dt_step)),
        1, task_.horizon / 2);

    RobotState predicted = predict_state(state, n_skip);

    if (ref_steps_ > 0)
        ref_offset_ = (ref_offset_ + n_skip) % ref_steps_;

    start_pos_[0] = predicted.pos[0];
    start_pos_[1] = predicted.pos[1];
    start_pos_[2] = predicted.pos[2];

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

    for (int j = 0; j < NUM_JOINTS; ++j)
        activations_out[j] = best_trajectory_[j];

    last_compute_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_start).count();
}

MPPILocomotion::CostBreakdown
MPPILocomotion::diagnose_cost(const RobotState& state)
{
    mjData* d = data_[task_.n_samples];
    set_mj_state(d, state);

    double activation[NUM_JOINTS];
    std::memcpy(activation, predicted_activation_, sizeof(activation));

    CostBreakdown bd;
    const CostWeights& w = task_.cost;

    for (int t = 0; t < task_.horizon; ++t) {
        double act_cmd[NUM_JOINTS];
        for (int j = 0; j < NUM_JOINTS; ++j)
            act_cmd[j] = best_trajectory_[t * NUM_JOINTS + j];

        double tau_out[NUM_JOINTS];
        double effort_accum = 0.0;
        double ref_accum    = 0.0;
        const double* ref   = reference_at(t);
        for (int sub = 0; sub < task_.substeps; ++sub) {
            double q_cur[NUM_JOINTS], dq_cur[NUM_JOINTS];
            for (int j = 0; j < NUM_JOINTS; ++j) {
                q_cur[j]  = d->qpos[act_qpos_adr_[j]];
                dq_cur[j] = d->qvel[act_qvel_adr_[j]];
            }
            hill_compute_torques(act_cmd, q_cur, dq_cur, muscle_, activation, tau_out);
            if (w.act_effort > 0.0) {
                for (int j = 0; j < NUM_LEG_JOINTS; ++j)
                    effort_accum += activation[j] * activation[j] * muscle_.tau_max[j];
            }
            if (w.act_reference > 0.0 && ref) {
                for (int j = 0; j < NUM_LEG_JOINTS; ++j) {
                    double dt = (tau_out[j] - ref[j]) / muscle_.tau_max[j];
                    ref_accum += dt * dt;
                }
            }
            for (int j = 0; j < NUM_JOINTS; ++j)
                d->ctrl[j] = tau_out[j];
            mj_step(model_, d);
        }
        bd.act_effort += w.act_effort    * effort_accum / task_.substeps
                       + w.act_reference * ref_accum    / task_.substeps;

        bd.height += w.height * std::abs(d->xpos[base_bid_ * 3 + 2] - height_target_);

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
                                          const double activations[NUM_JOINTS],
                                          double tau_out[NUM_JOINTS])
{
    hill_compute_torques(activations, state.q, state.dq, muscle_,
                         muscle_state_.activation, tau_out);
    for (int j = NUM_LEG_JOINTS; j < NUM_JOINTS; ++j)
        tau_out[j] = 0.0;
}
