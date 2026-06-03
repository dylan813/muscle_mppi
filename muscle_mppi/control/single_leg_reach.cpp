#include "single_leg_reach.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

// -----------------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------------

SingleLegReach::SingleLegReach(const std::string& task_name,
                                const std::string& yaml_path,
                                const std::string& log_dir)
    : BaseMPPI(load_task(task_name, yaml_path))
{
    muscle_ = task_.muscle;

    for (int m = 0; m < NUM_MUSCLES; ++m) {
        action_lo_[m] = 0.0;
        action_hi_[m] = 1.0;
    }

    std::fill(best_traj_.begin(), best_traj_.end(), 0.0);

    // Load reach-specific config directly — these don't belong in the shared TaskConfig.
    {
        YAML::Node root = YAML::LoadFile(yaml_path);
        const YAML::Node& t = root[task_name];
        const YAML::Node& c = t["cost"];
        cost_foot_pos_  = c["foot_pos"]  ? c["foot_pos"].as<double>()  : 0.0;
        cost_joint_vel_ = c["joint_vel"] ? c["joint_vel"].as<double>() : 0.0;
        cost_terminal_  = c["terminal"]  ? c["terminal"].as<double>()  : 0.0;
        const YAML::Node& fts = t["foot_targets"];
        n_foot_targets_ = std::min((int)fts.size(), MAX_FOOT_TARGETS);
        for (int i = 0; i < n_foot_targets_; ++i)
            for (int k = 0; k < 3; ++k)
                foot_targets_[i][k] = fts[i][k].as<double>();
    }

    foot_body_id_ = mj_name2id(model_, mjOBJ_BODY, "FL_foot");
    if (foot_body_id_ < 0)
        throw std::runtime_error("Body not found: FL_foot");

    for (int k = 0; k < 3; ++k) active_target_[k] = foot_targets_[0][k];

    std::filesystem::create_directories(log_dir);
    lat_log_.open(log_dir + "/single_leg_latency.csv", std::ios::out | std::ios::trunc);
    lat_log_ << "call,total_ms,warm_start_ms,run_iterations_ms,final_hill_ms,"
                "avg_rollout_hill_ms,avg_rollout_mjstep_ms,avg_rollout_cost_ms\n";

    act_log_.open(log_dir + "/single_leg_activations.csv", std::ios::out | std::ios::trunc);
    act_log_ << "call,best_cost";
    for (int m = 0; m < NUM_MUSCLES; ++m) act_log_ << ",cmd_m" << m;
    for (int m = 0; m < NUM_MUSCLES; ++m) act_log_ << ",act_m" << m;
    for (int j = 0; j < NUM_JOINTS;  ++j) act_log_ << ",tau_j" << j;
    for (int m = 0; m < NUM_MUSCLES; ++m) act_log_ << ",samp_mean_m" << m << ",samp_std_m" << m;
    act_log_ << "\n";
}

// -----------------------------------------------------------------------------
// Cost
// -----------------------------------------------------------------------------

double SingleLegReach::step_cost(const mjData* d)
{
    double cost = 0.0;

    const double* fp = d->xpos + 3 * foot_body_id_;
    double dx = fp[0] - active_target_[0];
    double dy = fp[1] - active_target_[1];
    double dz = fp[2] - active_target_[2];
    cost += cost_foot_pos_ * (dx*dx + dy*dy + dz*dz);

    for (int j = 0; j < NUM_JOINTS; ++j) {
        double v = d->qvel[act_qvel_adr_[j]];
        cost += cost_joint_vel_ * v * v;
    }

    return cost;
}

double SingleLegReach::terminal_cost(const mjData* d)
{
    const double* fp = d->xpos + 3 * foot_body_id_;
    double dx = fp[0] - active_target_[0];
    double dy = fp[1] - active_target_[1];
    double dz = fp[2] - active_target_[2];
    return cost_terminal_ * (dx*dx + dy*dy + dz*dz);
}

void SingleLegReach::maybe_advance_target(double foot_err)
{
    if (n_foot_targets_ <= 1) return;
    if (foot_err < CONVERGENCE_THRESH)
        ++hold_count_;
    else
        hold_count_ = 0;
    if (hold_count_ >= HOLD_STEPS) {
        hold_count_ = 0;
        target_idx_ = (target_idx_ + 1) % n_foot_targets_;
        for (int k = 0; k < 3; ++k) active_target_[k] = foot_targets_[target_idx_][k];
        std::fill(best_traj_.begin(), best_traj_.end(), 0.0);
    }
}

// -----------------------------------------------------------------------------
// Rollout
// -----------------------------------------------------------------------------

double SingleLegReach::rollout(int s, const RobotState& state)
{
    using Clock = std::chrono::steady_clock;
    using Us    = std::chrono::microseconds;

    mjData* d = data_[s];
    set_mj_state(d, state);

    double activation[NUM_MUSCLES];
    std::memcpy(activation, rollout_act_, NUM_MUSCLES * sizeof(double));

    double total_cost = 0.0;

    for (int t = 0; t < task_.horizon; ++t) {
        double act_cmd[NUM_MUSCLES];
        for (int m = 0; m < NUM_MUSCLES; ++m) {
            double noisy = trajectory_[t * NUM_MUSCLES + m]
                         + noise_[s * task_.horizon * NUM_MUSCLES + t * NUM_MUSCLES + m];
            act_cmd[m] = std::clamp(noisy, 0.0, 1.0);
        }

        double tau_out[NUM_JOINTS];
        for (int sub = 0; sub < task_.substeps; ++sub) {
            double q_cur[NUM_JOINTS], dq_cur[NUM_JOINTS];
            for (int j = 0; j < NUM_JOINTS; ++j) {
                q_cur[j]  = d->qpos[act_qpos_adr_[j]];
                dq_cur[j] = d->qvel[act_qvel_adr_[j]];
            }

            auto t_h = Clock::now();
            hill_compute_torques(act_cmd, q_cur, dq_cur, muscle_, task_.dt,
                                 activation, tau_out);
            lat_hill_us_.fetch_add(
                std::chrono::duration_cast<Us>(Clock::now() - t_h).count(),
                std::memory_order_relaxed);

            for (int j = 0; j < model_->nu; ++j) d->ctrl[j] = 0.0;
            for (int j = 0; j < NUM_JOINTS; ++j)
                d->ctrl[JOINT_OFFSET + j] = tau_out[j];

            auto t_mj = Clock::now();
            mj_step(model_, d);
            lat_mjstep_us_.fetch_add(
                std::chrono::duration_cast<Us>(Clock::now() - t_mj).count(),
                std::memory_order_relaxed);

            if (!std::isfinite(d->qpos[act_qpos_adr_[0]]))
                return 1e6;
        }

        auto t_c = Clock::now();
        total_cost += step_cost(d);
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

// -----------------------------------------------------------------------------
// Solve
// -----------------------------------------------------------------------------

void SingleLegReach::update(const RobotState& state, double tau_out[NUM_JOINTS])
{
    if (!state.valid) {
        double act_cmd[NUM_MUSCLES];
        for (int m = 0; m < NUM_MUSCLES; ++m) act_cmd[m] = best_traj_[m];
        hill_compute_torques(act_cmd, state.q, state.dq, muscle_, task_.dt,
                             real_act_, tau_out);
        return;
    }

    using Clock = std::chrono::steady_clock;
    using Us    = std::chrono::microseconds;
    auto elapsed_us = [](auto t0){ return std::chrono::duration_cast<Us>(Clock::now() - t0).count(); };

    lat_hill_us_.store(0,   std::memory_order_relaxed);
    lat_mjstep_us_.store(0, std::memory_order_relaxed);
    lat_cost_us_.store(0,   std::memory_order_relaxed);

    auto t0 = Clock::now();

    auto t_ws = Clock::now();
    warm_start(1);
    long long ws_us = elapsed_us(t_ws);

    auto t_ri = Clock::now();
    run_iterations(state);
    long long ri_us = elapsed_us(t_ri);

    // Compute t=0 sample distribution from the last iteration's noise + current mean.
    double samp_mean[NUM_MUSCLES] = {};
    double samp_sq[NUM_MUSCLES]   = {};
    for (int s = 0; s < task_.n_samples; ++s) {
        for (int m = 0; m < NUM_MUSCLES; ++m) {
            double v = std::clamp(trajectory_[m] + noise_[s * task_.horizon * NUM_MUSCLES + m],
                                  action_lo_[m], action_hi_[m]);
            samp_mean[m] += v;
            samp_sq[m]   += v * v;
        }
    }
    for (int m = 0; m < NUM_MUSCLES; ++m) {
        samp_mean[m] /= task_.n_samples;
        samp_sq[m]    = std::sqrt(std::max(0.0, samp_sq[m] / task_.n_samples
                                                - samp_mean[m] * samp_mean[m]));
    }

    double act_cmd[NUM_MUSCLES];
    for (int m = 0; m < NUM_MUSCLES; ++m) act_cmd[m] = best_traj_[m];

    auto t_fh = Clock::now();
    hill_compute_torques(act_cmd, state.q, state.dq, muscle_, task_.dt,
                         real_act_, tau_out);
    long long fh_us = elapsed_us(t_fh);

    long long total_us = elapsed_us(t0);

    std::memcpy(rollout_act_, real_act_, NUM_MUSCLES * sizeof(double));

    {
        mjData* d_check = data_[task_.n_samples];
        set_mj_state(d_check, state);
        const double* fp = d_check->xpos + 3 * foot_body_id_;
        double err = 0.0;
        for (int k = 0; k < 3; ++k) {
            double dk = fp[k] - active_target_[k];
            err += dk * dk;
        }
        maybe_advance_target(std::sqrt(err));
    }

    if (act_log_.is_open()) {
        act_log_ << ++act_call_count_ << "," << best_cost_;
        for (int m = 0; m < NUM_MUSCLES; ++m) act_log_ << "," << act_cmd[m];
        for (int m = 0; m < NUM_MUSCLES; ++m) act_log_ << "," << real_act_[m];
        for (int j = 0; j < NUM_JOINTS;  ++j) act_log_ << "," << tau_out[j];
        for (int m = 0; m < NUM_MUSCLES; ++m)
            act_log_ << "," << samp_mean[m] << "," << samp_sq[m];
        act_log_ << "\n";
        act_log_.flush();
    }

    if (lat_log_.is_open()) {
        long long n_rollouts = (long long)task_.n_iterations * task_.n_samples;
        double avg_hill_ms   = lat_hill_us_.load(std::memory_order_relaxed)   * 1e-3 / n_rollouts;
        double avg_mjstep_ms = lat_mjstep_us_.load(std::memory_order_relaxed) * 1e-3 / n_rollouts;
        double avg_cost_ms   = lat_cost_us_.load(std::memory_order_relaxed)   * 1e-3 / n_rollouts;

        lat_log_ << ++lat_call_count_   << ","
                 << total_us  * 1e-3    << ","
                 << ws_us     * 1e-3    << ","
                 << ri_us     * 1e-3    << ","
                 << fh_us     * 1e-3    << ","
                 << avg_hill_ms         << ","
                 << avg_mjstep_ms       << ","
                 << avg_cost_ms         << "\n";
        lat_log_.flush();
    }
}
