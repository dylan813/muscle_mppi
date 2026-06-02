#include "single_leg_reach.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>

// -----------------------------------------------------------------------------
// Hill model for net-activation parameterisation.
//
// hill_compute_torques() in muscle.h requires NUM_MUSCLES == 2*NUM_JOINTS, which
// is violated when NUM_MUSCLES=3 (one net-activation slot per joint). This
// function replicates the same physics using a local 6-element activation state,
// reusing the force-curve helpers (active_force_length, force_vel,
// passive_force_length) from muscle.h.
// -----------------------------------------------------------------------------
static void hill_net_act_torques(
    const double        u[NUM_JOINTS],           // net activations ∈ [-1, 1]
    const double        q[NUM_JOINTS],
    const double        dq[NUM_JOINTS],
    const MuscleParams& p,
    double              dt,
    double              baseline,
    double              act_state[2 * NUM_JOINTS],  // in/out, always 6 elements
    double              tau_out[NUM_JOINTS])
{
    const double alpha = p.act_bandwidth * dt;

    for (int j = 0; j < NUM_JOINTS; ++j) {
        // u[j] ∈ [0, 1]: 0.5 = neutral (both at baseline), 1.0 = full agonist, 0.0 = full antagonist.
        const double ag_target  = baseline + std::clamp(2.0 * u[j] - 1.0, 0.0, 1.0) * (1.0 - baseline);
        const double ant_target = baseline + std::clamp(1.0 - 2.0 * u[j], 0.0, 1.0) * (1.0 - baseline);

        double& act1 = act_state[2 * j];
        double& act2 = act_state[2 * j + 1];
        act1 = std::clamp(act1 + alpha * (ag_target  - act1), 0.0, 1.0);
        act2 = std::clamp(act2 + alpha * (ant_target - act2), 0.0, 1.0);

        static constexpr double eps = 1e-6;
        const double r1 = (p.lce_max[j] - p.lce_min[j] + eps)
                        / (p.phi_max[j]  - p.phi_min[j]  + eps);
        const double r2 = -r1;

        const double lce1 = q[j] * r1 + (p.lce_min[j] - r1 * p.phi_min[j]);
        const double lce2 = q[j] * r2 + (p.lce_min[j] - r2 * p.phi_max[j]);

        const double lmin = p.lce_min[j], lmax = p.lce_max[j];
        const double FL1 = active_force_length(lce1, lmin, 1.0, lmax)
                         + 0.15 * active_force_length(lce1, lmin, 0.5*(lmin+0.95), 0.95);
        const double FL2 = active_force_length(lce2, lmin, 1.0, lmax)
                         + 0.15 * active_force_length(lce2, lmin, 0.5*(lmin+0.95), 0.95);

        const double c   = p.FVmax[j] - 1.0;
        const double FV1 = force_vel(r1 * dq[j], c, p.vmax[j], p.FVmax[j]);
        const double FV2 = force_vel(r2 * dq[j], c, p.vmax[j], p.FVmax[j]);

        const double b_p  = 0.5 * (lmax + 1.0);
        const double PFL1 = passive_force_length(lce1, p.pFLmax[j], b_p);
        const double PFL2 = passive_force_length(lce2, p.pFLmax[j], b_p);

        const double F1 = (FL1 * FV1 * act1 + PFL1) * p.peak_force[j];
        const double F2 = (FL2 * FV2 * act2 + PFL2) * p.peak_force[j];
        tau_out[j] = -(F1 * r1 + F2 * r2);
    }
}

// -----------------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------------

SingleLegReach::SingleLegReach(const std::string& task_name,
                                const std::string& yaml_path,
                                const std::string& log_dir)
    : BaseMPPI(load_task(task_name, yaml_path))
{
    muscle_ = task_.muscle;

    for (int j = 0; j < NUM_JOINTS; ++j) {
        action_lo_[j] = 0.0;
        action_hi_[j] = 1.0;
    }

    // Neutral activation (0.5 = both muscles at baseline, zero net torque).
    std::fill(best_traj_.begin(), best_traj_.end(), 0.5);

    foot_body_id_ = mj_name2id(model_, mjOBJ_BODY, "FL_foot");
    if (foot_body_id_ < 0)
        throw std::runtime_error("Body not found: FL_foot");

    std::filesystem::create_directories(log_dir);
    lat_log_.open(log_dir + "/single_leg_latency.csv", std::ios::out | std::ios::trunc);
    lat_log_ << "call,total_ms,warm_start_ms,run_iterations_ms,final_hill_ms,"
                "avg_rollout_hill_ms,avg_rollout_mjstep_ms,avg_rollout_cost_ms\n";
}

// -----------------------------------------------------------------------------
// Cost
// -----------------------------------------------------------------------------

double SingleLegReach::step_cost(const mjData* d,
                                  const double u[NUM_JOINTS],
                                  const double tau_out[NUM_JOINTS],
                                  const double tau_prev[NUM_JOINTS],
                                  int /*horizon_step*/)
{
    double cost = 0.0;

    const double* fp = d->xpos + 3 * foot_body_id_;
    double dx = fp[0] - task_.foot_target[0];
    double dy = fp[1] - task_.foot_target[1];
    double dz = fp[2] - task_.foot_target[2];
    cost += task_.cost.foot_pos * (dx*dx + dy*dy + dz*dz);

    if (task_.cost.joint_vel > 0.0) {
        for (int j = 0; j < NUM_JOINTS; ++j) {
            double v = d->qvel[act_qvel_adr_[j]];
            cost += task_.cost.joint_vel * v * v;
        }
    }

    if (task_.cost.act_effort > 0.0) {
        for (int j = 0; j < NUM_JOINTS; ++j)
            cost += task_.cost.act_effort * u[j] * u[j];
    }

    if (task_.cost.torque > 0.0) {
        for (int j = 0; j < NUM_JOINTS; ++j)
            cost += task_.cost.torque * tau_out[j] * tau_out[j];
    }

    if (task_.cost.torque_rate > 0.0) {
        for (int j = 0; j < NUM_JOINTS; ++j) {
            double dr = tau_out[j] - tau_prev[j];
            cost += task_.cost.torque_rate * dr * dr;
        }
    }

    return cost;
}

double SingleLegReach::terminal_cost(const mjData* d)
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

double SingleLegReach::rollout(int s, const RobotState& state)
{
    using Clock = std::chrono::steady_clock;
    using Us    = std::chrono::microseconds;

    mjData* d = data_[s];
    set_mj_state(d, state);

    double activation[2 * NUM_JOINTS];
    std::memcpy(activation, rollout_act_, 2 * NUM_JOINTS * sizeof(double));

    double total_cost = 0.0;
    double tau_prev[NUM_JOINTS] = {};

    for (int t = 0; t < task_.horizon; ++t) {
        double u[NUM_JOINTS];
        for (int j = 0; j < NUM_JOINTS; ++j) {
            double noisy = trajectory_[t * NUM_JOINTS + j]
                         + noise_[s * task_.horizon * NUM_JOINTS + t * NUM_JOINTS + j];
            u[j] = std::clamp(noisy, -1.0, 1.0);
        }

        double tau_out[NUM_JOINTS];
        for (int sub = 0; sub < task_.substeps; ++sub) {
            double q_cur[NUM_JOINTS], dq_cur[NUM_JOINTS];
            for (int j = 0; j < NUM_JOINTS; ++j) {
                q_cur[j]  = d->qpos[act_qpos_adr_[j]];
                dq_cur[j] = d->qvel[act_qvel_adr_[j]];
            }

            auto t_h = Clock::now();
            hill_net_act_torques(u, q_cur, dq_cur, muscle_, task_.dt,
                                 BASELINE, activation, tau_out);
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
        total_cost += step_cost(d, u, tau_out, tau_prev, t);
        lat_cost_us_.fetch_add(
            std::chrono::duration_cast<Us>(Clock::now() - t_c).count(),
            std::memory_order_relaxed);
        std::memcpy(tau_prev, tau_out, NUM_JOINTS * sizeof(double));
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
        double u[NUM_JOINTS];
        for (int j = 0; j < NUM_JOINTS; ++j) u[j] = best_traj_[j];
        hill_net_act_torques(u, state.q, state.dq, muscle_, task_.dt,
                             BASELINE, real_act_, tau_out);
        return;
    }

    using Clock = std::chrono::steady_clock;
    using Us    = std::chrono::microseconds;
    auto elapsed_us = [](auto t0){ return std::chrono::duration_cast<Us>(Clock::now() - t0).count(); };

    lat_hill_us_.store(0,    std::memory_order_relaxed);
    lat_mjstep_us_.store(0,  std::memory_order_relaxed);
    lat_cost_us_.store(0,    std::memory_order_relaxed);

    auto t0 = Clock::now();

    auto t_ws = Clock::now();
    warm_start(1);
    long long ws_us = elapsed_us(t_ws);

    auto t_ri = Clock::now();
    run_iterations(state);
    long long ri_us = elapsed_us(t_ri);

    double u[NUM_JOINTS];
    for (int j = 0; j < NUM_JOINTS; ++j) u[j] = best_traj_[j];

    auto t_fh = Clock::now();
    hill_net_act_torques(u, state.q, state.dq, muscle_, task_.dt,
                         BASELINE, real_act_, tau_out);
    long long fh_us = elapsed_us(t_fh);

    long long total_us = elapsed_us(t0);

    std::memcpy(rollout_act_, real_act_, 2 * NUM_JOINTS * sizeof(double));

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
