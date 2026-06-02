#include "single_leg_reach.h"

#include <algorithm>
#include <cmath>
#include <cstring>
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
        const double ag_target  = baseline + std::clamp( u[j], 0.0, 1.0) * (1.0 - baseline);
        const double ant_target = baseline + std::clamp(-u[j], 0.0, 1.0) * (1.0 - baseline);

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
                                const std::string& yaml_path)
    : BaseMPPI(load_task(task_name, yaml_path))
{
    muscle_ = task_.muscle;

    for (int j = 0; j < NUM_JOINTS; ++j) {
        action_lo_[j] = -1.0;
        action_hi_[j] =  1.0;
    }

    foot_body_id_ = mj_name2id(model_, mjOBJ_BODY, "FL_foot");
    if (foot_body_id_ < 0)
        throw std::runtime_error("Body not found: FL_foot");
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
            hill_net_act_torques(u, q_cur, dq_cur, muscle_, task_.dt,
                                 BASELINE, activation, tau_out);

            for (int j = 0; j < model_->nu; ++j) d->ctrl[j] = 0.0;
            for (int j = 0; j < NUM_JOINTS; ++j)
                d->ctrl[JOINT_OFFSET + j] = tau_out[j];

            mj_step(model_, d);

            if (!std::isfinite(d->qpos[act_qpos_adr_[0]]))
                return 1e6;
        }

        total_cost += step_cost(d, u, tau_out, tau_prev, t);
        std::memcpy(tau_prev, tau_out, NUM_JOINTS * sizeof(double));
    }

    total_cost += terminal_cost(d);
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

    warm_start(1);
    run_iterations(state);

    double u[NUM_JOINTS];
    for (int j = 0; j < NUM_JOINTS; ++j) u[j] = best_traj_[j];
    hill_net_act_torques(u, state.q, state.dq, muscle_, task_.dt,
                         BASELINE, real_act_, tau_out);

    std::memcpy(rollout_act_, real_act_, 2 * NUM_JOINTS * sizeof(double));
}
