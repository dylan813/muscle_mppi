#pragma once

// Small stateless helpers shared by the controllers (and sims): the Unitree PD
// law, frame rotation, base-body lookup, the goal-facing orientation target,
// and the MPPI warm-start / softmin update steps.
//
// Shared on purpose: the math inside both MPPI loops is identical for the muscle and PD variants so their
// results stay comparable, and editing it changes both. If one variant needs
// different behavior, fork that piece into the variant's own folder first
// rather than changing it here (e.g. settle_standing() is muscle-only and PD
// never calls it; nominal_pose lives in PD's task config).

#include <algorithm>
#include <cmath>
#include <vector>

#include <mujoco/mujoco.h>

// ============================================================================
// PD law and frame rotation
// ============================================================================

// Unitree's PD+feedforward-torque law, ported verbatim from unitree_mujoco's
// Go2Bridge::run() (unitree_mujoco/simulate/src/unitree_sdk2_bridge.h:183-185)
// — the same computation the real Go2's onboard motor firmware performs in
// mode 0x01, and what mppi_controller.cpp already drives via DDS LowCmd_
// q/kp/dq/kd/tau fields. Callers pass dq_des=0, tau_ff=0 unless they have a
// specific feedforward term, matching how this repo's own DDS controller
// already uses it (mppi_controller.cpp sets dq()=0.0) — not a simplification
// specific to this port.
inline double unitree_pd_torque(double kp, double kd, double q_des, double q,
                                double dq_des, double dq, double tau_ff)
{
    return tau_ff + kp * (q_des - q) + kd * (dq_des - dq);
}

// Rotate a world-frame vector into body-frame axes. xmat is the body->world
// rotation (row-major 3x3, as in mjData::xmat or mju_quat2Mat), so its
// transpose maps world->body. v_world and v_body must not alias.
inline void world_to_body(const double xmat[9], const double v_world[3], double v_body[3])
{
    v_body[0] = v_world[0]*xmat[0] + v_world[1]*xmat[3] + v_world[2]*xmat[6];
    v_body[1] = v_world[0]*xmat[1] + v_world[1]*xmat[4] + v_world[2]*xmat[7];
    v_body[2] = v_world[0]*xmat[2] + v_world[1]*xmat[5] + v_world[2]*xmat[8];
}

// Base (trunk) body id, trying the names used across Unitree models; falls back to 1.
inline int find_base_body(const mjModel* m)
{
    for (const char* name : {"trunk", "base", "base_link"}) {
        int bid = mj_name2id(m, mjOBJ_BODY, name);
        if (bid >= 0) return bid;
    }
    return 1;
}

// ============================================================================
// Goal-facing orientation target
// ============================================================================

// Goal-facing orientation target (w, x, y, z) for the locomotion cost, shared
// by both variants. Mirrors RTWholeBodyMPPI's body_ref[3:7], computed once per
// update() via calculate_orientation_quaternion: R_z(yaw) * R_y(pitch) built
// from the direction pos -> goal_pos, zero roll. Only active when farther than
// kGoalFacingMinDist from the goal and not settled at a waypoint (dwelling);
// otherwise the target is identity (upright, no yaw preference).
//
// Callers compute it once per tick and hold it fixed across that tick's whole
// rollout batch.
static constexpr double kGoalFacingMinDist = 0.1;   // m

inline void goal_facing_quat(const double goal_pos[3], const double pos[3],
                             bool dwelling, double quat_out[4])
{
    const double dx = goal_pos[0] - pos[0];
    const double dy = goal_pos[1] - pos[1];
    const double dz = goal_pos[2] - pos[2];
    const double goal_delta = std::sqrt(dx*dx + dy*dy + dz*dz);

    if (goal_delta > kGoalFacingMinDist && !dwelling) {
        const double yaw   = std::atan2(dy, dx);
        const double pitch = -std::atan2(dz, std::sqrt(dx*dx + dy*dy));
        const double z_axis[3] = {0.0, 0.0, 1.0};
        const double y_axis[3] = {0.0, 1.0, 0.0};
        double q_yaw[4], q_pitch[4];
        mju_axisAngle2Quat(q_yaw,   z_axis, yaw);
        mju_axisAngle2Quat(q_pitch, y_axis, pitch);
        mju_mulQuat(quat_out, q_yaw, q_pitch);  // matches scipy's yaw_quat * pitch_quat order
    } else {
        quat_out[0] = 1.0; quat_out[1] = 0.0; quat_out[2] = 0.0; quat_out[3] = 0.0;
    }
}

// ============================================================================
// MPPI update steps
// ============================================================================

// MPPI update building blocks shared by both variants. They work on any
// action width (NUM_MUSCLES for the muscle variant, NUM_JOINTS for PD), with
// trajectories stored row-major as [t * action_dim + a].

// Warm-start: shift a (horizon × action_dim) trajectory forward by one step,
// holding the final step constant (mirrors RTWholeBodyMPPI).
inline void shift_trajectory(std::vector<double>& traj, int horizon, int action_dim)
{
    std::vector<double> shifted(horizon * action_dim);
    for (int t = 0; t < horizon - 1; ++t)
        for (int a = 0; a < action_dim; ++a)
            shifted[t * action_dim + a] = traj[(t + 1) * action_dim + a];
    for (int a = 0; a < action_dim; ++a)
        shifted[(horizon - 1) * action_dim + a] = traj[(horizon - 1) * action_dim + a];
    traj = std::move(shifted);
}

// Softmin sample weights over min-max normalised costs:
//   w_s = exp(-(c_s - c_min) / (c_max - c_min) / lambda),  normalised to sum to 1.
// All weights are equal when every cost is the same (range <= 1e-12).
// Fills weights (resized to costs.size()) and returns c_min, which callers
// log as a diagnostic.
inline double softmin_weights(const std::vector<double>& costs, double lambda,
                              std::vector<double>& weights)
{
    const int n = static_cast<int>(costs.size());
    const double cmin   = *std::min_element(costs.begin(), costs.end());
    const double cmax   = *std::max_element(costs.begin(), costs.end());
    const double crange = cmax - cmin;

    weights.resize(n);
    double wsum = 0.0;
    for (int s = 0; s < n; ++s) {
        const double s_hat = (crange > 1e-12) ? (costs[s] - cmin) / crange : 0.0;
        weights[s] = std::exp(-s_hat / lambda);
        wsum      += weights[s];
    }
    for (int s = 0; s < n; ++s) weights[s] /= wsum;

    return cmin;
}
