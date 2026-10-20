#pragma once

// Policy observation y: what the flow-matching policy (learning/) conditions
// on, taken at the state update() plans from. Each term is something MPPI's
// plan depends on, through the cost or through the state every rollout starts
// from.
//
//   index  term                                              count
//   0–2    goal − CoM, heading frame (forward, left, up)     3
//   3–5    gravity direction, body frame                     3
//   6–8    trunk linear velocity, body frame                 3
//   9–11   trunk angular velocity, body frame                3
//   12–23  joint positions                                   12
//   24–35  joint velocities                                  12
//   36–59  muscle activations, interleaved as act[]          24
//
// Heading frame: world axes turned by the trunk's yaw only, so the goal offset
// doesn't move with the trunk's pitch and roll, and its third value is exactly
// the height error the cost's pos_z term scores.

#include <cmath>

#include <mujoco/mujoco.h>

#include "../utils/tasks.h"
#include "../../common/control_utils.h"

// Start index of each term in y.
static constexpr int OBS_GOAL    = 0;
static constexpr int OBS_GRAVITY = OBS_GOAL + 3;
static constexpr int OBS_LIN_VEL = OBS_GRAVITY + 3;
static constexpr int OBS_ANG_VEL = OBS_LIN_VEL + 3;
static constexpr int OBS_Q       = OBS_ANG_VEL + 3;
static constexpr int OBS_DQ      = OBS_Q + NUM_JOINTS;
static constexpr int OBS_ACT     = OBS_DQ + NUM_JOINTS;
static constexpr int OBS_DIM     = OBS_ACT + NUM_MUSCLES;

// com_pos: whole-robot CoM in the world frame, the point the position cost
// scores (MPPILocomotion::base_com_state()). act: muscle activations in
// hill_compute_torques' layout, [agonist_j0, antagonist_j0, agonist_j1, ...].
inline void build_observation(const RobotState& state, const double com_pos[3],
                              const double act[NUM_MUSCLES], const double goal_pos[3],
                              double y[OBS_DIM])
{
    double R[9];
    mju_quat2Mat(R, state.quat);   // body -> world, row-major (as mjData::xmat)

    // Heading: yaw of the trunk's forward (body x) axis, computed as legged_gym does.
    const double yaw = std::atan2(R[3], R[0]);
    const double c = std::cos(yaw), s = std::sin(yaw);
    const double dx = goal_pos[0] - com_pos[0];
    const double dy = goal_pos[1] - com_pos[1];
    y[OBS_GOAL]     =  c * dx + s * dy;
    y[OBS_GOAL + 1] = -s * dx + c * dy;
    y[OBS_GOAL + 2] = goal_pos[2] - com_pos[2];

    const double gravity_world[3] = {0.0, 0.0, -1.0};
    world_to_body(R, gravity_world, y + OBS_GRAVITY);
    world_to_body(R, state.vel, y + OBS_LIN_VEL);   // state.vel is world frame (free joint qvel)

    for (int k = 0; k < 3; ++k)           y[OBS_ANG_VEL + k] = state.gyro[k];   // already body frame
    for (int j = 0; j < NUM_JOINTS; ++j)  y[OBS_Q + j]       = state.q[j];
    for (int j = 0; j < NUM_JOINTS; ++j)  y[OBS_DQ + j]      = state.dq[j];
    for (int m = 0; m < NUM_MUSCLES; ++m) y[OBS_ACT + m]     = act[m];
}
