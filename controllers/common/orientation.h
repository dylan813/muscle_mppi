#pragma once

#include <cmath>

#include <mujoco/mujoco.h>

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
