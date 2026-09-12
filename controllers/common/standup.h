#pragma once

// Stand-up procedure shared by the DDS controller (muscle/mppi_controller.cpp)
// and both standalone sims: a tanh-ramped PD blend from the crouched
// stand-down pose to the standing pose, mirroring unitree_mujoco's stand_go2.cpp.

#include <cmath>

#include "types.h"

// Crouched pose the robot starts from.
static constexpr double kStandDownPose[NUM_JOINTS] = {
     0.0473455,  1.22187, -2.44375,   // FR
    -0.0473455,  1.22187, -2.44375,   // FL
     0.0473455,  1.22187, -2.44375,   // RR
    -0.0473455,  1.22187, -2.44375,   // RL
};

// Standing pose the ramp converges to.
static constexpr double kStandUpPose[NUM_JOINTS] = {
    0.0, 0.67, -1.3,   // FR
    0.0, 0.67, -1.3,   // FL
    0.0, 0.67, -1.3,   // RR
    0.0, 0.67, -1.3,   // RL
};

static constexpr double kStandupSecs = 3.0;   // tanh ramp
static constexpr double kHoldSecs    = 1.0;   // hold pose before handing to MPPI
static constexpr double kStandupKd   = 3.5;   // PD damping during stand-up

// MPPI solves to run before handing control to it (and, in the sims, before logging).
static constexpr int    kConvergenceSolves = 10;

// PD targets at `t` seconds into the stand-up: kp ramps 20 -> 50 and the
// joint target blends stand-down -> stand-up, both along tanh(t / 1.2).
inline void standup_targets(double t, double& kp, double q_des[NUM_JOINTS])
{
    const double phase = std::tanh(t / 1.2);
    kp = phase * 50.0 + (1.0 - phase) * 20.0;
    for (int j = 0; j < NUM_JOINTS; ++j)
        q_des[j] = phase * kStandUpPose[j] + (1.0 - phase) * kStandDownPose[j];
}
