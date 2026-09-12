#pragma once

// Types shared by the muscle-actuated (controllers/muscle/) and PD-actuated
// (controllers/pd/) variants. Variant-specific types (MuscleParams/NUM_MUSCLES,
// PDParams, each variant's TaskConfig) stay in that variant's utils/.

#include <string>

static constexpr int NUM_JOINTS = 12;   // 4 legs × 3 joints (FR, FL, RR, RL)

struct RobotState {
    double pos[3]          = {};
    double vel[3]          = {};
    double quat[4]         = {1,0,0,0};  // w, x, y, z
    double gyro[3]         = {};
    double q[NUM_JOINTS]   = {};
    double dq[NUM_JOINTS]  = {};
    bool   valid           = false;
};

// One waypoint in a task's phase sequence, matching RTWholeBodyMPPI's per-phase
// goal_pos/cmd_vel/desired_gait/goal_thresh/waiting_times arrays. Each variant's
// locomotion controller (MPPILocomotion / MPPILocomotionPD) advances
// phase_index_ once the robot has stayed within goal_thresh of the current
// phase's goal_pos for waiting_time consecutive in-threshold ticks.
struct TaskPhase {
    double goal_pos[3]  = {};
    double cmd_vel[2]   = {};  // [vx, vy] body-frame velocity command

    // Categorical gait name: "in_place" | "walk" | "walk_fast" | "trot".
    // Resolved to a gait TSV path by each variant's locomotion controller
    // (see mppi_locomotion.cpp / mppi_locomotion_pd.cpp).
    std::string desired_gait;

    // Optional escape hatch: an explicit gait TSV path, overriding desired_gait
    // when non-empty. Used by tooling (e.g. the CMA-ES muscle-parameter sweep in
    // analysis/optimize/objective.py) that regenerates a gait file per candidate
    // and needs mppi_sim to load that exact file rather than a canonical one.
    std::string gait_path;

    double goal_thresh  = 0.2;
    int    waiting_time = 0;   // dwell ticks required within goal_thresh

    // Optional per-phase override of TaskConfig::noise_sigma_act, applied while
    // this phase is active and reverted to the task-level baseline on the next
    // phase that doesn't set one. Mirrors RTWholeBodyMPPI's next_goal(), which
    // doubles thigh/calf exploration noise specifically during trot phases
    // regardless of the task config's declared baseline. has_noise_sigma_act
    // distinguishes "not set" from a legitimate all-zero override.
    double noise_sigma_act[NUM_JOINTS] = {};
    bool   has_noise_sigma_act = false;
};

struct MotionCommand {
    double vx          = 0.0;
    double vy          = 0.0;
    double goal_pos[3] = {};  // world-frame position target [x, y, z]
};
