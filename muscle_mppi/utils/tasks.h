#pragma once

#include <string>
#include <vector>

// These are overridable at compile time via -DNUM_JOINTS=N -DJOINT_OFFSET=N.
// Defaults: full quadruped (12 joints, offset 0).
// Single-leg build: -DNUM_JOINTS=3 -DJOINT_OFFSET=3
#ifndef NUM_JOINTS
static constexpr int NUM_JOINTS   = 12;   // 4 legs × 3 joints (FR, FL, RR, RL)
static constexpr int JOINT_OFFSET = 0;    // actuators start at index 0
#endif
#ifndef NUM_MUSCLES
static constexpr int NUM_MUSCLES  = 2 * NUM_JOINTS;
#endif  // antagonistic pair per joint

struct MuscleParams {
    double act_bandwidth = 100.0;              // activation filter bandwidth (Hz)
    double peak_force[NUM_JOINTS] = {};        // peak isometric force (N), scales FL*FV output
    double lce_min[NUM_JOINTS]    = {};        // min fiber length (normalized by l_opt)
    double lce_max[NUM_JOINTS]    = {};        // max fiber length (normalized by l_opt)
    double phi_min[NUM_JOINTS]    = {};        // joint angle (rad) mapping to lce_min for agonist
    double phi_max[NUM_JOINTS]    = {};        // joint angle (rad) mapping to lce_max for agonist
    double vmax[NUM_JOINTS]       = {};        // max contraction velocity (fiber lengths / s)
    double FVmax[NUM_JOINTS]      = {};        // eccentric force amplification (>1)
    double pFLmax[NUM_JOINTS]     = {};        // passive force at max extension
    double kd_sim[NUM_JOINTS]     = {};        // MuJoCo joint damping (applied to sim dofs)
};

// One waypoint in a task's phase sequence, matching RTWholeBodyMPPI's per-phase
// goal_pos/cmd_vel/desired_gait/goal_thresh/waiting_times arrays. MPPILocomotion
// advances phase_index_ once the robot has stayed within goal_thresh of the
// current phase's goal_pos for waiting_time consecutive in-threshold ticks.
struct TaskPhase {
    double goal_pos[3]  = {};
    double cmd_vel[2]   = {};  // [vx, vy] body-frame velocity command

    // Categorical gait name: "in_place" | "walk" | "walk_fast" | "trot".
    // Resolved to a gait TSV path by MPPILocomotion (see mppi_locomotion.cpp).
    std::string desired_gait;

    // Optional escape hatch: an explicit gait TSV path, overriding desired_gait
    // when non-empty. Used by tooling (e.g. the CMA-ES muscle-parameter sweep in
    // analysis/optimize/objective.py) that regenerates a gait file per candidate
    // and needs mppi_sim to load that exact file rather than a canonical one.
    std::string gait_path;

    double goal_thresh  = 0.2;
    int    waiting_time = 0;   // dwell ticks required within goal_thresh
};

struct TaskConfig {
    std::string  model_path;

    // Ordered waypoint sequence for locomotion tasks (MPPILocomotion). Empty for
    // non-locomotion tasks (e.g. "reach", driven by SingleLegReach instead).
    std::vector<TaskPhase> phases;

    double       height_target            = 0.0;
    double       nominal_pose[NUM_JOINTS] = {};
    MuscleParams muscle;
    int          n_samples    = 16;
    int          horizon      = 25;
    double       lambda       = 0.1;
    double       dt           = 0.002;

    // OpenMP thread count for the parallel rollout loop. 0 (default) leaves the
    // OpenMP runtime default in place (typically all available cores).
    int          num_threads  = 0;

    // Noise sampling. "normal": iid Gaussian per timestep (default).
    // "cubic": draw n_knots iid Gaussians spread evenly across the horizon and
    // natural-cubic-spline interpolate between them, matching RTWholeBodyMPPI's
    // spline-parameterized sampling (smoother, lower-dimensional search).
    std::string  sample_type  = "normal";
    int          n_knots      = 4;

    // Per-joint noise sigma. Both muscles of each antagonistic pair receive the
    // same draw, scaled by this value. Used by BaseMPPI::sample_noise().
    double noise_sigma_act[NUM_JOINTS]   = {};

    // Normalized gravity torque at the nominal pose: tau_grav/(-r*peak_force) - (P1-P2).
    // Used by MPPILocomotion to seed the trajectory warm-start.
    double posture_bias[NUM_JOINTS] = {};
    double posture_FL1[NUM_JOINTS]  = {};
    double posture_FL2[NUM_JOINTS]  = {};

    // Co-contraction sampling parameters (per joint type: hip=0, thigh=1, calf=2).
};

struct MotionCommand {
    double vx          = 0.0;
    double vy          = 0.0;
    double goal_pos[3] = {};  // world-frame position target [x, y, z]
};

TaskConfig load_task(const std::string& task_name,
                     const std::string& yaml_path = "../utils/tasks.yaml");
