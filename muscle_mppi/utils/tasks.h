#pragma once

#include <string>

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

struct TaskConfig {
    std::string  model_path;
    double       height_target            = 0.0;
    double       nominal_pose[NUM_JOINTS] = {};
    MuscleParams muscle;
    int          n_samples    = 16;
    int          horizon      = 25;
    int          substeps     = 10;
    int          n_iterations = 3;
    double       lambda       = 0.1;
    double       dt           = 0.002;

    // Per-muscle activation noise (used by SingleLegReach / BaseMPPI::sample_noise).
    double noise_sigma[NUM_MUSCLES] = {};

    // Per-joint activation noise (used by MPPILocomotion::sample_activation_noise).
    // Applied to both muscles of each pair independently.
    double noise_sigma_act[NUM_JOINTS] = {};
};

struct MotionCommand {
    double vx = 0.0;
    double vy = 0.0;
};

TaskConfig load_task(const std::string& task_name,
                     const std::string& yaml_path = "../utils/tasks.yaml");
