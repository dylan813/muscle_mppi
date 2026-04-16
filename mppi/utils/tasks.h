#pragma once

#include <string>

// Joint counts — defined here so tasks.h has no upstream dependency
static constexpr int NUM_JOINTS     = 16;
static constexpr int NUM_LEG_JOINTS = 12;
static constexpr int NUM_WHEELS     = 4;

// Per-term cost weights — mirrors Q_diag / R_diag in MPPI_quad YAML configs
struct CostWeights {
    double height         = 3000.0;
    double orientation    = 5000.0;
    double lin_vel        = 300.0;
    double ang_vel        = 300.0;
    double joint_track    = 50.0;
    double effort         = 0.001;
    double terminal_joint = 200.0;
};

// Full task specification — analogous to MPPI_quad tasks.py TASKS dict entry
struct TaskConfig {
    const char* model_path = "../../unitree_mujoco/unitree_robots/go2w/scene_terrain.xml";

    // Goal
    double goal_pos[3]  = {0.0, 0.0, 0.0};
    double height_target = 0.464;       // overwritten at runtime via set_height_target()

    // Nominal pose: warm-starts trajectory + sets joint reference for cost
    double nominal_pose[NUM_JOINTS] = {};

    CostWeights cost;

    // MPPI engine parameters
    int    n_samples = 32;
    int    horizon   = 10;   // control steps
    int    substeps  = 10;   // physics steps per control step (substeps * dt = control period)
    double lambda    = 50.0;
    double dt        = 0.002;

    // PD gains — must match what is sent via LowCmd
    double kp = 50.0;
    double kd =  3.5;

    // Noise std per joint (position-target space, radians)
    double noise_sigma[NUM_JOINTS] = {
        0.02, 0.03, 0.03,       // FR leg
        0.02, 0.03, 0.03,       // FL leg
        0.02, 0.03, 0.03,       // RR leg
        0.02, 0.03, 0.03,       // RL leg
        0.05, 0.05, 0.05, 0.05  // wheels
    };
};

// Returns a fully-populated TaskConfig by name.
// Mirrors MPPI_quad's get_task(task_name).
TaskConfig get_task(const std::string& name);
