#pragma once

#include <string>

static constexpr int NUM_JOINTS     = 16;
static constexpr int NUM_LEG_JOINTS = 12;
static constexpr int NUM_WHEELS     = 4;

// Hill model parameters — one set per task (could be tuned per robot variant)
struct MuscleParams {
    // Per-joint torque limits (Nm) — from go2w.xml
    double tau_max[NUM_JOINTS] = {
        23.7,  23.7,  45.43,       // FR hip, thigh, calf
        23.7,  23.7,  45.43,       // FL
        23.7,  23.7,  45.43,       // RR
        23.7,  23.7,  45.43,       // RL
        15.0,  15.0,  15.0,  15.0  // wheels
    };

    // Per-joint velocity limits (rad/s) — denominator of Hill force-velocity curve
    double dq_max[NUM_JOINTS] = {
        21.0, 21.0, 21.0,          // FR
        21.0, 21.0, 21.0,          // FL
        21.0, 21.0, 21.0,          // RR
        21.0, 21.0, 21.0,          // RL
        50.0, 50.0, 50.0, 50.0    // wheels
    };

    // Activation dynamics blend factor per physics substep (dt = 0.002 s).
    // Derived from SATA alpha=0.6 at dt~0.01 s, scaled to dt=0.002 s.
    // a[t] = tanh(act_cmd) * alpha + a[t-1] * (1 - alpha)
    double activation_alpha = 0.15;
};

struct CostWeights {
    double height         = 3000.0;
    double orientation    = 5000.0;
    double lin_vel        = 300.0;
    double ang_vel        = 300.0;
    double joint_deviation = 100.0;  // penalises actual joint pos deviation from reference
    double terminal_joint = 200.0;
};

// Full task specification
struct TaskConfig {
    const char* model_path = "../../unitree_mujoco/unitree_robots/go2w/scene_terrain.xml";

    double goal_pos[3]   = {0.0, 0.0, 0.0};
    double height_target = 0.464;

    double nominal_pose[NUM_JOINTS] = {};

    CostWeights  cost;
    MuscleParams muscle;

    int    n_samples = 32;
    int    horizon   = 10;
    int    substeps  = 10;
    double lambda    = 50.0;
    double dt        = 0.002;
    // Note: no kp/kd — muscle model replaces PD entirely

    // Noise in activation space (dimensionless, ∈ [-1,1])
    double noise_sigma[NUM_JOINTS] = {
        0.05, 0.08, 0.08,   // FR leg  — larger than baseline (activation space is smoother)
        0.05, 0.08, 0.08,   // FL leg
        0.05, 0.08, 0.08,   // RR leg
        0.05, 0.08, 0.08,   // RL leg
        0.05, 0.05, 0.05, 0.05  // wheels
    };
};

TaskConfig get_task(const std::string& name);
