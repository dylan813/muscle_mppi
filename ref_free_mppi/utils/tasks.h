#pragma once
#include <string>

// Actuator order: FR(0-2), FL(3-5), RR(6-8), RL(9-11) hip/thigh/calf; wheels(12-15)
static constexpr int NUM_JOINTS     = 16;
static constexpr int NUM_LEG_JOINTS = 12;
static constexpr int NUM_WHEELS     = 4;

struct CostWeights {
    double height      = 100.0;
    double orientation = 10.0;
    double ang_vel     = 1.0;
    double joint_reg   = 0.0;
    double contact_vel = 0.5;
    double contact_frc = 5e-2;
    double terminal    = 2.5e3;
    double vel_cmd     = 0.5;
    double vel_des[3]  = {0.0, 0.0, 0.0};
};

struct RefFreeParams {
    int    K      = 6;
    int    I_iter = 2;
    double H_time = 0.5;
    double beta1  = 3.0;
    double beta2  = 3.0;
    double scale_q_leg   = 0.05;
    double scale_q_wheel = 0.0;
    double scale_v_leg   = 0.3;
    double scale_v_wheel = 0.0;
};

struct TaskConfig {
    const char* model_path =
        "../../unitree_mujoco/unitree_robots/go2w/scene.xml";

    double height_target = 0.0;  // overwritten by measured trunk height after standup
    double nominal_pose[NUM_JOINTS] = {};

    CostWeights   cost;
    RefFreeParams rf;

    int    n_samples = 16;
    double lambda    = 0.5;

    double dt       = 0.002;
    double dt_ctrl  = 0.02;
    int    substeps = 10;

    double kp = 50.0;
    double kd =  3.5;

    // kp_wheel applied as feedforward tau in rollouts; not sent via LowCmd kp
    double kp_wheel     = 5.0;
    double wheel_radius = 0.04;

    // Wheels (12-15) use ±1e9 so velocity clamping is skipped for those indices.
    double q_min[NUM_JOINTS] = {
        -1.0472, -1.5708, -2.7227,   // FR
        -1.0472, -1.5708, -2.7227,   // FL
        -1.0472, -0.5236, -2.7227,   // RR
        -1.0472, -0.5236, -2.7227,   // RL
        -1e9, -1e9, -1e9, -1e9       // wheels (unbounded)
    };
    double q_max[NUM_JOINTS] = {
         1.0472,  3.4907, -0.83776,  // FR
         1.0472,  3.4907, -0.83776,  // FL
         1.0472,  4.5379, -0.83776,  // RR
         1.0472,  4.5379, -0.83776,  // RL
         1e9,  1e9,  1e9,  1e9       // wheels
    };
};

TaskConfig get_task(const std::string& name);
