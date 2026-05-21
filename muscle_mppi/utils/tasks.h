#pragma once

#include <string>

static constexpr int NUM_JOINTS   = 3;   // FL leg: hip, thigh, calf
static constexpr int JOINT_OFFSET = 3;   // FL actuators start at index 3 in the model

struct MuscleParams {
    double activation_alpha = 0.15;
    double tau_max[NUM_JOINTS]    = {};
    double dq_max[NUM_JOINTS]     = {};
    double q_plus0[NUM_JOINTS]    = {};
    double q_minus0[NUM_JOINTS]   = {};
    double k_plus[NUM_JOINTS]     = {};
    double k_minus[NUM_JOINTS]    = {};
    double alpha_plus[NUM_JOINTS] = {};
    double alpha_minus[NUM_JOINTS]= {};
    double b_damp[NUM_JOINTS]     = {};
    double kd_sim[NUM_JOINTS]     = {};
};

struct CostWeights {
    double height        = 0.0;
    double orientation   = 0.0;
    double posture       = 0.0;
    double contact_vel   = 0.0;
    double contact_force = 0.0;
    double terminal      = 0.0;
    double act_effort    = 0.0;
    double vel_cmd       = 0.0;
    double vel_des[3]    = {};
    double foot_pos      = 0.0;
};

struct TaskConfig {
    std::string  model_path;
    double       height_target       = 0.0;
    double       nominal_pose[NUM_JOINTS] = {};
    double       foot_target[3]      = {};   // FL foot target in world frame (x, y, z)
    CostWeights  cost;
    MuscleParams muscle;
    int          n_samples    = 16;
    int          horizon      = 25;
    int          substeps     = 10;
    int          n_iterations = 3;
    double       lambda       = 0.1;
    double       beta1        = 3.0;
    double       beta2        = 3.0;
    double       dt           = 0.002;
    double       noise_sigma[NUM_JOINTS] = {};
};

struct MotionCommand {
    double vx     = 0.0;
    double vy     = 0.0;
    double wz     = 0.0;
    double height = 0.0;
};

TaskConfig load_task(const std::string& task_name,
                     const std::string& yaml_path = "../utils/tasks.yaml");
