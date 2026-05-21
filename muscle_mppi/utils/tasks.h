#pragma once

#include <string>

static constexpr int NUM_JOINTS   = 3;             // FL leg: hip, thigh, calf
static constexpr int NUM_MUSCLES  = 2 * NUM_JOINTS; // antagonistic pair per joint
static constexpr int JOINT_OFFSET = 3;             // FL actuators start at index 3 in the model

struct MuscleParams {
    double act_bandwidth = 100.0;              // activation filter bandwidth (Hz)
    double peak_force[NUM_JOINTS] = {};        // peak isometric force (N), scales FL*FV output
    double lce_min[NUM_JOINTS]    = {};        // min fiber length (normalized by l_opt)
    double lce_max[NUM_JOINTS]    = {};        // max fiber length (normalized by l_opt)
    double phi_min[NUM_JOINTS]    = {};        // joint angle (rad) mapping to lce_min for agonist
    double phi_max[NUM_JOINTS]    = {};        // joint angle (rad) mapping to lce_max for agonist
    double vmax[NUM_JOINTS]       = {};        // max contraction velocity (fiber lengths / s)
    double fvmax[NUM_JOINTS]      = {};        // eccentric force amplification (>1)
    double fpmax[NUM_JOINTS]      = {};        // passive force at max extension
    double kd_sim[NUM_JOINTS]     = {};        // MuJoCo joint damping (applied to sim dofs)
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
    double       height_target           = 0.0;
    double       nominal_pose[NUM_JOINTS] = {};
    double       foot_target[3]          = {};
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
    double       noise_sigma[NUM_MUSCLES] = {};   // one sigma per virtual muscle
};

struct MotionCommand {
    double vx     = 0.0;
    double vy     = 0.0;
    double wz     = 0.0;
    double height = 0.0;
};

TaskConfig load_task(const std::string& task_name,
                     const std::string& yaml_path = "../utils/tasks.yaml");
