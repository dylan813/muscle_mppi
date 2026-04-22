#pragma once
#include <string>

// Actuator order (matches go2w.xml):
//  0-2:  FR hip/thigh/calf
//  3-5:  FL hip/thigh/calf
//  6-8:  RR hip/thigh/calf
//  9-11: RL hip/thigh/calf
//  12-15: FR/FL/RR/RL wheel
static constexpr int NUM_JOINTS     = 16;
static constexpr int NUM_LEG_JOINTS = 12;
static constexpr int NUM_WHEELS     = 4;

// Reference-free cost weights — mirrors Table II of the paper.
// Wheel joints are excluded from contact-velocity penalty (rolling is expected).
struct CostWeights {
    double height      = 100.0;  // w_h:       L1 base height deviation
    double orientation = 10.0;   // w_orient:  SO(3) log angle squared
    double joint_reg   = 0.0;    // w_q:       L2 deviation from nominal pose
    double contact_vel = 0.5;    // w_c_vel:   L1 wheel body linear velocity (slip)
    double contact_frc = 5e-2;   // w_c_force: L1 vertical contact force deviation
    double terminal    = 2.5e3;  // w_H:       L1 horizontal base displacement
    double vel_cmd     = 0.0;    // w_vel:     velocity command tracking (0 = stand)
    double vel_des[3]  = {0.0, 0.0, 0.0};  // desired base velocity [m/s]
};

// Reference-free MPPI algorithm parameters (Sec. III of the paper).
struct RefFreeParams {
    int    K      = 6;     // spline nodes; K-1 Hermite segments cover H_time
    int    I_iter = 3;     // inner iterations per control step
    double H_time = 0.9;   // horizon duration [s]
    double beta1  = 3.0;   // trajectory-level annealing temperature (Eq. 6)
    double beta2  = 3.0;   // action-level annealing temperature (Eq. 7)

    // Position noise std per joint [rad] — leg / wheel
    double scale_q_leg   = 0.05;
    double scale_q_wheel = 0.15;  // wheels: larger, unbounded rotation

    // Velocity noise std per joint [rad/s] — leg / wheel
    double scale_v_leg   = 0.3;
    double scale_v_wheel = 1.0;   // wheels: rolling speed can vary widely
};

struct TaskConfig {
    const char* model_path =
        "../../unitree_mujoco/unitree_robots/go2w/scene_terrain.xml";

    double height_target = 0.45;           // overwritten at runtime
    double nominal_pose[NUM_JOINTS] = {};  // standing configuration

    CostWeights  cost;
    RefFreeParams rf;

    // MPPI sampling
    int    n_samples = 30;
    double lambda    = 0.5;   // temperature on min-max normalized costs ∈ [0,1]

    // Physics
    double dt       = 0.002;  // MuJoCo physics timestep [s]  — must match sim
    double dt_ctrl  = 0.02;   // control period [s]  (50 Hz)
    int    substeps = 10;     // physics steps per control step  (dt_ctrl / dt)

    // PD gains (must match values sent via LowCmd)
    double kp = 50.0;
    double kd =  3.5;

    // Joint position limits used for Hermite velocity clamping (Eq. 5).
    // Wheels are unbounded — clamping is skipped for indices 12-15.
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
