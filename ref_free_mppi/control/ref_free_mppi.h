#pragma once

#include <mujoco/mujoco.h>
#include <algorithm>
#include <vector>
#include <random>
#include <string>
#include "../utils/tasks.h"

// Sensor data assembled from rt/lowstate + rt/sportmodestate
struct RobotState {
    double pos[3]  = {};
    double vel[3]  = {};
    double quat[4] = {1, 0, 0, 0};  // w, x, y, z
    double gyro[3] = {};
    double q[NUM_JOINTS]  = {};
    double dq[NUM_JOINTS] = {};
    bool   valid = false;
};

class RefFreeMPPI {
public:
    explicit RefFreeMPPI(const std::string& task_name);
    ~RefFreeMPPI();

    void set_height_target(double z) { height_target_ = z; }

    // clamp(vx, 0, vel_des)/r — brakes when stationary, assists rolling when walking
    double wheel_omega_cmd(double vx) const {
        return std::clamp(vx, 0.0, task_.cost.vel_des[0]) / task_.wheel_radius;
    }

    void set_predict_delay(double delay_s) {
        predict_steps_ = std::max(1,
            static_cast<int>(std::round(delay_s / task_.dt_ctrl)));
    }

    void update(const RobotState& state,
                double q_out[NUM_JOINTS],
                double dq_out[NUM_JOINTS]);

    void get_torques(const RobotState& state, double tau_out[NUM_JOINTS]) const;

private:
    void spline_eval(const std::vector<double>& tq,
                     const std::vector<double>& tv,
                     double t,
                     double* q_out,
                     double* dq_out) const;

    void clamp_node_velocities(std::vector<double>& tq,
                               std::vector<double>& tv) const;

    void shift_best(int steps);

    void precompute_noise_schedule();

    double rollout(int s,
                   const RobotState& state,
                   const std::vector<double>& tq,
                   const std::vector<double>& tv);

    double step_cost(const mjData* d, double t_horizon) const;
    double terminal_cost(const mjData* d, const double pos0[3]) const;

    void set_mj_state(mjData* d, const RobotState& state) const;
    RobotState predict_state(const RobotState& state);

    TaskConfig task_;
    mjModel*   model_ = nullptr;

    std::vector<mjData*> data_;

    std::vector<double> theta_q_nom_;
    std::vector<double> theta_v_nom_;
    std::vector<double> theta_q_best_;
    std::vector<double> theta_v_best_;
    double best_cost_ = 1e12;

    std::vector<double> noise_sched_;

    std::vector<double> sample_tq_;
    std::vector<double> sample_tv_;
    std::vector<double> costs_;

    double height_target_;
    double dt_spline_;
    int    n_ctrl_;

    // LowCmd actuator order (FR/FL/RR/RL) ≠ MuJoCo body order; built from actuator_trnid.
    int act_qpos_adr_[NUM_JOINTS] = {};
    int act_qvel_adr_[NUM_JOINTS] = {};

    int    base_bid_ = 1;
    int    wheel_body_ids_[4] = {-1, -1, -1, -1};
    double wheel_axle_[4][3]  = {};
    double total_mass_        = 15.0;
    double scale_q_[NUM_JOINTS] = {};
    double scale_v_[NUM_JOINTS] = {};
    int    predict_steps_ = 1;

    std::mt19937 rng_;
    std::normal_distribution<double> normal_{0.0, 1.0};
};
