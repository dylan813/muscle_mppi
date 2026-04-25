#pragma once

#include <mujoco/mujoco.h>
#include <vector>
#include <random>
#include <string>
#include <algorithm>
#include "../utils/tasks.h"

struct RobotState {
    double pos[3]  = {};
    double vel[3]  = {};
    double quat[4] = {1,0,0,0};  // w, x, y, z
    double gyro[3] = {};
    double q[16]   = {};
    double dq[16]  = {};
    bool   valid   = false;
};

class BaseMPPI {
public:
    explicit BaseMPPI(const TaskConfig& task);
    virtual ~BaseMPPI();

    void   set_height_target(double z) { height_target_ = z; }
    double height_target()       const { return height_target_; }

    void set_act_reference_weight(double w) { task_.cost.act_reference = w; }

    double cost_min()  const { return *std::min_element(costs_.begin(), costs_.end()); }
    double cost_max()  const { return *std::max_element(costs_.begin(), costs_.end()); }
    double cost_mean() const {
        double s = 0.0;
        for (auto c : costs_) s += c;
        return s / static_cast<double>(costs_.size());
    }

protected:
    virtual double rollout(int s, const RobotState& state) = 0;

    void   sample_noise(int iter, int n_iters);
    void   set_mj_state(mjData* d, const RobotState& state);

    TaskConfig task_;

    mjModel*             model_ = nullptr;
    std::vector<mjData*> data_;        // [n_samples + 1]: rollout slots + prediction

    std::vector<double> trajectory_;
    std::vector<double> noise_;
    std::vector<double> costs_;
    std::vector<double> noise_sched_;

    // Actuator → MuJoCo DOF addresses (from actuator_trnid — no hardcoded mapping)
    int act_qpos_adr_[NUM_JOINTS] = {};
    int act_qvel_adr_[NUM_JOINTS] = {};

    double height_target_;

    std::mt19937 rng_;
    std::normal_distribution<double> normal_{0.0, 1.0};
};
