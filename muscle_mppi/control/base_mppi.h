#pragma once

#include <mujoco/mujoco.h>
#include <vector>
#include <random>
#include <string>
#include <algorithm>
#include "../utils/tasks.h"

struct RobotState {
    double pos[3]          = {};
    double vel[3]          = {};
    double quat[4]         = {1,0,0,0};  // w, x, y, z
    double gyro[3]         = {};
    double q[NUM_JOINTS]   = {};
    double dq[NUM_JOINTS]  = {};
    bool   valid           = false;
};

class BaseMPPI {
public:
    explicit BaseMPPI(const TaskConfig& task);
    virtual ~BaseMPPI();

    void   set_height_target(double z) { height_target_ = z; }
    double height_target()       const { return height_target_; }



    double cost_min()  const { return *std::min_element(costs_.begin(), costs_.end()); }
    double cost_mean() const {
        double s = 0.0;
        for (auto c : costs_) s += c;
        return s / static_cast<double>(costs_.size());
    }

protected:
    virtual double rollout(int s, const RobotState& state) = 0;

    void sample_noise(int iter);
    void set_mj_state(mjData* d, const RobotState& state);

    // Shift trajectory_ forward by n_skip steps, holding the tail constant.
    void warm_start(int n_skip);

    // Run task_.n_iterations of: sample → parallel rollouts → best tracking → softmin update.
    // Subclasses set action_lo_/action_hi_ in their constructor to define per-action clamping.
    void run_iterations(const RobotState& state);

    TaskConfig task_;

    mjModel*             model_ = nullptr;
    std::vector<mjData*> data_;        // [n_samples + 1]: rollout slots + prediction

    std::vector<double> trajectory_;
    std::vector<double> noise_;
    std::vector<double> costs_;

    std::vector<double> best_traj_;
    double              best_cost_  = 1e9;

    // Per-muscle clamp bounds used by run_iterations(). Set by subclass constructors.
    double action_lo_[NUM_MUSCLES] = {};
    double action_hi_[NUM_MUSCLES] = {};

    // Actuator → MuJoCo DOF addresses (built from JOINT_OFFSET — no hardcoded mapping)
    int  act_qpos_adr_[NUM_JOINTS] = {};
    int  act_qvel_adr_[NUM_JOINTS] = {};
    bool has_freejoint_ = false;

    double height_target_;

    std::mt19937 rng_;
    std::normal_distribution<double> normal_{0.0, 1.0};
};
