#pragma once

#include <mujoco/mujoco.h>
#include <vector>
#include <random>
#include <string>
#include "../utils/tasks.h"

class BaseMPPI {
public:
    explicit BaseMPPI(const TaskConfig& task);
    virtual ~BaseMPPI();

protected:
    virtual double rollout(int s, const RobotState& state) = 0;

    // Fills actions_[s][t][m] with this iteration's sampled activation
    // sequences, already clamped to [0, 1], for both the rollouts and the
    // weighted-average update (as BaseMPPIPD and RTWholeBodyMPPI's
    // perturb_action() do; see sample_actions_cubic()).
    void sample_actions();
    void sample_actions_cubic();
    void set_mj_state(mjData* d, const RobotState& state);

    TaskConfig task_;

    mjModel*             model_ = nullptr;
    std::vector<mjData*> data_;        // [n_samples + 1]: rollout slots + prediction

    std::vector<double> trajectory_;
    std::vector<double> actions_;
    std::vector<double> costs_;

    // Knot timesteps (0..horizon-1) used by sample_actions_cubic() when
    // task_.sample_type == "cubic". Built once in the constructor.
    std::vector<double> knot_x_;

    // Actuator → MuJoCo DOF addresses (built from the model — no hardcoded mapping)
    int  act_qpos_adr_[NUM_JOINTS] = {};
    int  act_qvel_adr_[NUM_JOINTS] = {};
    bool has_freejoint_ = false;

    std::mt19937 rng_;
    std::normal_distribution<double> normal_{0.0, 1.0};
};
