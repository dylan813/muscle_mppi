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

    // Cubic sampling's plans (sample_type "cubic") are set by their values at
    // the knot steps (knot_x_). Layouts: knots [n_knots × NUM_MUSCLES] as
    // [k * NUM_MUSCLES + m]; plan [horizon × NUM_MUSCLES] as [t * NUM_MUSCLES + m].
    //   plan_from_knots()  per muscle, the not-a-knot cubic through the knot
    //                      values, clamped to [0, 1]: how sample_actions_cubic()
    //                      builds each sample from its noisy knots
    //   knots_from_plan()  the plan's values at the knot steps
    // Both throw for any other sample_type.
    void plan_from_knots(const double* knots, double* plan) const;
    void knots_from_plan(const double* plan, double* knots) const;

protected:
    // MPPI's cost of sample s (row s of actions_), rolled out from what the
    // update in progress plans from, which the subclass holds.
    virtual double rollout(int s) = 0;

    // Fills actions_[s][t][m], s < n_samples, with this iteration's sampled
    // activation sequences, already clamped to [0, 1], for both the rollouts
    // and the weighted-average update (as BaseMPPIPD and RTWholeBodyMPPI's
    // perturb_action() do; see sample_actions_cubic()).
    void sample_actions();
    void sample_actions_cubic();
    void set_mj_state(mjData* d, const RobotState& state);

    // Samples per update: MPPI's n_samples, then the policy's num_samples
    // (task_.policy), as gpc's PolicyAugmentedController appends them.
    int total_samples() const { return task_.n_samples + task_.policy.num_samples; }

    TaskConfig task_;

    mjModel*             model_ = nullptr;
    std::vector<mjData*> data_;        // [total_samples() + 1]: rollout slots + prediction

    std::vector<double> trajectory_;
    std::vector<double> actions_;      // [total_samples() × horizon × NUM_MUSCLES]
    std::vector<double> costs_;        // [total_samples()]

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
