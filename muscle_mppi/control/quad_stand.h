#pragma once

#include "base_mppi.h"
#include "muscle.h"

struct StandCostWeights {
    double height      = 0.0;
    double orientation = 0.0;
    double posture     = 0.0;
    double joint_vel   = 0.0;
    double terminal    = 0.0;
};

// Quadruped standing task with per-joint activation noise (same sigma for
// both muscles of each antagonistic pair). Action space is NUM_MUSCLES wide
// (agonist/antagonist interleaved per joint), but noise is drawn per joint
// via sample_activation_noise — matching the walk task convention.
//
// step_cost / terminal_cost are stubs returning 0; fill them in to define
// the standing objective.
class QuadStand : public BaseMPPI {
public:
    explicit QuadStand(const std::string& task_name,
                       const std::string& yaml_path = "../utils/tasks.yaml");

    void update(const RobotState& state, double tau_out[NUM_JOINTS]);

    const MuscleParams&    muscle_params() const { return muscle_; }
    const StandCostWeights& cost_weights() const { return cost_; }
    double best_cost() const { return best_cost_; }

private:
    double rollout(int s, const RobotState& state) override;

    double step_cost(const mjData* d);
    double terminal_cost(const mjData* d);

    MuscleParams     muscle_;
    StandCostWeights cost_;

    // rollout_act_ is snapshotted from real_act_ before each solve so all
    // parallel rollouts start from the same activation state.
    double rollout_act_[NUM_MUSCLES] = {};
    double real_act_[NUM_MUSCLES]    = {};

    int base_bid_         = -1;
    int foot_body_ids_[4] = {};
    int n_feet_           = 0;
};
