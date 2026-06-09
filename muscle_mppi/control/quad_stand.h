#pragma once

#include "base_mppi.h"
#include "muscle.h"

struct StandCostWeights {
    double height      = 0.0;
    double orientation = 0.0;
    double posture     = 0.0;
    double terminal    = 0.0;   // L1 positional drift from start (vel_des=0, mirrors walk)
};

class QuadStand : public BaseMPPI {
public:
    explicit QuadStand(const std::string& task_name,
                       const std::string& yaml_path = "../utils/tasks.yaml");

    void update(const RobotState& state, double tau_out[NUM_JOINTS]);

    const MuscleParams&    muscle_params() const { return muscle_; }
    const StandCostWeights& cost_weights() const { return cost_; }
    double best_cost()       const { return best_cost_; }
    double trajectory_cost() const { return trajectory_cost_; }

private:
    double rollout(int s, const RobotState& state) override;

    double step_cost(const mjData* d, const double act_cmd[NUM_MUSCLES]);
    double terminal_cost(const mjData* d);

    MuscleParams     muscle_;
    StandCostWeights cost_;

    // Fixed constraint-line parameters at the nominal standing pose.
    // Posture cost: posture_FL1_[j]*a1 - posture_FL2_[j]*a2 = posture_bias_[j].
    double posture_bias_[NUM_JOINTS] = {};
    double posture_FL1_[NUM_JOINTS]  = {};
    double posture_FL2_[NUM_JOINTS]  = {};

    // rollout_act_ is snapshotted from real_act_ before each solve so all
    // parallel rollouts start from the same activation state.
    double rollout_act_[NUM_MUSCLES] = {};
    double real_act_[NUM_MUSCLES]    = {};

    double start_pos_[2] = {};   // x,y at solve start — terminal cost anchor
    double trajectory_cost_ = 1e9;

    double last_compute_ms_ = 20.0;

    int base_bid_ = -1;
};
