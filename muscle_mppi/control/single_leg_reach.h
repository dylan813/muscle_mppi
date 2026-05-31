#pragma once

#include "base_mppi.h"
#include "muscle.h"

class SingleLegReach : public BaseMPPI {
public:
    explicit SingleLegReach(const std::string& task_name,
                            const std::string& yaml_path = "../utils/tasks.yaml");

    void update(const RobotState& state, double activations_out[NUM_MUSCLES]);

    void compute_real_torques(const RobotState& state,
                              const double activations[NUM_MUSCLES],
                              double tau_out[NUM_JOINTS]);

    void set_command(const MotionCommand& cmd) { cmd_ = cmd; }
    const MotionCommand& command() const { return cmd_; }

    const MuscleState&  muscle_state()  const { return muscle_state_; }
    const MuscleParams& muscle_params() const { return muscle_; }

private:
    double rollout(int s, const RobotState& state) override;

    double step_cost(const mjData* d,
                     const double act_cmd[NUM_MUSCLES],
                     int horizon_step);

    double terminal_cost(const mjData* d);

    RobotState predict_state(const RobotState& state, int n_steps);

    MuscleParams   muscle_;
    MuscleState    muscle_state_;
    MotionCommand  cmd_;

    double  start_pos_[3];
    std::vector<double> best_traj_;
    double  best_cost_        = 1e9;
    double  last_compute_ms_  = 20.0;
    double  predicted_activation_[NUM_MUSCLES] = {};

    int base_bid_ = 1;
    int foot_body_id_ = -1;

    static constexpr double ACT_MIN = 0.0;
    static constexpr double ACT_MAX = 1.0;
};
