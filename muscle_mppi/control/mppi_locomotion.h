#pragma once

#include "base_mppi.h"
#include "gait_scheduler.h"
#include "muscle.h"

struct CostWeights {
    double height      = 0.0;
    double orientation = 0.0;
    double vel         = 0.0;   // body-frame velocity tracking (per step)
    double gait_ref    = 0.0;   // activation gait reference (cost only, not prior)
};

// Reference-free MPPI with direct per-muscle activation (co-contraction capable).
//
// Search space: act[m] ∈ [0, 1] per muscle per horizon step (NUM_MUSCLES × horizon).
// Layout: [agonist_j0, antagonist_j0, agonist_j1, ...] interleaved per joint.
// Activation dynamics in hill_compute_torques provide implicit trajectory smoothing.
//
// Storage layout for trajectory_ and best_traj_ (size horizon × NUM_MUSCLES):
//   [t * NUM_MUSCLES + m] = act[t][m]

class MPPILocomotion : public BaseMPPI {
public:
    explicit MPPILocomotion(const std::string& task_name,
                            const std::string& yaml_path = "../utils/tasks.yaml");

    // Run one MPPI solve; returns Hill-model torques directly.
    void update(const RobotState& state, double tau_out[NUM_JOINTS]);

    void set_command(const MotionCommand& cmd) { cmd_ = cmd; }
    const MotionCommand& command() const { return cmd_; }

    const MuscleParams& muscle_params() const { return muscle_; }

private:
    double rollout(int s, const RobotState& state) override;

    double step_cost(const mjData* d, const double act_cmd[NUM_MUSCLES],
                     const double gait_ref[NUM_MUSCLES]);

    RobotState predict_state(const RobotState& state, int n_steps);

    MuscleParams   muscle_;
    CostWeights    cost_;
    GaitScheduler  gait_sched_;
    MotionCommand  cmd_;


    double last_compute_ms_ = 20.0;

    // real_act_ tracks the activation state at the most recently issued command.
    // predicted_activation_ is propagated through latency compensation and seeds rollouts.
    double real_act_[NUM_MUSCLES]            = {};
    double predicted_activation_[NUM_MUSCLES] = {};

    int    base_bid_ = 1;
};
