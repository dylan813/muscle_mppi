#pragma once

#include "base_mppi.h"
#include "muscle.h"

#include <atomic>
#include <fstream>

struct CostWeights {
    double height        = 0.0;
    double orientation   = 0.0;
    double posture       = 0.0;
    double contact_vel   = 0.0;
    double contact_force = 0.0;
    double terminal      = 0.0;
    double vel_cmd       = 0.0;
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
                            const std::string& yaml_path = "../utils/tasks.yaml",
                            const std::string& log_dir   = "../../analysis/log");

    // Run one MPPI solve; returns Hill-model torques directly.
    void update(const RobotState& state, double tau_out[NUM_JOINTS]);

    void set_command(const MotionCommand& cmd) { cmd_ = cmd; }
    const MotionCommand& command() const { return cmd_; }

    const MuscleParams& muscle_params() const { return muscle_; }

private:
    double rollout(int s, const RobotState& state) override;

    double step_cost(const mjData* d);
    double terminal_cost(const mjData* d);

    RobotState predict_state(const RobotState& state, int n_steps);

    void sample_activation_noise(int iter);

    MuscleParams   muscle_;
    CostWeights    cost_;
    MotionCommand  cmd_;

    double last_compute_ms_ = 20.0;

    // real_act_ tracks the activation state at the most recently issued command.
    // predicted_activation_ is propagated through latency compensation and seeds rollouts.
    double real_act_[NUM_MUSCLES]          = {};
    double predicted_activation_[NUM_MUSCLES] = {};

    double start_pos_[2] = {};

    int    base_bid_         = 1;
    int    foot_body_ids_[4] = {};
    int    n_feet_           = 0;

    // Latency logging (CSV, one row per update() call)
    std::ofstream          lat_log_;
    long long              lat_call_count_ = 0;
    std::atomic<long long> lat_hill_us_{0};
    std::atomic<long long> lat_mjstep_us_{0};
    std::atomic<long long> lat_cost_us_{0};
};
