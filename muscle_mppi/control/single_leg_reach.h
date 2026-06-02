#pragma once

#include "base_mppi.h"
#include "muscle.h"

#include <atomic>
#include <fstream>

static_assert(NUM_MUSCLES == NUM_JOINTS,
    "SingleLegReach requires NUM_MUSCLES == NUM_JOINTS. "
    "Compile with -DNUM_JOINTS=3 -DJOINT_OFFSET=3 -DNUM_MUSCLES=3.");

// Single-leg reach-and-hold controller using net-activation MPPI + Hill muscle model.
//
// Action space: u[j] ∈ [-1, 1] per joint (net activation).
//   u[j] > 0 → agonist dominates, u[j] < 0 → antagonist dominates.
// The Hill model maps u to antagonistic torque pairs internally; co-contraction
// is structurally impossible in the action space.
//
// Activation filter state (6 elements = 2 per joint) is tracked separately from
// the MPPI trajectory dimension (3 elements) and carried across update() calls.
class SingleLegReach : public BaseMPPI {
public:
    explicit SingleLegReach(const std::string& task_name,
                             const std::string& yaml_path = "../utils/tasks.yaml",
                             const std::string& log_dir   = "../../log");

    // Run one MPPI solve and return the Hill-model torque for the best trajectory.
    void update(const RobotState& state, double tau_out[NUM_JOINTS]);

    const MuscleParams& muscle_params() const { return muscle_; }
    const double* active_target() const { return active_target_; }
    int target_idx() const { return target_idx_; }

private:
    double rollout(int s, const RobotState& state) override;

    double step_cost(const mjData* d);
    double terminal_cost(const mjData* d);

    MuscleParams muscle_;

    // Activation filter state: [agonist_j0, antagonist_j0, agonist_j1, ...] (6 elements).
    // rollout_act_ is reset each solve from the synced real state.
    // real_act_ advances with each update() call to track live execution.
    double rollout_act_[2 * NUM_JOINTS] = {};
    double real_act_[2 * NUM_JOINTS]    = {};

    static constexpr double BASELINE           = 0.05;
    static constexpr double CONVERGENCE_THRESH = 0.001;  // metres (1 mm)
    static constexpr int    HOLD_STEPS         = 60;    // consecutive solves (~1.9 s at 31 ms/solve)

    int    foot_body_id_ = -1;
    double active_target_[3] = {};
    int    target_idx_        = 0;
    int    hold_count_        = 0;

    void maybe_advance_target(double foot_err);

    // Latency logging (CSV, one row per update() call)
    std::ofstream          lat_log_;
    long long              lat_call_count_ = 0;
    std::atomic<long long> lat_hill_us_{0};    // hill_net_act_torques inside rollouts
    std::atomic<long long> lat_mjstep_us_{0};  // mj_step inside rollouts
    std::atomic<long long> lat_cost_us_{0};    // step_cost + terminal_cost inside rollouts
};
