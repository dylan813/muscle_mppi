#pragma once

#include "base_mppi.h"
#include "muscle.h"

#include <atomic>
#include <fstream>

static_assert(NUM_MUSCLES == 2 * NUM_JOINTS,
    "SingleLegReach requires NUM_MUSCLES == 2*NUM_JOINTS (direct per-muscle). "
    "Compile with -DNUM_JOINTS=3 -DJOINT_OFFSET=3 -DNUM_MUSCLES=6.");

// Single-leg reach-and-hold with direct per-muscle MPPI.
//
// Action space: act[m] ∈ [0, 1] for each of the NUM_MUSCLES muscles,
// laid out as [agonist_j0, antagonist_j0, agonist_j1, ...].
// Both muscles of each joint are commanded independently, allowing
// co-contraction. hill_compute_torques is used directly.
//
// Activation filter state (NUM_MUSCLES elements) is carried across update() calls.
class SingleLegReach : public BaseMPPI {
public:
    explicit SingleLegReach(const std::string& task_name,
                             const std::string& yaml_path = "../utils/tasks.yaml",
                             const std::string& log_dir   = "../../analysis/log");

    void update(const RobotState& state, double tau_out[NUM_JOINTS]);

    const MuscleParams& muscle_params() const { return muscle_; }
    const double* active_target() const { return active_target_; }
    int target_idx() const { return target_idx_; }

private:
    double rollout(int s, const RobotState& state) override;

    double step_cost(const mjData* d);
    double terminal_cost(const mjData* d);

    MuscleParams muscle_;

    // Activation filter state: [agonist_j0, antagonist_j0, ...] (NUM_MUSCLES elements).
    // rollout_act_ is reset each solve from the synced real state.
    // real_act_ advances with each update() call to track live execution.
    double rollout_act_[NUM_MUSCLES] = {};
    double real_act_[NUM_MUSCLES]    = {};

    static constexpr double CONVERGENCE_THRESH = 0.001;
    static constexpr int    HOLD_STEPS         = 60;

    int    foot_body_id_ = -1;
    double active_target_[3] = {};
    int    target_idx_        = 0;
    int    hold_count_        = 0;

    void maybe_advance_target(double foot_err);

    std::ofstream          lat_log_;
    long long              lat_call_count_ = 0;
    std::atomic<long long> lat_hill_us_{0};
    std::atomic<long long> lat_mjstep_us_{0};
    std::atomic<long long> lat_cost_us_{0};
};
