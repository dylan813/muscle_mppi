#pragma once

#include "base_mppi.h"
#include "muscle.h"

static_assert(NUM_MUSCLES == NUM_JOINTS,
    "SingleLegNetAct requires NUM_MUSCLES == NUM_JOINTS. "
    "Compile with -DNUM_JOINTS=3 -DJOINT_OFFSET=3 -DNUM_MUSCLES=3.");

// Stage-2 MPPI: action space = net joint activations u[j] ∈ [-1, 1].
//
// u[j] > 0 drives the agonist, u[j] < 0 drives the antagonist.
// The Hill model is used for physics: u is expanded to an antagonistic pair
// internally, so the muscle model is active without co-contraction in the
// action space. The activation filter state is 2*NUM_JOINTS (not NUM_MUSCLES)
// and is tracked separately from the MPPI trajectory.
class SingleLegNetAct : public BaseMPPI {
public:
    explicit SingleLegNetAct(const std::string& task_name,
                              const std::string& yaml_path = "../utils/tasks.yaml");

    // Run one MPPI solve. Returns the Hill-model torque for the best u trajectory.
    void update(const RobotState& state, double tau_out[NUM_JOINTS]);

    const MuscleParams& muscle_params() const { return muscle_; }

private:
    double rollout(int s, const RobotState& state) override;

    double step_cost(const mjData* d,
                     const double u[NUM_JOINTS],
                     const double tau_out[NUM_JOINTS],
                     const double tau_prev[NUM_JOINTS],
                     int horizon_step);
    double terminal_cost(const mjData* d);

    MuscleParams muscle_;

    // 6-element filter states (2*NUM_JOINTS, independent of NUM_MUSCLES).
    // rollout_act_ is reset per-rollout; real_act_ tracks the live execution.
    double rollout_act_[2 * NUM_JOINTS] = {};
    double real_act_[2 * NUM_JOINTS]    = {};

    // Co-activation floor: both muscles are at least BASELINE active.
    static constexpr double BASELINE = 0.05;

    int foot_body_id_ = -1;
};
