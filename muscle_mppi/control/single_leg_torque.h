#pragma once

#include "base_mppi.h"

static_assert(NUM_MUSCLES == NUM_JOINTS,
    "SingleLegTorque requires NUM_MUSCLES == NUM_JOINTS. "
    "Compile with -DNUM_JOINTS=3 -DJOINT_OFFSET=3 -DNUM_MUSCLES=3.");

// Stage-1 MPPI: action space = joint torques (3D), no Hill model.
// Each trajectory slot holds one torque per joint in [-tau_max[j], tau_max[j]].
// Use this to validate the cost and MPPI hyperparameters before adding muscles.
class SingleLegTorque : public BaseMPPI {
public:
    explicit SingleLegTorque(const std::string& task_name,
                              const std::string& yaml_path = "../utils/tasks.yaml");

    // Run one MPPI solve. Writes the first-step torque command into tau_out.
    void update(const RobotState& state, double tau_out[NUM_JOINTS]);

private:
    double rollout(int s, const RobotState& state) override;

    double step_cost(const mjData* d, const double tau_cmd[NUM_JOINTS], int horizon_step);
    double terminal_cost(const mjData* d);

    int foot_body_id_ = -1;

    std::vector<double> best_traj_;
    double best_cost_ = 1e9;
};
