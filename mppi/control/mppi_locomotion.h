#pragma once

#include "base_mppi.h"

// Locomotion controller — analogous to MPPI_quad's mppi_locomotion.py (MPPI class).
// Extends BaseMPPI with:
//   - PD-control rollout (position targets → torques → mj_step)
//   - Quadruped cost function (height, orientation, velocity, joint tracking, effort)
//   - MPPI update loop (sample → rollout → weight → shift → output)
//
// Constructed by task name: MPPI mppi("stand");
class MPPILocomotion : public BaseMPPI {
public:
    explicit MPPILocomotion(const std::string& task_name);

    // Run one MPPI iteration.
    // Returns 16-dim joint position targets. Send via LowCmd with kp=50, kd=3.5.
    void update(const RobotState& state, double q_targets_out[NUM_JOINTS]);

private:
    double rollout(int s, const RobotState& state) override;
    double step_cost(const mjData* d, const double q_des[NUM_JOINTS]);
    double terminal_cost(const mjData* d);

    double joint_ref_[NUM_JOINTS] = {};

    // Joint position limits from go2w.xml
    double q_min_[NUM_JOINTS] = {
        -1.0472, -1.5708, -2.7227,   // FR
        -1.0472, -1.5708, -2.7227,   // FL
        -1.0472, -0.5236, -2.7227,   // RR
        -1.0472, -0.5236, -2.7227,   // RL
        -1e9, -1e9, -1e9, -1e9       // wheels (unbounded)
    };
    double q_max_[NUM_JOINTS] = {
        1.0472,  3.4907, -0.83776,   // FR
        1.0472,  3.4907, -0.83776,   // FL
        1.0472,  4.5379, -0.83776,   // RR
        1.0472,  4.5379, -0.83776,   // RL
        1e9,  1e9,  1e9,  1e9        // wheels
    };
};
