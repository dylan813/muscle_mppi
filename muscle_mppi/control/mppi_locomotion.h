#pragma once

#include "base_mppi.h"
#include "gait_scheduler.h"
#include "muscle.h"

struct CostWeights {
    double pos_x       = 0.0;  // L1 world-frame x-position error
    double pos_y       = 0.0;  // L1 world-frame y-position error
    double pos_z       = 0.0;  // L1 z-position error (height)
    double orientation = 0.0;  // (1 - |q · q_ref|)^2 quaternion distance from upright
    double vel_x       = 0.0;  // body-frame x-velocity tracking (forward)
    double vel_y       = 0.0;  // body-frame y-velocity tracking (lateral)
    double vel_z       = 0.0;  // body-frame z-velocity tracking (vertical)
    double ang_vel     = 0.0;  // body-frame angular velocity damping (x, y, z equally)
    double gait_ref_weights[NUM_JOINTS] = {};  // per-joint activation tracking (replaces Q[7:19])
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
    const TaskConfig&   task_ref()      const { return task_; }

private:
    double rollout(int s, const RobotState& state) override;

    double step_cost(const mjData* d, const double gait_ref[NUM_MUSCLES]);

    MuscleParams   muscle_;
    CostWeights    cost_;
    GaitScheduler  gait_sched_;
    MotionCommand  cmd_;

    double gait_stiffness_  = 0.75;
    double last_compute_ms_ = 20.0;
    int    log_counter_     = 0;

    // Tracks the activation state at the most recently issued command.
    // Seeds rollouts — updated each update() after hill_compute_torques.
    double real_act_[NUM_MUSCLES] = {};

    int    base_bid_ = 1;
};
