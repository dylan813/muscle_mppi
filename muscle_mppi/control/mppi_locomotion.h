#pragma once

#include <string>
#include <unordered_map>

#include "base_mppi.h"
#include "gait_scheduler.h"
#include "muscle.h"

struct CostWeights {
    double pos_x       = 0.0;  // L1 world-frame x-position error
    double pos_y       = 0.0;  // L1 world-frame y-position error
    double pos_z       = 0.0;  // L1 z-position error (height)
    double orientation = 0.0;  // (1 - |q · q_ref|)^2 distance from the goal-facing quaternion
                                // (dynamic yaw+pitch toward the current goal, zero roll — see
                                // MPPILocomotion::update()'s goal_quat_ computation). Still
                                // penalizes roll/pitch deviation exactly as strictly as a fixed
                                // identity target would; only the yaw target becomes dynamic.
    double vel_x       = 0.0;  // body-frame x-velocity tracking (forward)
    double vel_y       = 0.0;  // body-frame y-velocity tracking (lateral)
    double vel_z       = 0.0;  // body-frame z-velocity tracking (vertical)
    double ang_vel     = 0.0;  // body-frame angular velocity damping (x, y, z equally)
    double gait_ref_weights[NUM_JOINTS] = {};  // per-joint activation tracking (replaces Q[7:19])
};

// Reference-free MPPI with direct per-muscle activation (co-contraction capable). this is wrong
//
// Search space: act[m] ∈ [0, 1] per muscle per horizon step (NUM_MUSCLES × horizon).
// Layout: [agonist_j0, antagonist_j0, agonist_j1, ...] interleaved per joint.
// Activation dynamics in hill_compute_torques provide implicit trajectory smoothing.
//
// Storage layout for trajectory_ (size horizon × NUM_MUSCLES):
//   [t * NUM_MUSCLES + m] = act[t][m]

class MPPILocomotion : public BaseMPPI {
public:
    explicit MPPILocomotion(const std::string& task_name,
                            const std::string& yaml_path = "../utils/tasks.yaml");

    // Run one MPPI solve; returns Hill-model torques directly.
    void update(const RobotState& state, double tau_out[NUM_JOINTS]);

    // Call once per control tick, before update(). Advances to the next task
    // phase once the robot has stayed within the current phase's goal_thresh
    // for waiting_time consecutive in-threshold ticks (not reset if it drifts
    // back out in between — cumulative, matching RTWholeBodyMPPI's next_goal()).
    // Keeps running (updating dwelling_) even after the task's final phase is
    // reached — matches RTWholeBodyMPPI's next_goal(), which the driver keeps
    // calling every in-threshold tick forever. No-op only if the task has no
    // phases at all.
    void advance_phase(const RobotState& state);

    void set_command(const MotionCommand& cmd) { cmd_ = cmd; }
    const MotionCommand& command() const { return cmd_; }

    // True once the final phase's dwell gate has passed (see advance_phase()).
    bool task_success() const { return task_success_; }

    const MuscleParams& muscle_params() const { return muscle_; }
    const TaskConfig&   task_ref()      const { return task_; }
    const double*       activation()    const { return real_act_; }

private:
    double rollout(int s, const RobotState& state) override;

    double step_cost(mjData* d, const double gait_ref[NUM_MUSCLES]);

    // Point cmd_ and active_gait_ at task_.phases[idx].
    void activate_phase(int idx);

    // Whole-robot (trunk + legs) CoM position (world frame) and CoM
    // velocity (body-frame axes). Non-const mjData*: calls mj_subtreeVel,
    // which writes into d->subtree_linvel/subtree_angmom.
    void base_com_state(mjData* d, double com_pos[3], double com_vel_body[3]) const;

    MuscleParams   muscle_;
    CostWeights    cost_;
    MotionCommand  cmd_;

    // All named gaits are loaded up front (mirrors RTWholeBodyMPPI's self.gaits
    // dict), keyed by their resolved TSV path so that both the 4 canonical
    // names and any per-phase gait_path override share one map without key
    // collisions. active_gait_ points at whichever entry the current phase uses.
    std::unordered_map<std::string, GaitScheduler> gaits_;
    GaitScheduler* active_gait_ = nullptr;

    int  phase_index_  = 0;
    int  dwell_ticks_  = 0;      // consecutive-ish ticks spent within goal_thresh
    bool task_success_ = false;  // true once the final phase's dwell gate passes

    // True while settled at a waypoint (mid-dwell, or holding after
    // task_success_) — mirrors RTWholeBodyMPPI's Timer.waiting. Disables
    // goal-facing heading tracking (see goal_quat_) while true, matching
    // update()'s `not self.timer.waiting` check in the original.
    bool dwelling_ = false;

    // Per-tick goal-facing orientation target used by step_cost(): identity
    // when close to the goal or dwelling, otherwise R_z(yaw)*R_y(pitch) built
    // from the direction to cmd_.goal_pos (mirrors RTWholeBodyMPPI's
    // calculate_orientation_quaternion). Computed once per update() call,
    // held fixed across that tick's whole rollout batch.
    double goal_quat_[4] = {1.0, 0.0, 0.0, 0.0};

    // Snapshot of task_.noise_sigma_act as loaded from YAML, taken once in the
    // constructor before any phase mutates task_.noise_sigma_act. activate_phase()
    // writes the active phase's override (or this baseline, if the phase has
    // none) into task_.noise_sigma_act, which is what BaseMPPI::sample_noise()/
    // sample_noise_cubic() actually read — so "fall back to task-level value"
    // always means the true original baseline, not whatever a previous phase
    // left behind.
    double base_noise_sigma_act_[NUM_JOINTS] = {};

    double gait_stiffness_  = 0.75;
    double last_compute_ms_ = 20.0;
    int    log_counter_     = 0;

    // Tracks the activation state at the most recently issued command.
    // Seeds rollouts — updated each update() after hill_compute_torques.
    double real_act_[NUM_MUSCLES] = {};

    int    base_bid_ = 1;
};
