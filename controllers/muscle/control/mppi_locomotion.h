#pragma once

#include <string>

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

// MPPI locomotion with direct per-muscle activation (co-contraction capable),
// tracking the active phase's activation-gait reference in step_cost().
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
                            const std::string& yaml_path = kDefaultTasksYaml);

    // Run one MPPI solve; returns Hill-model torques directly.
    void update(const RobotState& state, double tau_out[NUM_JOINTS]);

    // Call once per control tick, before update(). Advances through the task's
    // phases (see PhaseSequencer::advance() in common/gait.h).
    void advance_phase(const RobotState& state) {
        if (phases_.advance(state)) apply_phase_noise();
    }

    void set_command(const MotionCommand& cmd) { phases_.set_command(cmd); }
    const MotionCommand& command() const { return phases_.command(); }

    // True once the final phase's dwell gate has passed (see advance_phase()).
    bool task_success() const { return phases_.task_success(); }

    const MuscleParams& muscle_params() const { return muscle_; }
    const TaskConfig&   task_ref()      const { return task_; }
    const double*       activation()    const { return real_act_; }
    const double*       torque()        const { return last_tau_; }
    // The activation command the most recent update() issued, before the
    // activation filter — activation() is what that command became.
    const double*       act_cmd()       const { return last_act_cmd_; }

private:
    double rollout(int s, const RobotState& state) override;

    double step_cost(mjData* d, const double gait_ref[NUM_MUSCLES]);

    // Whole-robot (trunk + legs) CoM position (world frame) and CoM
    // velocity (body-frame axes). Non-const mjData*: calls mj_subtreeVel,
    // which writes into d->subtree_linvel/subtree_angmom.
    void base_com_state(mjData* d, double com_pos[3], double com_vel_body[3]) const;

    // Write the active phase's noise_sigma_act override (or the YAML baseline,
    // if it has none) into task_.noise_sigma_act, which sample_noise() reads.
    void apply_phase_noise();

    MuscleParams   muscle_;
    CostWeights    cost_;

    // Phase sequence, gaits and current command.
    PhaseSequencer<GaitScheduler> phases_;

    // task_.noise_sigma_act as loaded from YAML, snapshotted before any phase
    // override, so "no override" always restores the true baseline rather than
    // whatever a previous phase left behind.
    double base_noise_sigma_act_[NUM_JOINTS] = {};

    // Per-tick goal-facing orientation target used by step_cost(): identity
    // when close to the goal or dwelling, otherwise R_z(yaw)*R_y(pitch) built
    // from the direction to command().goal_pos (mirrors RTWholeBodyMPPI's
    // calculate_orientation_quaternion). Computed once per update() call,
    // held fixed across that tick's whole rollout batch.
    double goal_quat_[4] = {1.0, 0.0, 0.0, 0.0};

    double last_compute_ms_ = 20.0;
    int    log_counter_     = 0;

    // Tracks the activation state at the most recently issued command.
    // Seeds rollouts — updated each update() after hill_compute_torques.
    double real_act_[NUM_MUSCLES] = {};

    // Joint torques returned by the most recent update() (logged by mppi_sim).
    double last_tau_[NUM_JOINTS] = {};

    // The activation command that update() issued, i.e. what real_act_ filters
    // toward (logged by mppi_sim alongside the resulting activation).
    double last_act_cmd_[NUM_MUSCLES] = {};

    int    base_bid_ = 1;
};
