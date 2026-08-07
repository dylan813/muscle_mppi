#pragma once

#include <string>
#include <unordered_map>

#include "base_mppi_pd.h"
#include "gait_scheduler_pd.h"

struct CostWeights {
    double pos_x       = 0.0;  // L1 world-frame x-position error
    double pos_y       = 0.0;  // L1 world-frame y-position error
    double pos_z       = 0.0;  // L1 z-position error (height)
    double orientation = 0.0;  // (1 - |q · q_ref|)^2 distance from the goal-facing quaternion
                                // (dynamic yaw+pitch toward the current goal, zero roll — see
                                // MPPILocomotionPD::update()'s goal_quat_ computation). Still
                                // penalizes roll/pitch deviation exactly as strictly as a fixed
                                // identity target would; only the yaw target becomes dynamic.
    double vel_x       = 0.0;  // body-frame x-velocity tracking (forward)
    double vel_y       = 0.0;  // body-frame y-velocity tracking (lateral)
    double vel_z       = 0.0;  // body-frame z-velocity tracking (vertical)
    double ang_vel     = 0.0;  // body-frame angular velocity damping (x, y, z equally)
    double gait_ref_weights[NUM_JOINTS]      = {};  // per-joint gait joint-angle tracking (Q_diag[7:19])
    double joint_vel_weights[NUM_JOINTS]     = {};  // per-joint gait joint-velocity tracking (Q_diag[25:37])
    double control_effort_weights[NUM_JOINTS] = {}; // per-joint penalty on the PD-implied torque (R_diag)
};

// Direct joint-space PD MPPI locomotion controller — the PD-actuated mirror of
// control/mppi_locomotion.h's muscle-actuated MPPILocomotion.
//
// Search space: q_des[j] per joint per horizon step (NUM_JOINTS × horizon),
// bounded by each joint's actual MJCF range (see BaseMPPIPD::action_lo_/hi_).
// tau[j] = kp[j]*(q_des[j]-q[j]) - kd[j]*dq[j] (RTWholeBodyMPPI's PD law) is
// computed directly in rollout()/update() — no activation dynamics, no
// implicit smoothing from a muscle model.
//
// Storage layout for trajectory_ (size horizon × NUM_JOINTS):
//   [t * NUM_JOINTS + j] = q_des[t][j]
class MPPILocomotionPD : public BaseMPPIPD {
public:
    explicit MPPILocomotionPD(const std::string& task_name,
                              const std::string& yaml_path = "../pd_mppi/utils/tasks_pd.yaml");

    // Run one MPPI solve; returns PD torques directly.
    void update(const RobotState& state, double tau_out[NUM_JOINTS]);

    // Call once per control tick, before update(). Advances to the next task
    // phase once the robot has stayed within the current phase's goal_thresh
    // for waiting_time consecutive in-threshold ticks (not reset if it drifts
    // back out in between — cumulative, matching RTWholeBodyMPPI's next_goal()).
    // Keeps running (updating dwelling_) even after the task's final phase is
    // reached. No-op only if the task has no phases at all.
    void advance_phase(const RobotState& state);

    void set_command(const MotionCommand& cmd) { cmd_ = cmd; }
    const MotionCommand& command() const { return cmd_; }

    // True once the final phase's dwell gate has passed (see advance_phase()).
    bool task_success() const { return task_success_; }

    const PDParams&    pd_params() const { return pd_; }
    const TaskConfig&  task_ref()  const { return task_; }
    // Commanded joint targets at the most recently issued command — the
    // PD-variant analogue of MPPILocomotion::activation().
    const double*       q_des()     const { return real_q_des_; }

private:
    double rollout(int s, const RobotState& state) override;

    // q_des[NUM_JOINTS]: this step's commanded joint targets (the raw MPPI
    // action), used for RTWholeBodyMPPI's u_error control-effort term — which
    // recomputes the PD expression with its own cost-shaping gains rather than
    // reusing the actuator's applied torque. See step_cost()'s definition.
    double step_cost(mjData* d, const double gait_ref_q[NUM_JOINTS],
                     const double gait_ref_dq[NUM_JOINTS], const double q_des[NUM_JOINTS]);

    // Point cmd_ and active_gait_ at task_.phases[idx].
    void activate_phase(int idx);

    // Trunk-origin position (world frame) and trunk linear velocity
    // (body-frame axes), matching RTWholeBodyMPPI's cost reference — see the
    // definition in mppi_locomotion_pd.cpp.
    void base_state(mjData* d, double pos[3], double vel_body[3]) const;

    PDParams       pd_;
    CostWeights    cost_;
    MotionCommand  cmd_;

    // All named gaits are loaded up front, keyed by their resolved TSV path so
    // that both the 4 canonical names and any per-phase gait_path override
    // share one map without key collisions. active_gait_ points at whichever
    // entry the current phase uses.
    std::unordered_map<std::string, GaitSchedulerPD> gaits_;
    GaitSchedulerPD* active_gait_ = nullptr;

    int  phase_index_  = 0;
    int  dwell_ticks_  = 0;      // consecutive-ish ticks spent within goal_thresh
    bool task_success_ = false;  // true once the final phase's dwell gate passes

    // True while settled at a waypoint (mid-dwell, or holding after
    // task_success_) — mirrors RTWholeBodyMPPI's Timer.waiting. Disables
    // goal-facing heading tracking (see goal_quat_) while true.
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
    // none) into task_.noise_sigma_act, which is what BaseMPPIPD::sample_actions()/
    // sample_actions_cubic() actually read.
    double base_noise_sigma_act_[NUM_JOINTS] = {};

    double last_compute_ms_ = 20.0;
    int    log_counter_     = 0;

    // Tracks the desired joint positions at the most recently issued command.
    // Seeds rollouts.
    double real_q_des_[NUM_JOINTS] = {};

    int    base_bid_ = 1;
};
