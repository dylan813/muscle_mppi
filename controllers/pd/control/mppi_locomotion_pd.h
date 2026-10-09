#pragma once

#include <string>

#include "base_mppi_pd.h"
#include "gait_scheduler_pd.h"
#include "../../common/motor.h"

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
// muscle/control/mppi_locomotion.h's muscle-actuated MPPILocomotion.
//
// Search space: q_des[j] per joint per horizon step (NUM_JOINTS × horizon),
// bounded by each joint's actual MJCF range (see BaseMPPIPD::action_lo_/hi_).
// tau[j] = kp[j]*(q_des[j]-q[j]) - kd[j]*dq[j] (RTWholeBodyMPPI's PD law),
// with no implicit smoothing from a muscle model. The kp term is computed here
// and the motor (common/motor.h) applies the kd term as its velocity feedback,
// in the rollouts and on the robot alike. If the task has an
// `activation_dynamics:` block, the PD law acts on q_des passed through the
// muscle variant's first-order activation filter instead (see real_q_filt_).
//
// Storage layout for trajectory_ (size horizon × NUM_JOINTS):
//   [t * NUM_JOINTS + j] = q_des[t][j]
class MPPILocomotionPD : public BaseMPPIPD {
public:
    explicit MPPILocomotionPD(const std::string& task_name,
                              const std::string& yaml_path = kDefaultTasksPdYaml);

    // Run one MPPI solve. Step 0 of the chosen plan becomes the joint target
    // actuate() holds until the next update().
    void update(const RobotState& state);

    // One physics step of the actuators (task_.motor.substeps per update()):
    // target filter and PD law from state, then the motor (common/motor.h).
    // ctrl: what to write to the robot's ctrl for this step.
    void actuate(const RobotState& state, double ctrl[NUM_JOINTS]);

    // Joint damping the sim adds to its physical damping for the motor
    // (Motor::implicit_damping()).
    double motor_damping(int j) const { return motor_.implicit_damping(j); }

    // Call once per control tick, before update(). Advances through the task's
    // phases (see PhaseSequencer::advance() in common/gait.h).
    void advance_phase(const RobotState& state) {
        if (phases_.advance(state)) apply_phase_noise();
    }

    void set_command(const MotionCommand& cmd) { phases_.set_command(cmd); }
    const MotionCommand& command() const { return phases_.command(); }

    // True once the final phase's dwell gate has passed (see advance_phase()).
    bool task_success() const { return phases_.task_success(); }

    const PDParams&    pd_params() const { return pd_; }
    const TaskConfig&  task_ref()  const { return task_; }
    // Commanded joint targets at the most recently issued command — the
    // PD-variant analogue of MPPILocomotion::activation().
    const double*       q_des()     const { return real_q_des_; }
    // Joint targets the PD law acted on at that command: q_des() after the
    // activation-dynamics filter (equal to q_des() when it's off).
    const double*       q_des_filt() const { return real_q_filt_; }
    // Motor command of the most recent actuate(): the PD torque, before the
    // torque limit and the motor's lag — the PD-variant analogue of
    // MPPILocomotion::torque().
    const double*       torque()    const { return last_tau_; }
    // Torque the motor applied over the most recent actuate()'s step, given
    // `after`, the state that step ended at.
    void applied_torque(const RobotState& after, double tau[NUM_JOINTS]) const {
        std::copy(real_motor_tau_, real_motor_tau_ + NUM_JOINTS, tau);
        if (pending_) motor_.applied(pending_ctrl_, after.dq, tau);
    }

private:
    double rollout(int s, const RobotState& state) override;

    // q_des[NUM_JOINTS]: this step's commanded joint targets (the raw MPPI
    // action), used for RTWholeBodyMPPI's u_error control-effort term — which
    // recomputes the PD expression with its own cost-shaping gains rather than
    // reusing the actuator's applied torque. See step_cost()'s definition.
    double step_cost(mjData* d, const double gait_ref_q[NUM_JOINTS],
                     const double gait_ref_dq[NUM_JOINTS], const double q_des[NUM_JOINTS]);

    // Trunk-origin position (world frame) and trunk linear velocity
    // (body-frame axes), matching RTWholeBodyMPPI's cost reference — see the
    // definition in mppi_locomotion_pd.cpp.
    void base_state(mjData* d, double pos[3], double vel_body[3]) const;

    // Write the active phase's noise_sigma_act override (or the YAML baseline,
    // if it has none) into task_.noise_sigma_act, which sample_actions() reads.
    void apply_phase_noise();

    PDParams       pd_;
    CostWeights    cost_;

    // Phase sequence, gaits and current command.
    PhaseSequencer<GaitSchedulerPD> phases_;

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

    // Tracks the desired joint positions at the most recently issued command.
    // Seeds rollouts.
    double real_q_des_[NUM_JOINTS] = {};

    // Activation-dynamics state: the filtered joint targets the PD law acts on
    // (equal to real_q_des_ when the filter is off). Persists across actuate()
    // calls and seeds every rollout, as real_act_ does in the muscle variant.
    double real_q_filt_[NUM_JOINTS] = {};

    // The PD law's kp term, kp·(q_filt − q), first advancing q_filt one physics
    // step toward q_des (or setting it to q_des when the filter is off). The
    // motor adds the kd term.
    void proportional_torques(const double q_des[NUM_JOINTS], double q_filt[NUM_JOINTS],
                              const double q[NUM_JOINTS], double tau_out[NUM_JOINTS]) const;

    // Advance d one control step toward q_des: per physics step, target filter
    // and PD law from d's joint state, the motor, then mj_step(). q_filt and
    // motor_tau carry the actuator state.
    void step_model(mjData* d, const double q_des[NUM_JOINTS], double q_filt[NUM_JOINTS],
                    double motor_tau[NUM_JOINTS]) const;

    // Finish the motor's pending step with the velocity it ended at (state.dq).
    void settle_motor(const RobotState& state);

    // Motor command of the most recent actuate() (logged by pd_mppi_sim).
    double last_tau_[NUM_JOINTS] = {};

    // Motor between the PD law and the robot. real_motor_tau_ is the torque it
    // has applied (seeds rollouts, like real_q_filt_). The step actuate() sent
    // stays pending until the next call reports the velocity it ended at.
    Motor  motor_;
    double real_motor_tau_[NUM_JOINTS] = {};
    double pending_ctrl_[NUM_JOINTS]   = {};
    bool   pending_ = false;
};
