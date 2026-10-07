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

// Cost split by term: each is the weighted penalty step_cost() adds for the
// matching CostWeights fields. Summed over steps for a rollout.
struct CostTerms {
    double pos     = 0.0;   // pos_x, pos_y, pos_z
    double orient  = 0.0;   // orientation
    double vel     = 0.0;   // vel_x, vel_y, vel_z
    double ang_vel = 0.0;   // ang_vel
    double gait    = 0.0;   // gait_ref_weights

    double total() const { return pos + orient + vel + ang_vel + gait; }

    CostTerms& operator+=(const CostTerms& o) {
        pos += o.pos; orient += o.orient; vel += o.vel; ang_vel += o.ang_vel; gait += o.gait;
        return *this;
    }
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

    // Analysis only: mppi_sim logs these; the controller never reads them.
    // All refer to the most recent update().
    //
    // Cost of `state`, the state after the executed step, scored as the
    // rollouts score their first step.
    CostTerms executed_cost(const RobotState& state);
    // The plan update() chose, rolled out over the horizon from the state
    // update() started from (as each sample was), summed over steps.
    CostTerms plan_cost();
    // Lowest sample cost, and the effective sample size of the softmin
    // weights, 1/Σw²: 1 when one sample decides the plan, n_samples when all count equally.
    double sample_min() const { return sample_min_; }
    double ess()        const { return ess_; }
    // Hill factors at the state update() started from: the ones the
    // executed torque (torque()) came from.
    HillFactors muscle_factors() const {
        HillFactors f;
        hill_factors(solve_state_.q, solve_state_.dq, muscle_, f);
        return f;
    }

private:
    double rollout(int s, const RobotState& state) override;

    // Advance d one step under activation command act_cmd: activation
    // dynamics + Hill torques from d's joint state, then mj_step().
    void step_model(mjData* d, const double act_cmd[NUM_MUSCLES], double activation[NUM_MUSCLES]) const;

    // Cost of d's state. Call after mj_step() or set_mj_state().
    CostTerms step_cost(mjData* d, const double gait_ref[NUM_MUSCLES]) const;

    // Recompute what step_cost() reads for d's qpos/qvel (see definition).
    void refresh_derived(mjData* d) const;

    // Whole-robot (trunk + legs) CoM position (world frame) and CoM
    // velocity (body-frame axes). Reads what refresh_derived() computed.
    void base_com_state(const mjData* d, double com_pos[3], double com_vel_body[3]) const;

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

    // What the most recent update() planned from, so executed_cost() and
    // plan_cost() score against it after update() returns.
    RobotState          solve_state_;
    double              solve_act_[NUM_MUSCLES] = {};
    std::vector<double> solve_gait_ref_;   // horizon × NUM_MUSCLES; zeros without an active gait

    double sample_min_ = 0.0;
    double ess_        = 0.0;

    int    base_bid_ = 1;
};
