#pragma once

#include <limits>
#include <memory>
#include <string>

#include "base_mppi.h"
#include "gait_scheduler.h"
#include "muscle.h"
#include "observation.h"
#include "../../common/motor.h"

struct CostWeights {
    double pos_x       = 0.0;  // L1 world-frame x-position error
    double pos_y       = 0.0;  // L1 world-frame y-position error
    double pos_z       = 0.0;  // L1 z-position error (height)
    double orientation = 0.0;  // (1 - |q · q_ref|)^2 distance from the goal-facing quaternion
                                // (dynamic yaw+pitch toward the current goal, zero roll — see
                                // RolloutContext::goal_quat). Still
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

// Everything a rollout reads besides the plan it scores: the state and
// actuator state it starts from, and what step_cost() compares against.
// update() fills one per solve and rolls every sample out from it
// (solve_context()); score_plans() rolls plans out from any.
struct RolloutContext {
    RobotState          state;
    double              activation[NUM_MUSCLES] = {};   // filtered activations (real_act_ at the solve)
    MotorState          motor;                          // the motor's state (real_motor_ at the solve)
    std::vector<double> gait_ref;                       // horizon × NUM_MUSCLES; zeros without an active gait

    // Goal-facing orientation target: identity when close to the goal or
    // dwelling, otherwise R_z(yaw)*R_y(pitch) built from the direction to
    // command.goal_pos (mirrors RTWholeBodyMPPI's calculate_orientation_quaternion).
    double              goal_quat[4] = {1.0, 0.0, 0.0, 0.0};
    MotionCommand       command;                        // goal_pos, vx, vy
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
//
// With the task's `policy:` block, a learned policy's plans join MPPI's
// samples in every update() (sample_policy()).

class FlowPolicy;   // control/flow_policy.h

class MPPILocomotion : public BaseMPPI {
public:
    explicit MPPILocomotion(const std::string& task_name,
                            const std::string& yaml_path = kDefaultTasksYaml);
    ~MPPILocomotion() override;

    // Run one MPPI solve. Step 0 of the chosen plan becomes the activation
    // command actuate() holds until the next update().
    void update(const RobotState& state);

    // One physics step of the actuators (task_.motor.substeps per update()):
    // activation dynamics and Hill torques from state, then the motor
    // (common/motor.h). ctrl: what to write to the robot's ctrl for this step.
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

    const MuscleParams& muscle_params() const { return muscle_; }
    const TaskConfig&   task_ref()      const { return task_; }
    const double*       activation()    const { return real_act_; }
    // Motor command of the most recent actuate(): Hill torque minus driver
    // damping, before the torque limit and the motor's lag.
    const double*       torque()        const { return last_tau_; }
    // The activation command the most recent update() issued, before the
    // activation filter — activation() is what that command became.
    const double*       act_cmd()       const { return last_act_cmd_; }
    // Torque the motor applied over the most recent actuate()'s step, given
    // `after`, the state that step ended at.
    void applied_torque(const RobotState& after, double tau[NUM_JOINTS]) const {
        MotorState s = real_motor_;
        if (pending_) motor_.applied(pending_ctrl_, after.dq, s);
        std::copy(s.tau, s.tau + NUM_JOINTS, tau);
    }

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
    // Lowest cost among MPPI's samples and among the policy's (gpc's J_spc and
    // J_policy, gpc/training.py:86-95); NaN when that group has no samples.
    double mppi_min()   const { return mppi_min_; }
    double policy_min() const { return policy_min_; }
    // Share of the softmin weight on the policy's samples, in [0, 1]: how much
    // of the plan comes from them. NaN without a policy.
    double policy_weight() const { return policy_weight_; }
    // Cost terms of the lowest-cost sample among MPPI's and among the policy's,
    // each summed over the horizon as plan_cost() is; NaN when that group has
    // no samples. Shows which terms make one group's best sample cheaper.
    const CostTerms& mppi_best_terms()   const { return mppi_best_terms_; }
    const CostTerms& policy_best_terms() const { return policy_best_terms_; }
    // How far a plan's activation commands are from the gait reference: mean
    // |activation - reference| over the horizon and muscles. For the plan
    // update() chose, MPPI's best sample, and the policy's best sample (NaN
    // when that group has no samples).
    double ref_dev_plan()        const { return ref_dev_plan_; }
    double ref_dev_mppi_best()   const { return ref_dev_mppi_best_; }
    double ref_dev_policy_best() const { return ref_dev_policy_best_; }
    // Hill factors at the state update() started from: the ones the first
    // physics step's torque came from.
    HillFactors muscle_factors() const {
        HillFactors f;
        hill_factors(solve_.state.q, solve_.state.dq, muscle_, f);
        return f;
    }
    // What the most recent update() rolled every sample out from.
    const RolloutContext& solve_context() const { return solve_; }
    // MPPI's cost of each of n plans (n × horizon × NUM_MUSCLES, laid out as
    // plan()), rolled out from c as update() rolls out its samples, on OpenMP's
    // threads. Leaves the controller as it was (landscape_scorer).
    void score_plans(const RolloutContext& c, const double* plans, int n, double* costs);

    // Training data for the learned policy (learning/), from the most recent update():
    //   observation()  y at the state it planned from (control/observation.h)
    //   plan()         the plan it chose, horizon × NUM_MUSCLES as [t * NUM_MUSCLES + m];
    //                  step 0 is the command actuate() holds
    void observation(double y[OBS_DIM]);
    const std::vector<double>& plan() const { return trajectory_; }

private:
    double rollout(int s) override;   // sample s, from solve_

    // Roll plan (horizon × NUM_MUSCLES) out on d from c. Returns MPPI's cost of
    // it, summed over the horizon (1e6 if that isn't finite); terms gets the
    // same sum split by term.
    double rollout_plan(mjData* d, const double* plan, const RolloutContext& c, CostTerms& terms);

    // The policy's samples, rows n_samples onward of actions_, made in two parts
    // so update() can make them while MPPI's own samples roll out:
    //   policy_inputs()  on update()'s thread, after the shift: observation()'s
    //                    y, the plan MPPI samples around for every sample (as
    //                    gpc's BootstrappedPredictiveSampling passes its mean,
    //                    gpc/sampling.py:74), and noise from rng_
    //   sample_policy()  the policy on those inputs, on its own thread
    void policy_inputs();
    void sample_policy();

    // costs_[s] = rollout(s) for s in [begin, end), on OpenMP's threads.
    void roll_out(int begin, int end);

    // Mean |plan - solve_.gait_ref| over the horizon and muscles (ref_dev_*()).
    double ref_deviation(const double* plan) const;

    // Advance d one control step under activation command act_cmd: per physics
    // step, activation dynamics and Hill torques from d's joint state, the
    // motor, then mj_step(). activation and motor carry the actuator state.
    void step_model(mjData* d, const double act_cmd[NUM_MUSCLES], double activation[NUM_MUSCLES],
                    MotorState& motor) const;

    // Finish the motor's pending step with the velocity it ended at (state.dq).
    void settle_motor(const RobotState& state);

    // Cost of d's state against c's goal and command, and gait_ref (one step of
    // c.gait_ref). Call after mj_step() or set_mj_state().
    CostTerms step_cost(mjData* d, const double gait_ref[NUM_MUSCLES], const RolloutContext& c) const;

    // Recompute what step_cost() reads for d's qpos/qvel (see definition).
    void refresh_derived(mjData* d) const;

    // Whole-robot (trunk + legs) CoM position (world frame) and CoM
    // velocity (body-frame axes). Reads what refresh_derived() computed.
    void base_com_state(const mjData* d, double com_pos[3], double com_vel_body[3]) const;

    // Write the active phase's noise_sigma_act override (or the YAML baseline,
    // if it has none) into task_.noise_sigma_act, which sample_actions() reads.
    void apply_phase_noise();

    MuscleParams   muscle_;
    CostWeights    cost_;

    // Phase sequence, gaits and current command.
    PhaseSequencer<GaitScheduler> phases_;

    // task_.noise_sigma_act as loaded from YAML, snapshotted before any phase
    // override, so "no override" always restores the true baseline rather than
    // whatever a previous phase left behind.
    double base_noise_sigma_act_[NUM_JOINTS] = {};

    double last_compute_ms_ = 20.0;
    int    log_counter_     = 0;

    // Tracks the activation state at the most recently issued command.
    // Seeds rollouts — updated each actuate() by hill_compute_torques.
    double real_act_[NUM_MUSCLES] = {};

    // Motor command of the most recent actuate() (logged by mppi_sim).
    double last_tau_[NUM_JOINTS] = {};

    // Motor between the Hill model and the robot. real_motor_ is its state on
    // the robot (seeds rollouts, like real_act_). The step actuate() sent stays
    // pending until the next call reports the velocity it ended at.
    Motor      motor_;
    MotorState real_motor_;
    double     pending_ctrl_[NUM_JOINTS] = {};
    bool       pending_ = false;

    // The activation command that update() issued, i.e. what real_act_ filters
    // toward (logged by mppi_sim alongside the resulting activation).
    double last_act_cmd_[NUM_MUSCLES] = {};

    // What the most recent update() planned from: every rollout of that solve
    // started here (the goal-facing target held fixed across the batch), and
    // executed_cost() and plan_cost() score against it after update() returns.
    RolloutContext solve_;

    double sample_min_ = 0.0;
    double ess_        = 0.0;
    double mppi_min_   = std::numeric_limits<double>::quiet_NaN();
    double policy_min_ = std::numeric_limits<double>::quiet_NaN();
    double policy_weight_ = std::numeric_limits<double>::quiet_NaN();

    // Each rollout's cost terms, summed over the horizon ([total_samples()]),
    // and what the analysis getters above report from them.
    std::vector<CostTerms> sample_terms_;
    CostTerms mppi_best_terms_, policy_best_terms_;
    double ref_dev_plan_        = std::numeric_limits<double>::quiet_NaN();
    double ref_dev_mppi_best_   = std::numeric_limits<double>::quiet_NaN();
    double ref_dev_policy_best_ = std::numeric_limits<double>::quiet_NaN();

    std::unique_ptr<FlowPolicy> policy_;   // task_.policy; null without one

    // The policy's inputs for the update in progress (policy_inputs()).
    double              policy_y_[OBS_DIM] = {};
    std::vector<double> policy_prev_, policy_noise_;

    int    base_bid_ = 1;
};
