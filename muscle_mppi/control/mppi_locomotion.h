#pragma once

#include "base_mppi.h"
#include "muscle.h"
#include "gait_scheduler.h"

// -----------------------------------------------------------------------
// Hill-model locomotion controller with gait-reference tracking.
//
// Control inputs : muscle activation commands ∈ [-1, 1] (MPPI trajectory)
// Cost space     : body pose/velocity, joint positions, wheel velocities
//                  tracked against a GaitScheduler reference over the horizon
//
// The GaitScheduler generates a reference trajectory each update from the
// current robot state + motion command.  MPPI finds the activation sequence
// that keeps the simulated rollout close to this reference.
//
// This separates *what to do* (gait reference) from *how to actuate* (muscle
// activations), giving MPPI meaningful gradient on every step even when the
// robot drifts from the nominal pose.
//
// Usage:
//   mppi.set_command({.vx=0.3});       // e.g. walk forward at 0.3 m/s
//   mppi.update(state, activations);   // plan one MPPI step
//   mppi.compute_real_torques(state, activations, tau);
//   send tau via LowCmd tau mode
// -----------------------------------------------------------------------
class MPPILocomotion : public BaseMPPI {
public:
    explicit MPPILocomotion(const std::string& task_name);

    // Plan one MPPI step.  Returns 16-dim activation commands ∈ [-1, 1].
    void update(const RobotState& state, double activations_out[NUM_JOINTS]);

    // Apply Hill model for the real robot.  Advances persistent muscle_state_.
    void compute_real_torques(const RobotState& state,
                              const double activations[NUM_JOINTS],
                              double tau_out[NUM_JOINTS]);

    // Set motion command (call from control loop or nav stack)
    void set_command(const MotionCommand& cmd) { gait_sched_.set_command(cmd); }
    const MotionCommand& command() const { return gait_sched_.command(); }

    const MuscleState&  muscle_state()  const { return muscle_state_; }
    const MuscleParams& muscle_params() const { return muscle_; }

    // Per-term cost breakdown from a single zero-noise rollout.
    // Call after update() so gait_ref_ is populated.  Borrows data_[0].
    struct CostBreakdown {
        double height = 0, orientation = 0, lin_vel = 0, ang_vel = 0;
        double joint_track = 0, wheel_vel = 0, act_smooth = 0, terminal = 0;
        double total() const {
            return height + orientation + lin_vel + ang_vel
                 + joint_track + wheel_vel + act_smooth + terminal;
        }
    };
    CostBreakdown diagnose_cost(const RobotState& state);

private:
    double rollout(int s, const RobotState& state) override;

    // step_cost: deviation of simulated state from gait reference,
    //            plus activation smoothness regularisation.
    double step_cost(const mjData* d,
                     const StepReference& ref,
                     const double act_cmd[NUM_JOINTS],
                     const double act_prev[NUM_JOINTS]);

    double terminal_cost(const mjData* d, const StepReference& ref);

    MuscleParams  muscle_;
    MuscleState   muscle_state_;   // persistent activation state (real robot)
    GaitScheduler gait_sched_;
    GaitReference gait_ref_;       // pre-computed once per update(), shared across rollout threads

    static constexpr double ACT_MIN = -1.0;
    static constexpr double ACT_MAX =  1.0;
};
