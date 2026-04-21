#pragma once

#include "base_mppi.h"
#include "muscle.h"

// -----------------------------------------------------------------------
// Reference-free Hill-model locomotion controller.
//
// Control inputs : muscle activation commands ∈ [-1, 1] (MPPI trajectory)
// Cost space     : body height, orientation (upright), leg posture near
//                  nominal standing pose, activation smoothness, and a
//                  terminal displacement cost that drives the robot toward
//                  the commanded velocity goal.
//
// No gait references or predefined contact sequences are used.
// Locomotion emerges from the Hill model dynamics + cost minimization.
//
// Usage:
//   mppi.set_command({.vx=0.3});       // walk forward at 0.3 m/s
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
    void set_command(const MotionCommand& cmd) { cmd_ = cmd; }
    const MotionCommand& command() const { return cmd_; }

    const MuscleState&  muscle_state()  const { return muscle_state_; }
    const MuscleParams& muscle_params() const { return muscle_; }

    // Per-term cost breakdown from a single zero-noise rollout.
    // Call after update() so start_pos_ is populated.  Borrows data_[0].
    struct CostBreakdown {
        double height = 0, orientation = 0, posture = 0, act_smooth = 0, terminal = 0;
        double total() const {
            return height + orientation + posture + act_smooth + terminal;
        }
    };
    CostBreakdown diagnose_cost(const RobotState& state);

private:
    double rollout(int s, const RobotState& state) override;

    double step_cost(const mjData* d,
                     const double act_cmd[NUM_JOINTS],
                     const double act_prev[NUM_JOINTS]);

    double terminal_cost(const mjData* d);

    MuscleParams  muscle_;
    MuscleState   muscle_state_;   // persistent activation state (real robot)
    MotionCommand cmd_;
    double        start_pos_[3];   // captured at update() start for terminal cost

    static constexpr double ACT_MIN = -1.0;
    static constexpr double ACT_MAX =  1.0;
};
