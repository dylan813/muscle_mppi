#pragma once

#include "base_mppi.h"
#include "muscle.h"

// Hill-model locomotion controller.
//
// Action space: muscle activation commands ∈ [-1, 1] per joint.
// No PD anywhere — the muscle model is the sole actuator model.
//
// Trajectory warm-started from gravity-compensation activation (MuJoCo qfrc_bias
// at nominal pose), so rollouts start near standing equilibrium rather than
// free-falling from zero activation.
//
// Rollout per substep:
//   a[j]  = tanh(act_cmd[j]) * alpha + a_prev[j] * (1-alpha)   // activation dynamics
//   vel   = clamp(1 - sign(a)*dq/dq_max, 0, 2)                 // Hill curve
//   tau   = a * tau_max * vel                                   // Hill torque
//   d->ctrl[j] = tau;  mj_step()
//
// Real-robot usage:
//   double act[16], tau[16];
//   mppi_.update(state, act);
//   mppi_.compute_real_torques(state, act, tau);
//   send tau via LowCmd PosStopF/tau mode
class MPPILocomotion : public BaseMPPI {
public:
    explicit MPPILocomotion(const std::string& task_name);

    // Plan one MPPI step. Returns 16-dim activation commands ∈ [-1, 1].
    void update(const RobotState& state, double activations_out[NUM_JOINTS]);

    // Apply Hill model for the real robot. Advances persistent muscle_state_.
    void compute_real_torques(const RobotState& state,
                              const double activations[NUM_JOINTS],
                              double tau_out[NUM_JOINTS]);

    const MuscleState&  muscle_state()  const { return muscle_state_; }
    const MuscleParams& muscle_params() const { return muscle_; }

private:
    double rollout(int s, const RobotState& state) override;
    double step_cost(const mjData* d);
    double terminal_cost(const mjData* d);

    MuscleParams muscle_;
    MuscleState  muscle_state_;  // real-robot persistent activation

    double joint_ref_[NUM_JOINTS] = {};  // reference pose for joint_deviation cost

    // Activation bounds: clamp trajectory to [-1, 1]
    static constexpr double ACT_MIN = -1.0;
    static constexpr double ACT_MAX =  1.0;
};
