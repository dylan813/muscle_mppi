#pragma once

#include "base_mppi.h"
#include "muscle.h"

// -----------------------------------------------------------------------
// Reference-free Hill-model locomotion controller.
//
// Cost function mirrors Schramm et al. 2026 (Eq. 17-18), Table II walking:
//   Running: w_h|Δz| + w_orient*θ² + w_q*||q-q0||² + w_cv*||v_c||₁ + w_cf*||f_c-f0||₁
//   Terminal: w_H * ||p_H - p_target||₁
//
// Action space: muscle activations ∈ [-1,1] → Hill model → torques.
// No gait references or predefined contact sequences.
// -----------------------------------------------------------------------
class MPPILocomotion : public BaseMPPI {
public:
    explicit MPPILocomotion(const std::string& task_name);

    void update(const RobotState& state, double activations_out[NUM_JOINTS]);

    void compute_real_torques(const RobotState& state,
                              const double activations[NUM_JOINTS],
                              double tau_out[NUM_JOINTS]);

    void set_command(const MotionCommand& cmd) { cmd_ = cmd; }
    const MotionCommand& command() const { return cmd_; }

    const MuscleState&  muscle_state()  const { return muscle_state_; }
    const MuscleParams& muscle_params() const { return muscle_; }

    struct CostBreakdown {
        double height = 0, orientation = 0, posture = 0;
        double contact_vel = 0, contact_force = 0;
        double act_smooth = 0, terminal = 0;
        double total() const {
            return height + orientation + posture
                 + contact_vel + contact_force + act_smooth + terminal;
        }
    };
    CostBreakdown diagnose_cost(const RobotState& state);

private:
    double rollout(int s, const RobotState& state) override;

    double step_cost(const mjData* d,
                     const double act_cmd[NUM_JOINTS],
                     const double act_prev[NUM_JOINTS]);

    double terminal_cost(const mjData* d);

    RobotState predict_state(const RobotState& state, int n_steps);

    MuscleParams         muscle_;
    MuscleState          muscle_state_;
    MotionCommand        cmd_;
    double               start_pos_[3];
    std::vector<double>  best_trajectory_;   // lowest-cost rollout seen this step
    double               best_cost_ = 1e9;
    double               last_compute_ms_ = 20.0;
    double               predicted_activation_[NUM_JOINTS] = {};

    // Wheel body IDs and nominal contact force (computed at construction)
    int    wheel_body_ids_[4];  // FR, FL, RR, RL wheel_link body IDs
    double f_nominal_;          // nominal vertical contact force per wheel (N)

    static constexpr double ACT_MIN = -1.0;
    static constexpr double ACT_MAX =  1.0;
};
