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
    explicit MPPILocomotion(const std::string& task_name,
                           const std::string& yaml_path = "../utils/tasks.yaml");

    void update(const RobotState& state, double activations_out[NUM_JOINTS]);

    void compute_real_torques(const RobotState& state,
                              const double activations[NUM_JOINTS],
                              double tau_out[NUM_JOINTS]);

    void set_command(const MotionCommand& cmd) { cmd_ = cmd; }
    const MotionCommand& command() const { return cmd_; }

    // Load a reference activation trajectory produced by extract_reference.py.
    // Path to a CSV file: T rows × NUM_JOINTS columns, no header row.
    // ref_dt: timestep of each row in seconds (must match dial-mpc dt, default 0.02 s).
    // Once loaded, the act_reference cost weight is used; keep it 0.0 to disable.
    void load_reference(const std::string& csv_path, double ref_dt = 0.02);

    const MuscleState&  muscle_state()  const { return muscle_state_; }
    const MuscleParams& muscle_params() const { return muscle_; }

    struct CostBreakdown {
        double height = 0, orientation = 0, posture = 0;
        double contact_vel = 0, contact_force = 0;
        double vel_tracking = 0, act_effort = 0, terminal = 0;
        double total() const {
            return height + orientation + posture
                 + contact_vel + contact_force + vel_tracking + act_effort + terminal;
        }
    };
    CostBreakdown diagnose_cost(const RobotState& state);

private:
    double rollout(int s, const RobotState& state) override;

    double step_cost(const mjData* d,
                     const double act_cmd[NUM_JOINTS],
                     int horizon_step);

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

    // Base (trunk) body ID — for forward-offset height cost
    int base_bid_ = 1;

    // Wheel body IDs, axle directions, and nominal contact force
    int    wheel_body_ids_[4];     // FR, FL, RR, RL wheel_link body IDs
    double wheel_axle_[4][3] = {}; // spin-joint axle in body frame (unit vector)
    double f_nominal_;             // nominal vertical contact force per wheel (N)

    // Optional dial-mpc reference activation trajectory (LowState joint order).
    // Populated by load_reference(); empty when act_reference weight == 0.
    std::vector<double> ref_act_;   // flattened [ref_steps_ × NUM_JOINTS]
    int                 ref_steps_ = 0;
    double              ref_dt_    = 0.02;

    // Index into ref_act_ at the start of the current MPPI update.
    // Advances by n_skip each call; wraps mod ref_steps_ for looping gaits.
    int ref_offset_ = 0;

    // Returns the reference activation for rollout horizon step t.
    // Handles wrapping so a finite reference loops indefinitely.
    const double* ref_act_at(int t) const {
        if (ref_steps_ == 0) return nullptr;
        int idx = (ref_offset_ + t) % ref_steps_;
        return ref_act_.data() + idx * NUM_JOINTS;
    }

    static constexpr double ACT_MIN = -1.0;
    static constexpr double ACT_MAX =  1.0;
};
