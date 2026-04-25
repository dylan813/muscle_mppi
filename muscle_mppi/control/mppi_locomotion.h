#pragma once

#include "base_mppi.h"
#include "muscle.h"

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

    int base_bid_ = 1;

    int    wheel_body_ids_[4];
    double wheel_axle_[4][3] = {};
    double f_nominal_;

    std::vector<double> reference_;
    int                 ref_steps_    = 0;
    int                 ref_n_joints_ = NUM_JOINTS;
    double              ref_dt_       = 0.02;
    int                 ref_offset_   = 0;

    // Wraps reference index so a finite trajectory loops indefinitely.
    const double* reference_at(int t) const {
        if (ref_steps_ == 0) return nullptr;
        int idx = (ref_offset_ + t) % ref_steps_;
        return reference_.data() + idx * ref_n_joints_;
    }

    static constexpr double ACT_MIN = -1.0;
    static constexpr double ACT_MAX =  1.0;
};
