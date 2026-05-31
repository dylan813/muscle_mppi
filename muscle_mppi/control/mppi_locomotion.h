#pragma once

#include "base_mppi.h"
#include "muscle.h"

// Reference-free MPPI with cubic Hermite spline parameterisation.
//
// Search space: K position + velocity control nodes per joint (Sec. III-A of the paper).
// Muscle inverse replaces the paper's PD controller: given (q_des, dq_des) from the
// spline, agonist/antagonist activations are solved analytically from the Hill model.
//
// Storage layout for trajectory_ and best_trajectory_ (size K × 2 × NUM_JOINTS):
//   [k * 2 * NUM_JOINTS + j]             = θ^q_k[j]  (position node)
//   [k * 2 * NUM_JOINTS + NUM_JOINTS + j] = θ^v_k[j]  (velocity node)
//
// Storage layout for noise_ (size N_samples × K × 2 × NUM_JOINTS):
//   same stride per sample.

class MPPILocomotion : public BaseMPPI {
public:
    explicit MPPILocomotion(const std::string& task_name,
                            const std::string& yaml_path = "../utils/tasks.yaml");

    // Returns the first spline node of the best trajectory as joint targets.
    void update(const RobotState& state,
                double q_des_out[NUM_JOINTS],
                double dq_des_out[NUM_JOINTS]);

    // Muscle inverse + Hill model: (q_des, dq_des) → activations → torques.
    void compute_real_torques(const RobotState& state,
                              const double q_des[NUM_JOINTS],
                              const double dq_des[NUM_JOINTS],
                              double tau_out[NUM_JOINTS]);

    void set_command(const MotionCommand& cmd) { cmd_ = cmd; }
    const MotionCommand& command() const { return cmd_; }

    const MuscleState&  muscle_state()  const { return muscle_state_; }
    const MuscleParams& muscle_params() const { return muscle_; }

    struct CostBreakdown {
        double height = 0, orientation = 0, posture = 0;
        double contact_vel = 0, contact_force = 0;
        double vel_tracking = 0, terminal = 0;
        double total() const {
            return height + orientation + posture
                 + contact_vel + contact_force + vel_tracking + terminal;
        }
    };
    CostBreakdown diagnose_cost(const RobotState& state);

private:
    double rollout(int s, const RobotState& state) override;

    double step_cost(const mjData* d, int horizon_step);
    double terminal_cost(const mjData* d);

    RobotState predict_state(const RobotState& state, int n_steps);

    // Evaluate the Hermite spline at time `t_sec` (seconds from horizon start).
    // traj: pointer to a [K × 2 × NUM_JOINTS] node array.
    void hermite_eval(const double* traj, double t_sec,
                      double q_des[NUM_JOINTS], double dq_des[NUM_JOINTS]) const;

    // Compute agonist/antagonist activations to track (q_des, dq_des) via the Hill inverse.
    // Replaces the paper's PD controller.
    void activations_from_target(int j,
                                  double q_des, double dq_des,
                                  double q_cur, double dq_cur,
                                  double& act1_out, double& act2_out) const;

    // Clamp spline nodes: q to [phi_min, phi_max]; v via derivative constraint (Eq. 5).
    void clamp_spline_nodes(std::vector<double>& nodes) const;

    // Spline-aware noise sampling: fills noise_ for one MPPI iteration.
    void sample_spline_noise(int iter);

    MuscleParams   muscle_;
    MuscleState    muscle_state_;
    MotionCommand  cmd_;

    int    n_nodes_  = 10;   // K
    double dt_node_  = 0.1;  // Δt between nodes (seconds)

    double start_pos_[3] = {};
    std::vector<double> best_trajectory_;
    double best_cost_       = 1e9;
    double last_compute_ms_ = 20.0;
    double predicted_activation_[NUM_MUSCLES] = {};

    int    base_bid_       = 1;
    int    foot_body_ids_[4] = {};  // all four feet
    int    n_feet_         = 0;
    double f_nominal_      = 0.0;
};
