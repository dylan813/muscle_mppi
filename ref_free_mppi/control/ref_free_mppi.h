#pragma once

#include <mujoco/mujoco.h>
#include <vector>
#include <random>
#include <string>
#include "../utils/tasks.h"

// Sensor data assembled from rt/lowstate + rt/sportmodestate
struct RobotState {
    double pos[3]  = {};
    double vel[3]  = {};
    double quat[4] = {1, 0, 0, 0};  // w, x, y, z
    double gyro[3] = {};
    double q[NUM_JOINTS]  = {};
    double dq[NUM_JOINTS] = {};
    bool   valid = false;
};

// Reference-Free Sampling-Based MPC — Schramm et al., arXiv:2511.19204
//
// Key algorithmic differences vs. vanilla MPPI (Algorithm 1 in the paper):
//   1. Cubic Hermite spline parameterisation of joint position AND velocity
//      control points, enabling dynamically consistent trajectories.
//   2. Diffusion-inspired noise annealing: larger noise for later nodes
//      and earlier inner iterations; smaller noise near execution.
//   3. Best-trajectory tracking (τ_best): executed actions always come from
//      the best fully-simulated trajectory, not an untested weighted mixture.
//   4. State prediction: simulate one control step ahead to compensate for
//      the ~20 ms computation delay at 50 Hz.
//   5. Reference-free cost: no gait priors; locomotion emerges from high-level
//      objectives only.
//
// Wheel joints (indices 12-15) are treated separately throughout:
//   - Larger noise scales (unbounded rotation).
//   - Velocity clamping (Eq. 5) skipped (limits are ±∞).
//   - Contact-velocity penalty uses wheel-body slip (not foot-planted stance).
class RefFreeMPPI {
public:
    explicit RefFreeMPPI(const std::string& task_name);
    ~RefFreeMPPI();

    void set_height_target(double z) { height_target_ = z; }

    // Update state-prediction horizon based on measured solve time.
    // Call after each solve: set_predict_delay(solve_ms / 1000.0)
    void set_predict_delay(double delay_s) {
        predict_steps_ = std::max(1,
            static_cast<int>(std::round(delay_s / task_.dt_ctrl)));
    }

    // One MPC step.  Fills q_out[NUM_JOINTS] and dq_out[NUM_JOINTS] with
    // the joint position and velocity targets to send via LowCmd.
    void update(const RobotState& state,
                double q_out[NUM_JOINTS],
                double dq_out[NUM_JOINTS]);

private:
    // ---- Spline helpers ----

    // Evaluate cubic Hermite spline at time t ∈ [0, H_time].
    // tq, tv: flat arrays sized [K × NUM_JOINTS].
    void spline_eval(const std::vector<double>& tq,
                     const std::vector<double>& tv,
                     double t,
                     double* q_out,
                     double* dq_out) const;

    // Clamp per-node velocities to satisfy the bound-preserving rule (Eq. 5).
    // Skipped for wheel joints (unbounded position limits).
    void clamp_node_velocities(std::vector<double>& tq,
                               std::vector<double>& tv) const;

    // Shift τ_best forward by `steps` control steps via spline re-sampling.
    // steps = predict_steps_ so warm-start and state prediction stay in sync.
    void shift_best(int steps);

    // ---- MPPI core ----

    // Precompute noise_sched_[I_iter × K]:  σ^i_k = exp(-(I-i)/(β₁I) - (K-k)/(β₂K))
    void precompute_noise_schedule();

    // Simulate one sample trajectory; returns total cost.
    // Uses data_[s] as scratch space.
    double rollout(int s,
                   const RobotState& state,
                   const std::vector<double>& tq,
                   const std::vector<double>& tv);

    // Reference-free running cost (Eq. 17) evaluated at current mjData.
    double step_cost(const mjData* d, double t_horizon) const;

    // Terminal cost (Eq. 18): L1 base displacement to desired target.
    double terminal_cost(const mjData* d, const double pos0[3]) const;

    // ---- State management ----

    // Write RobotState into mjData then call mj_forward.
    void set_mj_state(mjData* d, const RobotState& state) const;

    // Predict state one control step ahead using τ_best, compensating for
    // computation delay (Sec. III-E).
    RobotState predict_state(const RobotState& state);

    // ---- Members ----

    TaskConfig task_;
    mjModel*   model_ = nullptr;

    // data_[0..N-1]: parallel rollout scratch; data_[N]: state prediction
    std::vector<mjData*> data_;

    // Nominal and best-trajectory spline control points [K × NUM_JOINTS]
    std::vector<double> theta_q_nom_;   // τ_0: updated by MPPI each inner iter
    std::vector<double> theta_v_nom_;
    std::vector<double> theta_q_best_;  // τ_best: always best fully-simulated
    std::vector<double> theta_v_best_;
    double best_cost_ = 1e12;

    // Precomputed noise schedule [I_iter × K]
    std::vector<double> noise_sched_;

    // Per-sample control points [N × K × NUM_JOINTS]
    std::vector<double> sample_tq_;
    std::vector<double> sample_tv_;
    std::vector<double> costs_;

    double height_target_;

    double dt_spline_;  // Δt between consecutive spline nodes = H_time/(K-1)
    int    n_ctrl_;     // number of control steps in the rollout horizon

    // Per-actuator qpos/qvel address arrays — corrects LowCmd→qpos body ordering.
    // LowCmd actuator order (FR/FL/RR/RL) ≠ MuJoCo body-definition order.
    // Built in constructor from model_->actuator_trnid → jnt_qposadr / jnt_dofadr.
    int act_qpos_adr_[NUM_JOINTS] = {};
    int act_qvel_adr_[NUM_JOINTS] = {};

    // Base (trunk) body ID — used for height and orientation costs
    int base_bid_ = 1;

    // Wheel body IDs for contact penalties (FR/FL/RR/RL wheel_link)
    int wheel_body_ids_[4] = {-1, -1, -1, -1};

    // Wheel spin-joint axle directions in body frame (unit vectors).
    // Lateral slip = velocity component along this axis; rolling is perpendicular.
    double wheel_axle_[4][3] = {};

    // Total robot mass (kg), computed from model at construction
    double total_mass_ = 15.0;

    // Per-joint noise scale (position [rad] and velocity [rad/s])
    // Indexed by joint; wheels (12-15) get larger values.
    double scale_q_[NUM_JOINTS] = {};
    double scale_v_[NUM_JOINTS] = {};

    // Number of control steps to simulate ahead for delay compensation.
    // Initialised to 1; updated each solve via set_predict_delay().
    int predict_steps_ = 1;

    std::mt19937 rng_;
    std::normal_distribution<double> normal_{0.0, 1.0};
};
