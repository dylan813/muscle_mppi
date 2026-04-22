#include "ref_free_mppi.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <stdexcept>
#include <iostream>

static void mujoco_warning_noop(const char*) {}

// ---------------------------------------------------------------------------
// Cubic Hermite basis functions (Eqs. 3-4) and their s-derivatives.
//   h = [h00, h10, h01, h11]   (position coefficients)
//   dh = [dh00/ds, ...]        (velocity coefficients, divide by Δt for q̇)
// ---------------------------------------------------------------------------
static void hermite_basis(double s, double h[4], double dh[4]) {
    const double s2 = s * s, s3 = s2 * s;
    h[0]  =  2*s3 - 3*s2 + 1;   // h00
    h[1]  =    s3 - 2*s2 + s;   // h10
    h[2]  = -2*s3 + 3*s2;       // h01
    h[3]  =    s3 -   s2;       // h11
    dh[0] =  6*s2 - 6*s;
    dh[1] =  3*s2 - 4*s + 1;
    dh[2] = -6*s2 + 6*s;
    dh[3] =  3*s2 - 2*s;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
RefFreeMPPI::RefFreeMPPI(const std::string& task_name)
    : task_(get_task(task_name))
    , height_target_(task_.height_target)
    , rng_(std::random_device{}())
{
    mju_user_warning = mujoco_warning_noop;

    char error[1000];
    model_ = mj_loadXML(task_.model_path, nullptr, error, sizeof(error));
    if (!model_)
        throw std::runtime_error("Failed to load model: " + std::string(error));
    model_->opt.timestep = task_.dt;

    // Build per-actuator qpos/qvel address arrays.
    // MuJoCo body order ≠ LowCmd actuator order; use actuator_trnid to get the
    // correct joint index for each actuator slot j (= LowCmd motor index).
    for (int j = 0; j < NUM_JOINTS; ++j) {
        const int jid     = model_->actuator_trnid[2 * j];
        act_qpos_adr_[j]  = model_->jnt_qposadr[jid];
        act_qvel_adr_[j]  = model_->jnt_dofadr[jid];
    }

    const int N = task_.n_samples;
    const int K = task_.rf.K;

    // N rollout buffers + 1 for state prediction
    data_.resize(N + 1);
    for (auto& d : data_) d = mj_makeData(model_);

    // Spline control points — warm-start at nominal standing pose
    theta_q_nom_.assign(K * NUM_JOINTS, 0.0);
    theta_v_nom_.assign(K * NUM_JOINTS, 0.0);
    for (int k = 0; k < K; ++k)
        for (int j = 0; j < NUM_JOINTS; ++j)
            theta_q_nom_[k * NUM_JOINTS + j] = task_.nominal_pose[j];

    theta_q_best_ = theta_q_nom_;
    theta_v_best_ = theta_v_nom_;

    // Sample buffers
    sample_tq_.resize(N * K * NUM_JOINTS);
    sample_tv_.resize(N * K * NUM_JOINTS);
    costs_.resize(N);

    // Derived timing
    dt_spline_ = task_.rf.H_time / (K - 1);
    n_ctrl_    = static_cast<int>(std::round(task_.rf.H_time / task_.dt_ctrl));

    precompute_noise_schedule();

    // Wheel body IDs — Go2W foot/wheel bodies
    const char* wheel_names[4] = {
        "FR_wheel_link", "FL_wheel_link", "RR_wheel_link", "RL_wheel_link"
    };
    for (int i = 0; i < 4; ++i) {
        wheel_body_ids_[i] = mj_name2id(model_, mjOBJ_BODY, wheel_names[i]);

        // Store spin-joint axle in body frame.
        // Actuator indices NUM_LEG_JOINTS+i correspond to the four wheel joints.
        const int jid = model_->actuator_trnid[2 * (NUM_LEG_JOINTS + i)];
        wheel_axle_[i][0] = model_->jnt_axis[3 * jid + 0];
        wheel_axle_[i][1] = model_->jnt_axis[3 * jid + 1];
        wheel_axle_[i][2] = model_->jnt_axis[3 * jid + 2];
    }

    // Total robot mass
    total_mass_ = 0.0;
    for (int i = 0; i < model_->nbody; ++i)
        total_mass_ += model_->body_mass[i];
    if (total_mass_ < 1.0) total_mass_ = 15.0;  // fallback

    // Per-joint noise scales: leg vs wheel
    for (int j = 0; j < NUM_LEG_JOINTS; ++j) {
        scale_q_[j] = task_.rf.scale_q_leg;
        scale_v_[j] = task_.rf.scale_v_leg;
    }
    for (int j = NUM_LEG_JOINTS; j < NUM_JOINTS; ++j) {
        scale_q_[j] = task_.rf.scale_q_wheel;
        scale_v_[j] = task_.rf.scale_v_wheel;
    }

    std::cout << "RefFreeMPPI: model loaded, mass=" << total_mass_ << " kg\n";
    std::cout << "  K=" << K << " nodes, H=" << task_.rf.H_time
              << "s, I=" << task_.rf.I_iter << " iters, N=" << N << " samples\n";
}

RefFreeMPPI::~RefFreeMPPI() {
    for (auto* d : data_) mj_deleteData(d);
    mj_deleteModel(model_);
}

// ---------------------------------------------------------------------------
// Noise annealing schedule (Eq. 8)
//   σ^i_k = exp( -(I-i)/(β₁·I) - (K-k)/(β₂·K) )
// Larger σ for later nodes (k small after inversion = later horizon) and
// earlier iterations.  Stored as noise_sched_[i * K + k].
// ---------------------------------------------------------------------------
void RefFreeMPPI::precompute_noise_schedule() {
    const int I = task_.rf.I_iter;
    const int K = task_.rf.K;
    noise_sched_.resize(I * K);

    // Eq. (8) defines the per-node COVARIANCE Σ^i_{θ_k} = exp(...) · I.
    // Algorithm 1 samples θ ~ N(θ_nom, σ^i_k · scale), where the second argument
    // is the variance.  The std dev to multiply in code is sqrt(Σ) = exp(-0.5·(...)).
    //
    // Mapping: paper iterates i ∈ {I,...,1}; we use iter=0..I-1 with i = I - iter,
    // so (I-i) = iter.
    //   iter=0   → exp(0)          = 1.0  (full exploration, first iteration)
    //   iter=I-1 → exp(-0.5*(I-1)/(β₁I))  (refined, last iteration)
    //   k=0      → exp(-0.5/β₂)    (less noise near execution)
    //   k=K-1    → exp(-0.5/(β₂K)) ≈ 1  (more noise far in horizon)
    for (int iter = 0; iter < I; ++iter)
        for (int k = 0; k < K; ++k)
            noise_sched_[iter * K + k] = std::exp(
                - 0.5 * static_cast<double>(iter)   / (task_.rf.beta1 * I)
                - 0.5 * static_cast<double>(K - k)  / (task_.rf.beta2 * K));
}

// ---------------------------------------------------------------------------
// Cubic Hermite spline evaluation (Eq. 2)
// ---------------------------------------------------------------------------
void RefFreeMPPI::spline_eval(const std::vector<double>& tq,
                               const std::vector<double>& tv,
                               double t,
                               double* q_out,
                               double* dq_out) const {
    const int K = task_.rf.K;
    // Clamp to valid range and find segment
    t = std::clamp(t, 0.0, task_.rf.H_time);
    int k = static_cast<int>(t / dt_spline_);
    if (k >= K - 1) k = K - 2;

    const double s  = (t - k * dt_spline_) / dt_spline_;
    double h[4], dh[4];
    hermite_basis(s, h, dh);

    const int base_k  = k       * NUM_JOINTS;
    const int base_k1 = (k + 1) * NUM_JOINTS;
    const double dt   = dt_spline_;

    for (int j = 0; j < NUM_JOINTS; ++j) {
        const double qk  = tq[base_k  + j];
        const double vk  = tv[base_k  + j];
        const double qk1 = tq[base_k1 + j];
        const double vk1 = tv[base_k1 + j];

        // Eq. 2: q(t) = h00·qk + h10·Δt·vk + h01·qk1 + h11·Δt·vk1
        q_out[j] = h[0]*qk + h[1]*dt*vk + h[2]*qk1 + h[3]*dt*vk1;

        // q̇(t) = (1/Δt)·(dh00·qk + dh10·Δt·vk + dh01·qk1 + dh11·Δt·vk1)
        dq_out[j] = (dh[0]*qk + dh[1]*dt*vk + dh[2]*qk1 + dh[3]*dt*vk1) / dt;
    }
}

// ---------------------------------------------------------------------------
// Velocity clamping at spline nodes (Eq. 5)
// |θ^v_k| ≤ min(q_max - θ^q_k, θ^q_k - q_min) / (Δt/2)
// Skipped for wheel joints (unbounded limits ±1e9).
// ---------------------------------------------------------------------------
void RefFreeMPPI::clamp_node_velocities(std::vector<double>& tq,
                                         std::vector<double>& tv) const {
    const int K    = task_.rf.K;
    const double half_dt = dt_spline_ * 0.5;

    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < NUM_LEG_JOINTS; ++j) {
            const double q    = tq[k * NUM_JOINTS + j];
            const double dmax = std::min(task_.q_max[j] - q,
                                         q - task_.q_min[j]);
            if (dmax <= 0.0) continue;
            const double v_limit = dmax / half_dt;
            double& v = tv[k * NUM_JOINTS + j];
            v = std::clamp(v, -v_limit, v_limit);
        }
        // Wheel joints (j = NUM_LEG_JOINTS..NUM_JOINTS-1): no clamping
    }
}

// ---------------------------------------------------------------------------
// Shift τ_best forward by `steps` control steps (warm-start, Eq. 15).
// The paper shifts by ⌊Δt/dt_ctrl⌋ where Δt is the measured computation
// delay — must match predict_steps_ so the warm-start trajectory and the
// predicted state are anchored at the same future time.
// ---------------------------------------------------------------------------
void RefFreeMPPI::shift_best(int steps) {
    const int    K      = task_.rf.K;
    const double offset = steps * task_.dt_ctrl;
    const double H      = task_.rf.H_time;

    std::vector<double> new_q(K * NUM_JOINTS);
    std::vector<double> new_v(K * NUM_JOINTS);

    for (int k = 0; k < K; ++k) {
        double t = std::min(offset + k * dt_spline_, H);
        spline_eval(theta_q_best_, theta_v_best_, t,
                    &new_q[k * NUM_JOINTS],
                    &new_v[k * NUM_JOINTS]);
    }

    theta_q_best_ = new_q;
    theta_v_best_ = new_v;
}

// ---------------------------------------------------------------------------
// Write RobotState into mjData (qpos/qvel) then call mj_forward.
// qpos layout: [base_xyz(3), base_quat(4), joints(16)] = 23
// qvel layout: [base_linvel(3), base_angvel(3), joint_vel(16)] = 22
// ---------------------------------------------------------------------------
void RefFreeMPPI::set_mj_state(mjData* d, const RobotState& state) const {
    mj_resetData(model_, d);

    d->qpos[0] = state.pos[0];
    d->qpos[1] = state.pos[1];
    d->qpos[2] = state.pos[2];
    d->qpos[3] = state.quat[0];  // w
    d->qpos[4] = state.quat[1];  // x
    d->qpos[5] = state.quat[2];  // y
    d->qpos[6] = state.quat[3];  // z
    for (int j = 0; j < NUM_JOINTS; ++j) d->qpos[act_qpos_adr_[j]] = state.q[j];

    d->qvel[0] = state.vel[0];
    d->qvel[1] = state.vel[1];
    d->qvel[2] = state.vel[2];
    d->qvel[3] = state.gyro[0];
    d->qvel[4] = state.gyro[1];
    d->qvel[5] = state.gyro[2];
    for (int j = 0; j < NUM_JOINTS; ++j) d->qvel[act_qvel_adr_[j]] = state.dq[j];

    mj_forward(model_, d);
}

// ---------------------------------------------------------------------------
// State prediction (Sec. III-E / Eq. 14)
// Simulate `delay_steps` control steps forward from `state` using τ_best
// to compensate for computation delay.  delay_steps is set at construction
// and updated each solve based on measured timing.
// ---------------------------------------------------------------------------
RobotState RefFreeMPPI::predict_state(const RobotState& state) {
    mjData* d = data_.back();  // prediction slot (index N)
    set_mj_state(d, state);

    for (int step = 0; step < predict_steps_; ++step) {
        double q_des[NUM_JOINTS], dq_des[NUM_JOINTS];
        const double t = step * task_.dt_ctrl;
        spline_eval(theta_q_best_, theta_v_best_, t, q_des, dq_des);

        for (int sub = 0; sub < task_.substeps; ++sub) {
            for (int j = 0; j < NUM_JOINTS; ++j) {
                const double kp_j = (j < NUM_LEG_JOINTS) ? task_.kp : 0.0;
                const double kd_j = (j < NUM_LEG_JOINTS) ? task_.kd : 0.5;
                d->ctrl[j] = kp_j * (q_des[j]  - d->qpos[act_qpos_adr_[j]])
                           + kd_j * (dq_des[j] - d->qvel[act_qvel_adr_[j]]);
            }
            mj_step(model_, d);
        }
    }

    RobotState pred;
    pred.pos[0] = d->qpos[0];  pred.pos[1] = d->qpos[1];  pred.pos[2] = d->qpos[2];
    pred.quat[0] = d->qpos[3]; pred.quat[1] = d->qpos[4];
    pred.quat[2] = d->qpos[5]; pred.quat[3] = d->qpos[6];
    pred.vel[0] = d->qvel[0];  pred.vel[1] = d->qvel[1];  pred.vel[2] = d->qvel[2];
    pred.gyro[0] = d->qvel[3]; pred.gyro[1] = d->qvel[4]; pred.gyro[2] = d->qvel[5];
    for (int j = 0; j < NUM_JOINTS; ++j) {
        pred.q[j]  = d->qpos[act_qpos_adr_[j]];
        pred.dq[j] = d->qvel[act_qvel_adr_[j]];
    }
    pred.valid = true;
    return pred;
}

// ---------------------------------------------------------------------------
// Reference-free running cost (Eq. 17)
//
// c_t = w_h   · |z - z_des|
//      + w_ori · θ²           (θ = 2·acos|quat_w| = SO(3) rotation angle)
//      + w_q   · Σ(q - q0)²  (leg joints only)
//      + w_cvel· Σ_wheels |v_lateral|   (slip along wheel axle)
//      + w_cfrc· Σ_wheels |f_z - f0|
// ---------------------------------------------------------------------------
double RefFreeMPPI::step_cost(const mjData* d, double /*t_horizon*/) const {
    const CostWeights& w = task_.cost;
    double cost = 0.0;

    // Height (L1)
    cost += w.height * std::abs(d->qpos[2] - height_target_);

    // Orientation: angle of rotation from upright (Lie algebra log on SO(3))
    const double qw = std::abs(d->qpos[3]);
    const double theta = 2.0 * std::acos(std::min(qw, 1.0));
    cost += w.orientation * theta * theta;

    // Joint regularisation (L2, leg joints only — wheels rotate freely)
    for (int j = 0; j < NUM_LEG_JOINTS; ++j) {
        const double dq = d->qpos[act_qpos_adr_[j]] - task_.nominal_pose[j];
        cost += w.joint_reg * dq * dq;
    }

    // Velocity command tracking: penalise deviation from desired base vel
    if (w.vel_cmd > 0.0) {
        for (int k = 0; k < 3; ++k) {
            double dv = d->qvel[k] - w.vel_des[k];
            cost += w.vel_cmd * dv * dv;
        }
    }

    // Lateral slip penalty (L1) — velocity component along the wheel axle.
    // Rolling (velocity perpendicular to the axle) is free; only sideways slip
    // is penalised.  This avoids fighting against forward locomotion while still
    // discouraging traction loss.
    if (w.contact_vel > 0.0) {
        for (int i = 0; i < 4; ++i) {
            if (wheel_body_ids_[i] < 0) continue;
            mjtNum vel6[6];
            mj_objectVelocity(model_, d, mjOBJ_BODY, wheel_body_ids_[i], vel6, 0);
            // vel6 = [rot(3), lin(3)] in world frame (flg_local=0)
            const double vx = vel6[3], vy = vel6[4], vz = vel6[5];

            // Rotate axle from body frame to world frame using d->xmat (row-major).
            const mjtNum* xmat = d->xmat + 9 * wheel_body_ids_[i];
            const double ax = xmat[0]*wheel_axle_[i][0] + xmat[1]*wheel_axle_[i][1] + xmat[2]*wheel_axle_[i][2];
            const double ay = xmat[3]*wheel_axle_[i][0] + xmat[4]*wheel_axle_[i][1] + xmat[5]*wheel_axle_[i][2];
            const double az = xmat[6]*wheel_axle_[i][0] + xmat[7]*wheel_axle_[i][1] + xmat[8]*wheel_axle_[i][2];

            // Lateral slip = projection of wheel velocity onto the axle direction
            const double slip = ax*vx + ay*vy + az*vz;
            cost += w.contact_vel * std::abs(slip);
        }
    }

    // Contact force (L1 deviation from nominal load-sharing force per wheel).
    // mj_contactForce layout: [normal, tan1, tan2, spin, roll1, roll2] in contact frame.
    // force6[0] is the non-negative normal force scalar.  frame[2] is the z-component
    // of the contact normal in world frame — sign depends on geom1/geom2 assignment,
    // so we take abs() to always get a positive vertical force.
    if (w.contact_frc > 0.0) {
        const double f0 = total_mass_ * 9.81 / 4.0;

        double f_wheel[4] = {};
        for (int c = 0; c < d->ncon; ++c) {
            const int b1 = model_->geom_bodyid[d->contact[c].geom1];
            const int b2 = model_->geom_bodyid[d->contact[c].geom2];
            for (int i = 0; i < 4; ++i) {
                if (wheel_body_ids_[i] < 0) continue;
                if (b1 != wheel_body_ids_[i] && b2 != wheel_body_ids_[i]) continue;
                mjtNum force6[6];
                mj_contactForce(model_, d, c, force6);
                // mj_contactForce layout: [normal, tan1, tan2, spin, roll1, roll2].
                // force6[0] is the scalar normal force; frame[2] is the z-component
                // of the contact normal in world frame, giving the vertical projection.
                f_wheel[i] += std::abs(force6[0] * d->contact[c].frame[2]);
            }
        }

        for (int i = 0; i < 4; ++i) {
            if (wheel_body_ids_[i] < 0) continue;
            cost += w.contact_frc * std::abs(f_wheel[i] - f0);
        }
    }

    return cost;
}

// ---------------------------------------------------------------------------
// Terminal cost (Eq. 18): L1 horizontal displacement from desired target.
// p_target = pos0 + vel_des · H_time
// ---------------------------------------------------------------------------
double RefFreeMPPI::terminal_cost(const mjData* d, const double pos0[3]) const {
    const CostWeights& w = task_.cost;
    const double px = d->qpos[0] - (pos0[0] + w.vel_des[0] * task_.rf.H_time);
    const double py = d->qpos[1] - (pos0[1] + w.vel_des[1] * task_.rf.H_time);
    return w.terminal * (std::abs(px) + std::abs(py));
}

// ---------------------------------------------------------------------------
// Rollout one sample trajectory; returns total cost (or 1e6 on failure).
// ---------------------------------------------------------------------------
double RefFreeMPPI::rollout(int s,
                             const RobotState& state,
                             const std::vector<double>& tq,
                             const std::vector<double>& tv) {
    mjData* d = data_[s];
    set_mj_state(d, state);

    const double pos0[3] = {state.pos[0], state.pos[1], state.pos[2]};
    double total = 0.0;

    for (int step = 0; step < n_ctrl_; ++step) {
        const double t = step * task_.dt_ctrl;

        double q_des[NUM_JOINTS], dq_des[NUM_JOINTS];
        spline_eval(tq, tv, t, q_des, dq_des);

        for (int sub = 0; sub < task_.substeps; ++sub) {
            for (int j = 0; j < NUM_JOINTS; ++j) {
                const double kp_j = (j < NUM_LEG_JOINTS) ? task_.kp : 0.0;
                const double kd_j = (j < NUM_LEG_JOINTS) ? task_.kd : 0.5;
                d->ctrl[j] = kp_j * (q_des[j]  - d->qpos[act_qpos_adr_[j]])
                           + kd_j * (dq_des[j] - d->qvel[act_qvel_adr_[j]]);
            }
            mj_step(model_, d);
        }

        // Early exit on divergence
        if (!std::isfinite(d->qpos[2]) || d->qpos[2] < -0.5)
            return 1e6;

        total += step_cost(d, t);
    }

    total += terminal_cost(d, pos0);
    return std::isfinite(total) ? total : 1e6;
}

// ---------------------------------------------------------------------------
// Main MPPI update — Algorithm 1 from the paper.
// ---------------------------------------------------------------------------
void RefFreeMPPI::update(const RobotState& state,
                          double q_out[NUM_JOINTS],
                          double dq_out[NUM_JOINTS]) {
    if (!state.valid) {
        // Before first valid state: output nominal pose with zero velocity
        spline_eval(theta_q_best_, theta_v_best_, 0.0, q_out, dq_out);
        return;
    }

    const int N = task_.n_samples;
    const int K = task_.rf.K;
    const int I = task_.rf.I_iter;

    // --- State prediction (Eq. 14) — must come BEFORE shift_best ---
    // Simulates forward using the OLD τ_best (the prefix the robot has been
    // executing during computation), matching x_t = simulate(x_{t-Δt}, τ_best[0:⌊Δt/dt⌋], Δt).
    const RobotState state_pred = predict_state(state);

    // --- Warm-start: shift τ_best by the measured delay (Eq. 15) ---
    // Now that prediction is done with the old trajectory, advance the warm-start
    // so it is anchored at the same future time as state_pred.
    shift_best(predict_steps_);

    // --- Initialise nominal trajectory from τ_best (Algorithm 1, line 6) ---
    theta_q_nom_ = theta_q_best_;
    theta_v_nom_ = theta_v_best_;

    best_cost_ = 1e12;

    // ---- Inner optimisation loop (lines 7-20) ----
    for (int iter = 0; iter < I; ++iter) {

        // n=0: unperturbed nominal trajectory (Algorithm 1 — n starts at 0,
        // with τ_0 already set; noisy samples are n=1..N-1).
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < NUM_JOINTS; ++j) {
                sample_tq_[k * NUM_JOINTS + j] = theta_q_nom_[k * NUM_JOINTS + j];
                sample_tv_[k * NUM_JOINTS + j] = theta_v_nom_[k * NUM_JOINTS + j];
            }

        // n=1..N-1: noisy perturbed samples
        for (int n = 1; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                const double sigma = noise_sched_[iter * K + k];
                for (int j = 0; j < NUM_JOINTS; ++j) {
                    const int idx = n * K * NUM_JOINTS + k * NUM_JOINTS + j;
                    sample_tq_[idx] = theta_q_nom_[k * NUM_JOINTS + j]
                                    + sigma * scale_q_[j] * normal_(rng_);
                    sample_tv_[idx] = theta_v_nom_[k * NUM_JOINTS + j]
                                    + sigma * scale_v_[j] * normal_(rng_);
                }
            }
            // Clamp leg-joint velocities to bound-preserving rule (Eq. 5)
            std::vector<double> tq_n(sample_tq_.begin() + n * K * NUM_JOINTS,
                                     sample_tq_.begin() + (n+1) * K * NUM_JOINTS);
            std::vector<double> tv_n(sample_tv_.begin() + n * K * NUM_JOINTS,
                                     sample_tv_.begin() + (n+1) * K * NUM_JOINTS);
            clamp_node_velocities(tq_n, tv_n);
            std::copy(tq_n.begin(), tq_n.end(),
                      sample_tq_.begin() + n * K * NUM_JOINTS);
            std::copy(tv_n.begin(), tv_n.end(),
                      sample_tv_.begin() + n * K * NUM_JOINTS);
        }

        // Parallel rollouts — OpenMP thread pool, one buffer per sample index.
        // tq_n/tv_n are declared inside the loop so each thread gets its own copy.
        // costs_[n] writes are non-overlapping; state_pred and sample_tq_/tv_ are read-only.
        #pragma omp parallel for schedule(dynamic, 1)
        for (int n = 0; n < N; ++n) {
            std::vector<double> tq_n(sample_tq_.begin() + n * K * NUM_JOINTS,
                                     sample_tq_.begin() + (n+1) * K * NUM_JOINTS);
            std::vector<double> tv_n(sample_tv_.begin() + n * K * NUM_JOINTS,
                                     sample_tv_.begin() + (n+1) * K * NUM_JOINTS);
            costs_[n] = rollout(n, state_pred, tq_n, tv_n);
        }

        // --- Best-trajectory update (Sec. III-D) ---
        int n_best = static_cast<int>(
            std::min_element(costs_.begin(), costs_.end()) - costs_.begin());
        if (costs_[n_best] < best_cost_) {
            best_cost_ = costs_[n_best];
            const int base = n_best * K * NUM_JOINTS;
            theta_q_best_.assign(sample_tq_.begin() + base,
                                  sample_tq_.begin() + base + K * NUM_JOINTS);
            theta_v_best_.assign(sample_tv_.begin() + base,
                                  sample_tv_.begin() + base + K * NUM_JOINTS);
        }

        // --- MPPI weight update (Eqs. 10-12) ---
        const double cost_min = *std::min_element(costs_.begin(), costs_.end());
        const double cost_max = *std::max_element(costs_.begin(), costs_.end());
        const double range    = cost_max - cost_min;

        std::vector<double> weights(N);
        double weight_sum = 0.0;
        for (int n = 0; n < N; ++n) {
            double hat = (range > 0.0) ? (costs_[n] - cost_min) / range : 0.0;
            weights[n]  = std::exp(-hat / task_.lambda);
            weight_sum += weights[n];
        }

        // Importance-weighted average → update nominal control points
        std::vector<double> new_tq(K * NUM_JOINTS, 0.0);
        std::vector<double> new_tv(K * NUM_JOINTS, 0.0);
        for (int n = 0; n < N; ++n) {
            const double w = weights[n] / weight_sum;
            const int base  = n * K * NUM_JOINTS;
            for (int i = 0; i < K * NUM_JOINTS; ++i) {
                new_tq[i] += w * sample_tq_[base + i];
                new_tv[i] += w * sample_tv_[base + i];
            }
        }

        // Clamp the updated nominal velocities
        clamp_node_velocities(new_tq, new_tv);
        theta_q_nom_ = new_tq;
        theta_v_nom_ = new_tv;
    }

    // Execute from τ_best node 0 (position + velocity at t = 0)
    for (int j = 0; j < NUM_JOINTS; ++j) {
        q_out[j]  = theta_q_best_[j];
        dq_out[j] = theta_v_best_[j];
    }
}
