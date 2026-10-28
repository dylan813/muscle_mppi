#include "mppi_locomotion.h"
#include "activation_gait.h"
#include "flow_policy.h"
#include "../../common/control_utils.h"
#include "../../common/harness.h"   // settle_standing() for the warm start

#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <future>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <omp.h>
#include <yaml-cpp/yaml.h>

// ============================================================================
// Named gaits
// ============================================================================
//
// Mirrors RTWholeBodyMPPI's GAIT_*_PATH constants (mppi_locomotion.py): a fixed
// set of categorical gaits, each the activation version of one joint-space
// source gait in controllers/pd/gaits/. The activation files are generated into
// controllers/muscle/gaits/ at startup whenever they're missing or were built
// with different muscle parameters/stiffness (muscle/control/activation_gait.h).
// A phase selects a gait by name (TaskPhase::desired_gait) or, as an escape
// hatch, an explicit TSV path (TaskPhase::gait_path, used as-is) — see
// resolve_gait_key() in common/gait.h.
static const std::unordered_map<std::string, std::string> kNamedGaitSources = {
    {"in_place",  "FAST_0_0_10cm"},
    {"walk",      "MED_0_1_10cm"},
    {"walk_fast", "FAST_0_1_10cm"},
    {"trot",      "MED_0_5_15cm"},
};

static NamedGaitPaths named_activation_gaits()
{
    NamedGaitPaths named;
    for (const auto& kv : kNamedGaitSources) named[kv.first] = activation_gait_path(kv.second);
    return named;
}
static const NamedGaitPaths kNamedGaits = named_activation_gaits();

// ============================================================================
// Constructor
// ============================================================================

MPPILocomotion::MPPILocomotion(const std::string& task_name, const std::string& yaml_path)
    : BaseMPPI(load_task(task_name, yaml_path))
{
    muscle_ = task_.muscle;
    solve_.gait_ref.assign(task_.horizon * NUM_MUSCLES, 0.0);
    sample_terms_.resize(total_samples());

    // BaseMPPI already initializes trajectory_, actions_, costs_ to the correct
    // sizes (horizon × NUM_MUSCLES). Activations are clamped to [0, 1] in
    // sample_actions() and update().

    base_bid_ = find_base_body(model_);

    {
        YAML::Node root = YAML::LoadFile(yaml_path);
        const YAML::Node& c = root[task_name]["cost"];
        cost_.pos_x       = c["pos_x"]       ? c["pos_x"].as<double>()       : 0.0;
        cost_.pos_y       = c["pos_y"]       ? c["pos_y"].as<double>()       : 0.0;
        cost_.pos_z       = c["pos_z"]       ? c["pos_z"].as<double>()       : 0.0;
        cost_.orientation = c["orientation"] ? c["orientation"].as<double>() : 0.0;
        cost_.vel_x       = c["vel_x"]       ? c["vel_x"].as<double>()       : 0.0;
        cost_.vel_y       = c["vel_y"]       ? c["vel_y"].as<double>()       : 0.0;
        cost_.vel_z       = c["vel_z"]       ? c["vel_z"].as<double>()       : 0.0;
        cost_.ang_vel     = c["ang_vel"]     ? c["ang_vel"].as<double>()     : 0.0;
        if (c["gait_ref_weights"]) {
            const auto& gw = c["gait_ref_weights"];
            for (int j = 0; j < NUM_JOINTS; ++j)
                cost_.gait_ref_weights[j] = gw[j].as<double>();
        }
    }

    // Make every named gait the phases use current for these muscle parameters
    // (regenerating it if needed), then load the task's gaits and activate
    // phase 0 (see PhaseSequencer::init()), then apply phase 0's noise override
    // against the YAML baseline.
    for (const TaskPhase& p : task_.phases)
        if (p.gait_path.empty())
            ensure_activation_gait(kNamedGaitSources.at(resolve_gait_key(p, kNamedGaits)), muscle_);
    std::memcpy(base_noise_sigma_act_, task_.noise_sigma_act, sizeof(base_noise_sigma_act_));
    phases_.init(task_.phases, kNamedGaits);
    apply_phase_noise();

    // Warm start: seed trajectory_ and real_act_ with activations that hold the
    // robot's standing pose. The planner model is stood up exactly as the sims
    // do (settle_standing(), common/harness.h — same stand-up, timestep, joint
    // damping and spawn height), so the pose and holding torques are the ones
    // MPPI actually takes over from. Each joint's activation pair is then the
    // point on its torque-balance line at muscle.stiffness (hill_invert_torque,
    // static so FV = 1 — the same inversion the gait-tracking cost uses).
    {
        const StandingEquilibrium stand = settle_standing(model_, task_.spawn_height_offset);
        double seed[NUM_MUSCLES] = {};
        for (int j = 0; j < NUM_JOINTS; ++j)
            hill_invert_torque(stand.q[j], /*dq=*/0.0, stand.tau[j], j, muscle_,
                               muscle_.stiffness, seed[2 * j], seed[2 * j + 1]);
        for (int t = 0; t < task_.horizon; ++t)
            for (int m = 0; m < NUM_MUSCLES; ++m)
                trajectory_[t * NUM_MUSCLES + m] = seed[m];
        std::memcpy(real_act_, seed, NUM_MUSCLES * sizeof(double));

        // Motor (common/motor.h): its velocity feedback is the driver's
        // damping, and it starts out applying the holding torque. From here on
        // it applies the torque limits, and its implicit share joins the joint
        // damping MuJoCo integrates.
        double lo[NUM_JOINTS], hi[NUM_JOINTS];
        for (int j = 0; j < NUM_JOINTS; ++j) {
            lo[j] = model_->actuator_ctrlrange[2 * j];
            hi[j] = model_->actuator_ctrlrange[2 * j + 1];
        }
        motor_.init(task_.motor, task_.motor.driver_kd, lo, hi);
        motor_.settle(stand.tau, real_motor_);
        for (int j = 0; j < NUM_JOINTS; ++j) {
            model_->dof_damping[act_qvel_adr_[j]] += motor_.implicit_damping(j);
            model_->actuator_ctrllimited[j] = 0;
        }
    }

    if (!task_.policy.path.empty())
        policy_ = std::make_unique<FlowPolicy>(task_.policy.path, task_.horizon, knot_x_, task_.policy.device);
}

MPPILocomotion::~MPPILocomotion() = default;

void MPPILocomotion::apply_phase_noise()
{
    // Per-phase noise_sigma_act override, falling back to the task-level
    // baseline — mirrors RTWholeBodyMPPI's next_goal(), which doubles thigh/
    // calf exploration noise specifically during trot phases.
    const TaskPhase* p = phases_.current_phase();
    if (p && p->has_noise_sigma_act)
        std::memcpy(task_.noise_sigma_act, p->noise_sigma_act, sizeof(task_.noise_sigma_act));
    else
        std::memcpy(task_.noise_sigma_act, base_noise_sigma_act_, sizeof(task_.noise_sigma_act));
}

// ============================================================================
// Rollout
// ============================================================================

double MPPILocomotion::rollout(int s)
{
    const int stride = task_.horizon * NUM_MUSCLES;
    return rollout_plan(data_[s], &actions_[s * stride], solve_, sample_terms_[s]);
}

double MPPILocomotion::rollout_plan(mjData* d, const double* plan, const RolloutContext& c, CostTerms& terms)
{
    set_mj_state(d, c.state);

    double activation[NUM_MUSCLES];
    std::memcpy(activation, c.activation, NUM_MUSCLES * sizeof(double));
    MotorState motor = c.motor;

    double total_cost = 0.0;
    terms = CostTerms{};

    for (int t = 0; t < task_.horizon; ++t) {
        step_model(d, &plan[t * NUM_MUSCLES], activation, motor);

        const CostTerms step = step_cost(d, &c.gait_ref[t * NUM_MUSCLES], c);
        total_cost += step.total();
        terms      += step;
    }

    return std::isfinite(total_cost) ? total_cost : 1e6;
}

void MPPILocomotion::step_model(mjData* d, const double act_cmd[NUM_MUSCLES],
                                double activation[NUM_MUSCLES], MotorState& motor) const
{
    for (int i = 0; i < task_.motor.substeps; ++i) {
        double q[NUM_JOINTS], dq[NUM_JOINTS], tau[NUM_JOINTS], ctrl[NUM_JOINTS], tau_cmd[NUM_JOINTS];
        for (int j = 0; j < NUM_JOINTS; ++j) {
            q[j]  = d->qpos[act_qpos_adr_[j]];
            dq[j] = d->qvel[act_qvel_adr_[j]];
        }

        hill_compute_torques(act_cmd, q, dq, muscle_, task_.motor.physics_dt, activation, tau);
        motor_.command(tau, dq, motor, ctrl, tau_cmd);

        for (int j = 0; j < model_->nu; ++j) d->ctrl[j] = 0.0;
        for (int j = 0; j < NUM_JOINTS; ++j) d->ctrl[j] = ctrl[j];

        mj_step(model_, d);

        for (int j = 0; j < NUM_JOINTS; ++j) dq[j] = d->qvel[act_qvel_adr_[j]];
        motor_.applied(ctrl, dq, motor);
    }
}

void MPPILocomotion::actuate(const RobotState& state, double ctrl[NUM_JOINTS])
{
    settle_motor(state);

    double tau[NUM_JOINTS];
    hill_compute_torques(last_act_cmd_, state.q, state.dq, muscle_, task_.motor.physics_dt, real_act_, tau);
    motor_.command(tau, state.dq, real_motor_, ctrl, last_tau_);

    std::memcpy(pending_ctrl_, ctrl, NUM_JOINTS * sizeof(double));
    pending_ = true;
}

void MPPILocomotion::settle_motor(const RobotState& state)
{
    if (!pending_) return;
    motor_.applied(pending_ctrl_, state.dq, real_motor_);
    pending_ = false;
}

// ============================================================================
// Cost function
// ============================================================================

// mj_step() integrates qpos/qvel but leaves every derived quantity (subtree_com,
// xmat, cvel, qfrc_bias, ...) at the PRE-step state. Recompute what
// step_cost() reads for the new state — otherwise the base and gait terms would
// score the previous step while orientation and the joint terms (read straight
// from qpos/qvel) score this one:
//   mj_kinematics + mj_comPos   subtree_com, xmat (position, orientation)
//   mj_comVel + mj_subtreeVel   subtree_linvel (velocity)
//   mj_rne without acceleration qfrc_bias, gravity + Coriolis (gait); needs
//                               mj_comPos and mj_comVel first
// The next mj_step recomputes all of these anyway, so the dynamics are unaffected.
void MPPILocomotion::refresh_derived(mjData* d) const
{
    mj_kinematics(model_, d);
    mj_comPos(model_, d);
    mj_comVel(model_, d);
    mj_subtreeVel(model_, d);
    mj_rne(model_, d, 0, d->qfrc_bias);
}

// Whole-robot CoM position (world frame) and CoM velocity (body-frame axes).
//
// base_bid_ is the trunk (root) body, so its subtree is the entire robot
// (trunk + all legs): d->subtree_com is the mass-weighted CoM of that whole
// subtree and d->subtree_linvel its velocity, in world-aligned axes, rotated
// here into the body frame.
void MPPILocomotion::base_com_state(const mjData* d, double com_pos[3], double com_vel_body[3]) const
{
    const double* com = d->subtree_com + base_bid_ * 3;
    com_pos[0] = com[0];
    com_pos[1] = com[1];
    com_pos[2] = com[2];

    world_to_body(d->xmat + base_bid_ * 9, d->subtree_linvel + base_bid_ * 3, com_vel_body);
}

CostTerms MPPILocomotion::step_cost(mjData* d, const double gait_ref[NUM_MUSCLES], const RolloutContext& c) const
{
    const CostWeights& w = cost_;
    CostTerms cost;

    refresh_derived(d);

    double pos[3], vel_body[3];
    base_com_state(d, pos, vel_body);

    cost.pos = w.pos_x * std::abs(pos[0] - c.command.goal_pos[0])
             + w.pos_y * std::abs(pos[1] - c.command.goal_pos[1])
             + w.pos_z * std::abs(pos[2] - c.command.goal_pos[2]);

    const double q_dot  = d->qpos[3]*c.goal_quat[0] + d->qpos[4]*c.goal_quat[1]
                         + d->qpos[5]*c.goal_quat[2] + d->qpos[6]*c.goal_quat[3];
    const double q_dist = 1.0 - std::abs(q_dot);
    cost.orient = w.orientation * q_dist * q_dist;

    if (w.vel_x > 0.0 || w.vel_y > 0.0 || w.vel_z > 0.0) {
        const double ex = vel_body[0] - c.command.vx;
        const double ey = vel_body[1] - c.command.vy;
        cost.vel = w.vel_x * ex*ex + w.vel_y * ey*ey + w.vel_z * vel_body[2]*vel_body[2];
    }

    if (w.ang_vel > 0.0) {
        const double wx = d->qvel[3], wy = d->qvel[4], wz = d->qvel[5];
        cost.ang_vel = w.ang_vel * (wx*wx + wy*wy + wz*wz);
    }

    for (int j = 0; j < NUM_JOINTS; ++j) {
        if (w.gait_ref_weights[j] == 0.0) continue;
        const double q_j   = d->qpos[act_qpos_adr_[j]];
        const double dq_j  = d->qvel[act_qvel_adr_[j]];
        const double tau_j = d->qfrc_bias[act_qvel_adr_[j]];
        double a1_imp, a2_imp;
        hill_invert_torque(q_j, dq_j, tau_j, j, muscle_, muscle_.stiffness, a1_imp, a2_imp);
        const double e1 = a1_imp - gait_ref[2 * j];
        const double e2 = a2_imp - gait_ref[2 * j + 1];
        cost.gait += w.gait_ref_weights[j] * (e1*e1 + e2*e2);
    }

    return cost;
}

CostTerms MPPILocomotion::executed_cost(const RobotState& state)
{
    mjData* d = data_[total_samples()];
    set_mj_state(d, state);
    return step_cost(d, solve_.gait_ref.data(), solve_);
}

CostTerms MPPILocomotion::plan_cost()
{
    CostTerms terms;
    rollout_plan(data_[total_samples()], trajectory_.data(), solve_, terms);
    return terms;
}

void MPPILocomotion::score_plans(const RolloutContext& c, const double* plans, int n, double* costs)
{
    const int stride = task_.horizon * NUM_MUSCLES;
    if (c.gait_ref.size() != static_cast<size_t>(stride))
        throw std::runtime_error("score_plans: the context's gait_ref is not horizon x NUM_MUSCLES");

    // Each thread rolls out on its own mjData, so data_ stays as update() left it.
    #pragma omp parallel
    {
        mjData* d = mj_makeData(model_);
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            CostTerms terms;
            costs[i] = rollout_plan(d, plans + i * stride, c, terms);
        }
        mj_deleteData(d);
    }
}

void MPPILocomotion::observation(double y[OBS_DIM])
{
    mjData* d = data_[total_samples()];
    set_mj_state(d, solve_.state);
    refresh_derived(d);

    double com[3], com_vel_body[3];
    base_com_state(d, com, com_vel_body);
    build_observation(solve_.state, com, solve_.activation, command().goal_pos, y);
}

// ============================================================================
// Policy samples
// ============================================================================

void MPPILocomotion::policy_inputs()
{
    const int K      = task_.policy.num_samples;
    const int stride = task_.horizon * NUM_MUSCLES;

    observation(policy_y_);

    policy_prev_.resize(K * stride);
    for (int k = 0; k < K; ++k) std::copy(trajectory_.begin(), trajectory_.end(), policy_prev_.begin() + k * stride);
    policy_noise_.resize(K * task_.n_knots * NUM_MUSCLES);
    for (double& v : policy_noise_) v = normal_(rng_);
}

void MPPILocomotion::sample_policy()
{
    const int K      = task_.policy.num_samples;
    const int stride = task_.horizon * NUM_MUSCLES;

    // Clamped as every sample is (gpc: hydrax clips all samples, policy's included).
    double* plans = &actions_[task_.n_samples * stride];
    policy_->sample(policy_y_, K, policy_prev_.data(), policy_noise_.data(), task_.policy.warm_start, plans);
    for (int i = 0; i < K * stride; ++i) plans[i] = std::clamp(plans[i], 0.0, 1.0);
}

void MPPILocomotion::roll_out(int begin, int end)
{
    #pragma omp parallel for schedule(dynamic)
    for (int s = begin; s < end; ++s)
        costs_[s] = rollout(s);
}

// Index of the lowest of costs[begin, end), or -1 for an empty range.
static int best_sample(const std::vector<double>& costs, int begin, int end)
{
    if (begin == end) return -1;
    return static_cast<int>(std::min_element(costs.begin() + begin, costs.begin() + end) - costs.begin());
}

double MPPILocomotion::ref_deviation(const double* plan) const
{
    const int size = task_.horizon * NUM_MUSCLES;
    double sum = 0.0;
    for (int i = 0; i < size; ++i) sum += std::abs(plan[i] - solve_.gait_ref[i]);
    return sum / size;
}

// ============================================================================
// Main solve
// ============================================================================

void MPPILocomotion::update(const RobotState& state)
{
    const auto t_start = std::chrono::steady_clock::now();

    settle_motor(state);

    if (!state.valid) {
        std::fill(last_act_cmd_, last_act_cmd_ + NUM_MUSCLES, 0.0);
        return;
    }

    // What this solve plans from: every rollout below starts here, with the
    // goal-facing orientation target held fixed across the whole batch (see
    // common/control_utils.h).
    solve_.state = state;
    std::memcpy(solve_.activation, real_act_, NUM_MUSCLES * sizeof(double));
    solve_.motor = real_motor_;
    for (int t = 0; t < task_.horizon; ++t) {
        double* ref = &solve_.gait_ref[t * NUM_MUSCLES];
        std::fill(ref, ref + NUM_MUSCLES, 0.0);
        if (phases_.active_gait()) phases_.active_gait()->get_phase(t, ref);
    }
    goal_facing_quat(command().goal_pos, state.pos, phases_.dwelling(), solve_.goal_quat);
    solve_.command = command();

    // Warm-start: shift trajectory_ forward by 1 step (see common/control_utils.h).
    const int stride = task_.horizon * NUM_MUSCLES;
    shift_trajectory(trajectory_, task_.horizon, NUM_MUSCLES);

    sample_actions();

    // MPPI's and the policy's samples alike, as gpc's PolicyAugmentedController
    // rolls out and weighs them together. The policy makes its samples on its
    // own thread (on the GPU, or one core) while MPPI's roll out; then its
    // samples roll out.
    const int n = total_samples();
    std::future<void> policy_samples;
    if (policy_) {
        policy_inputs();
        policy_samples = std::async(std::launch::async, &MPPILocomotion::sample_policy, this);
    }
    roll_out(0, task_.n_samples);
    if (policy_) {
        policy_samples.get();   // rethrows the policy's errors
        roll_out(task_.n_samples, n);
    }

    // Softmin weights (normalised), scaled by the range of MPPI's own samples
    // so the policy's can't flatten them (see softmin_weights()); cmin, ESS and
    // the group minima are diagnostics only.
    std::vector<double> weights;
    const double cmin = softmin_weights(costs_, task_.lambda, weights, task_.n_samples);

    double w_sq = 0.0;
    for (double w : weights) w_sq += w * w;
    sample_min_ = cmin;
    ess_        = 1.0 / w_sq;
    policy_weight_ = policy_ ? std::accumulate(weights.begin() + task_.n_samples, weights.end(), 0.0)
                             : std::numeric_limits<double>::quiet_NaN();

    // Each group's best sample: its cost, cost terms, and distance from the gait reference.
    const double    nan = std::numeric_limits<double>::quiet_NaN();
    const CostTerms no_terms{nan, nan, nan, nan, nan};
    const int mppi_best   = best_sample(costs_, 0, task_.n_samples);
    const int policy_best = best_sample(costs_, task_.n_samples, n);
    mppi_min_            = mppi_best   < 0 ? nan      : costs_[mppi_best];
    policy_min_          = policy_best < 0 ? nan      : costs_[policy_best];
    mppi_best_terms_     = mppi_best   < 0 ? no_terms : sample_terms_[mppi_best];
    policy_best_terms_   = policy_best < 0 ? no_terms : sample_terms_[policy_best];
    ref_dev_mppi_best_   = mppi_best   < 0 ? nan      : ref_deviation(&actions_[mppi_best * stride]);
    ref_dev_policy_best_ = policy_best < 0 ? nan      : ref_deviation(&actions_[policy_best * stride]);

    // Weighted average update. actions_ is already clamped (sample_actions(), sample_policy()).
    std::vector<double> new_traj(stride, 0.0);
    for (int s = 0; s < n; ++s) {
        const double w = weights[s];
        for (int idx = 0; idx < stride; ++idx)
            new_traj[idx] += w * actions_[s * stride + idx];
    }
    for (auto& v : new_traj) v = std::clamp(v, 0.0, 1.0);
    trajectory_ = std::move(new_traj);
    ref_dev_plan_ = ref_deviation(trajectory_.data());

    // Cost breakdown logging — every 50 updates (0.5 s of sim time at dt = 0.01).
    static constexpr int LOG_INTERVAL = 50;
    if (++log_counter_ % LOG_INTERVAL == 0) {
        const CostTerms c = plan_cost();
        std::printf("[cost] pos=%6.1f  orient=%6.1f  vel=%6.1f  gait=%6.1f  | sample_min=%6.1f  dt=%.1fms\n",
                    c.pos, c.orient, c.vel + c.ang_vel, c.gait, cmin, last_compute_ms_);
    }

    // Output: step 0 of the weighted-mean trajectory, held by actuate() until the next update().
    std::memcpy(last_act_cmd_, trajectory_.data(), NUM_MUSCLES * sizeof(double));

    if (phases_.active_gait()) phases_.active_gait()->advance();

    last_compute_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_start).count();
}

