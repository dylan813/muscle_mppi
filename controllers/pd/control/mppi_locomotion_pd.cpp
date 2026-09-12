#include "mppi_locomotion_pd.h"
#include "../../common/control_utils.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <stdexcept>
#include <unordered_map>
#include <yaml-cpp/yaml.h>

// ============================================================================
// Named gaits
// ============================================================================
//
// A fixed set of categorical gaits, each backed by one pre-generated
// joint-angle gait TSV (position rows only, extracted from RTWholeBodyMPPI's
// original joint-space gait library — see pd/gaits/ and
// analysis/unit_tests/generate_pd_gaits.py) from the FAST/MED library in
// controllers/pd/gaits/. A phase selects a gait by name (TaskPhase::desired_gait)
// or, as an escape hatch, an explicit TSV path (TaskPhase::gait_path) — see
// resolve_gait_key() in common/gait.h. Same 4-gait mapping as the muscle variant
// (muscle/control/mppi_locomotion.cpp's kNamedGaits). Paths are repo-relative
// and resolved with repo_path() at load time.
static const char* GAIT_INPLACE_PATH   = "controllers/pd/gaits/FAST/gait_FAST_0_0_10cm.tsv";
static const char* GAIT_WALK_PATH      = "controllers/pd/gaits/MED/gait_MED_0_1_10cm.tsv";
static const char* GAIT_WALK_FAST_PATH = "controllers/pd/gaits/FAST/gait_FAST_0_1_10cm.tsv";
static const char* GAIT_TROT_PATH      = "controllers/pd/gaits/MED/gait_MED_0_5_15cm.tsv";

static const NamedGaitPaths kNamedGaits = {
    {"in_place",  GAIT_INPLACE_PATH},
    {"walk",      GAIT_WALK_PATH},
    {"walk_fast", GAIT_WALK_FAST_PATH},
    {"trot",      GAIT_TROT_PATH},
};

// ============================================================================
// Constructor
// ============================================================================

MPPILocomotionPD::MPPILocomotionPD(const std::string& task_name, const std::string& yaml_path)
    : BaseMPPIPD(load_task(task_name, yaml_path))
{
    pd_ = task_.pd;

    // Find base body.
    for (const char* name : {"trunk", "base", "base_link"}) {
        int bid = mj_name2id(model_, mjOBJ_BODY, name);
        if (bid >= 0) { base_bid_ = bid; break; }
    }

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
        if (c["joint_vel_weights"]) {
            const auto& jv = c["joint_vel_weights"];
            for (int j = 0; j < NUM_JOINTS; ++j)
                cost_.joint_vel_weights[j] = jv[j].as<double>();
        }
        if (c["control_effort_weights"]) {
            const auto& ce = c["control_effort_weights"];
            for (int j = 0; j < NUM_JOINTS; ++j)
                cost_.control_effort_weights[j] = ce[j].as<double>();
        }
    }

    // Load the task's gaits and activate phase 0 (see PhaseSequencer::init()).
    phases_.init(task_.phases, kNamedGaits, task_.noise_sigma_act, /*default_goal_z=*/0.0);

    // Seed trajectory_ and real_q_des_ with the task's nominal pose — sensible
    // cold-start for a joint-position action space (unlike the muscle variant,
    // there's no constraint-line posture solve; the nominal pose is already a
    // valid joint-angle target).
    for (int t = 0; t < task_.horizon; ++t)
        for (int j = 0; j < NUM_JOINTS; ++j)
            trajectory_[t * NUM_JOINTS + j] = task_.nominal_pose[j];
    std::memcpy(real_q_des_, task_.nominal_pose, sizeof(real_q_des_));
}

// ============================================================================
// Rollout
// ============================================================================

double MPPILocomotionPD::rollout(int s, const RobotState& state)
{
    mjData* d = data_[s];
    set_mj_state(d, state);

    const int stride  = task_.horizon * NUM_JOINTS;
    double total_cost = 0.0;

    for (int t = 0; t < task_.horizon; ++t) {
        double tau_out[NUM_JOINTS];
        double q_cur[NUM_JOINTS], dq_cur[NUM_JOINTS];
        for (int j = 0; j < NUM_JOINTS; ++j) {
            q_cur[j]  = d->qpos[act_qpos_adr_[j]];
            dq_cur[j] = d->qvel[act_qvel_adr_[j]];
        }

        for (int j = 0; j < NUM_JOINTS; ++j) {
            const double q_des = actions_[s * stride + t * NUM_JOINTS + j];
            tau_out[j] = unitree_pd_torque(pd_.kp[j], pd_.kd[j], q_des, q_cur[j],
                                           /*dq_des=*/0.0, dq_cur[j], /*tau_ff=*/0.0);
        }

        for (int j = 0; j < model_->nu; ++j) d->ctrl[j] = 0.0;
        for (int j = 0; j < NUM_JOINTS; ++j) d->ctrl[j] = tau_out[j];

        mj_step(model_, d);

        double gait_ref_q[NUM_JOINTS] = {}, gait_ref_dq[NUM_JOINTS] = {};
        if (phases_.active_gait()) phases_.active_gait()->get_phase(t, gait_ref_q, gait_ref_dq);
        total_cost += step_cost(d, gait_ref_q, gait_ref_dq,
                                &actions_[s * stride + t * NUM_JOINTS]);
    }

    return std::isfinite(total_cost) ? total_cost : 1e6;
}

// ============================================================================
// Cost function
// ============================================================================

// Trunk-origin position (world frame) and trunk linear velocity (body-frame axes).
void MPPILocomotionPD::base_state(mjData* d, double pos[3], double vel_body[3]) const
{
    // Track the raw free-joint trunk origin, matching RTWholeBodyMPPI:
    // quadruped_cost_np's x[:, :3] is qpos[0:3] and x[:, 19:22] is qvel[0:3]
    // (mppi_locomotion.py:240, 287-288), taken straight out of the rollout
    // state — never a center-of-mass quantity. MuJoCo's free joint reports
    // qvel[0:3] in world axes, so rotate into the body frame by R^T (xmat is
    // body->world), which is what RTWholeBodyMPPI's
    // batch_world_to_local_velocity does via rotation.inv().apply().
    pos[0] = d->qpos[0];
    pos[1] = d->qpos[1];
    pos[2] = d->qpos[2];

    world_to_body(d->xmat + base_bid_ * 9, d->qvel, vel_body);
}

double MPPILocomotionPD::step_cost(mjData* d, const double gait_ref_q[NUM_JOINTS],
                                   const double gait_ref_dq[NUM_JOINTS],
                                   const double q_des[NUM_JOINTS])
{
    const CostWeights& w = cost_;
    double cost = 0.0;

    double pos[3], vel_body[3];
    base_state(d, pos, vel_body);

    cost += w.pos_x * std::abs(pos[0] - command().goal_pos[0]);
    cost += w.pos_y * std::abs(pos[1] - command().goal_pos[1]);
    cost += w.pos_z * std::abs(pos[2] - command().goal_pos[2]);

    const double q_dot  = d->qpos[3]*goal_quat_[0] + d->qpos[4]*goal_quat_[1]
                         + d->qpos[5]*goal_quat_[2] + d->qpos[6]*goal_quat_[3];
    const double q_dist = 1.0 - std::abs(q_dot);
    cost += w.orientation * q_dist * q_dist;

    if (w.vel_x > 0.0 || w.vel_y > 0.0 || w.vel_z > 0.0) {
        const double ex = vel_body[0] - command().vx;
        const double ey = vel_body[1] - command().vy;
        cost += w.vel_x * ex*ex + w.vel_y * ey*ey + w.vel_z * vel_body[2]*vel_body[2];
    }

    if (w.ang_vel > 0.0) {
        const double wx = d->qvel[3], wy = d->qvel[4], wz = d->qvel[5];
        cost += w.ang_vel * (wx*wx + wy*wy + wz*wz);
    }

    for (int j = 0; j < NUM_JOINTS; ++j) {
        if (w.gait_ref_weights[j] == 0.0) continue;
        const double q_j = d->qpos[act_qpos_adr_[j]];
        const double e   = q_j - gait_ref_q[j];
        cost += w.gait_ref_weights[j] * e * e;
    }

    // Joint-velocity gait tracking (mirrors RTWholeBodyMPPI's Q_diag[25:37]).
    for (int j = 0; j < NUM_JOINTS; ++j) {
        if (w.joint_vel_weights[j] == 0.0) continue;
        const double dq_j = d->qvel[act_qvel_adr_[j]];
        const double e    = dq_j - gait_ref_dq[j];
        cost += w.joint_vel_weights[j] * e * e;
    }

    // Control-effort regularization (RTWholeBodyMPPI's R_diag term), computed
    // exactly as quadruped_cost_np does it (mppi_locomotion.py:220-221, 236):
    //     kp = 50; kd = 3
    //     u_error = kp * (u - x_joint) - kd * v_joint
    // Two deliberate details, both matching the reference rather than this
    // task's own actuator: (1) the cost uses kp=50, NOT the 55 the physical
    // actuator applies — RTWholeBodyMPPI hardcodes a separate cost-shaping
    // gain pair and we mirror it rather than "correcting" it; (2) x_joint /
    // v_joint are the POST-step joint state (Python scores the rollout state
    // recorded after the control was applied), which is what d holds here
    // since step_cost() runs after mj_step().
    static constexpr double kCostKp = 50.0;
    static constexpr double kCostKd = 3.0;
    for (int j = 0; j < NUM_JOINTS; ++j) {
        if (w.control_effort_weights[j] == 0.0) continue;
        const double u_error = kCostKp * (q_des[j] - d->qpos[act_qpos_adr_[j]])
                             - kCostKd * d->qvel[act_qvel_adr_[j]];
        cost += w.control_effort_weights[j] * u_error * u_error;
    }

    return cost;
}

// ============================================================================
// Main solve
// ============================================================================

void MPPILocomotionPD::update(const RobotState& state, double tau_out[NUM_JOINTS])
{
    const auto t_start = std::chrono::steady_clock::now();

    if (!state.valid) {
        for (int j = 0; j < NUM_JOINTS; ++j)
            tau_out[j] = unitree_pd_torque(pd_.kp[j], pd_.kd[j], real_q_des_[j], state.q[j],
                                           /*dq_des=*/0.0, state.dq[j], /*tau_ff=*/0.0);
        return;
    }

    // Goal-facing orientation target for this tick's cost, held fixed across
    // the whole rollout batch below (see common/control_utils.h).
    goal_facing_quat(command().goal_pos, state.pos, phases_.dwelling(), goal_quat_);

    // Warm-start: shift trajectory_ forward by 1 step (see common/control_utils.h).
    const int stride = task_.horizon * NUM_JOINTS;
    shift_trajectory(trajectory_, task_.horizon, NUM_JOINTS);

    sample_actions();

    #pragma omp parallel for schedule(dynamic)
    for (int s = 0; s < task_.n_samples; ++s)
        costs_[s] = rollout(s, state);

    // Softmin weights (normalised); cmin logged below as a diagnostic only.
    std::vector<double> weights;
    const double cmin = softmin_weights(costs_, task_.lambda, weights);

    // Weighted average update. actions_ is already clamped (built in
    // sample_actions()/sample_actions_cubic()), so no per-sample re-clamp
    // here — matches RTWholeBodyMPPI's perturb_action() clipping once and
    // reusing that same array for both rollout and this weighted average.
    std::vector<double> new_traj(stride, 0.0);
    for (int s = 0; s < task_.n_samples; ++s) {
        const double w = weights[s];
        for (int t = 0; t < task_.horizon; ++t)
            for (int j = 0; j < NUM_JOINTS; ++j) {
                const int idx = t * NUM_JOINTS + j;
                new_traj[idx] += w * actions_[s * stride + idx];
            }
    }
    for (int t = 0; t < task_.horizon; ++t)
        for (int j = 0; j < NUM_JOINTS; ++j) {
            const int idx = t * NUM_JOINTS + j;
            new_traj[idx] = std::clamp(new_traj[idx], action_lo_[j], action_hi_[j]);
        }
    trajectory_ = std::move(new_traj);

    // Cost breakdown logging — runs once per second (every 50 updates at 50 Hz).
    static constexpr int LOG_INTERVAL = 50;
    if (++log_counter_ % LOG_INTERVAL == 0) {
        mjData* dl = data_[task_.n_samples];
        set_mj_state(dl, state);

        double c_pos = 0, c_orient = 0, c_vel = 0, c_gait = 0, c_jvel = 0, c_effort = 0;

        for (int t = 0; t < task_.horizon; ++t) {
            double q_l[NUM_JOINTS], dq_l[NUM_JOINTS];
            for (int j = 0; j < NUM_JOINTS; ++j) {
                q_l[j]  = dl->qpos[act_qpos_adr_[j]];
                dq_l[j] = dl->qvel[act_qvel_adr_[j]];
            }
            double tau_l[NUM_JOINTS];
            for (int j = 0; j < NUM_JOINTS; ++j) {
                const double q_des = trajectory_[t * NUM_JOINTS + j];
                tau_l[j] = unitree_pd_torque(pd_.kp[j], pd_.kd[j], q_des, q_l[j],
                                             /*dq_des=*/0.0, dq_l[j], /*tau_ff=*/0.0);
            }
            for (int j = 0; j < model_->nu; ++j) dl->ctrl[j] = 0.0;
            for (int j = 0; j < NUM_JOINTS; ++j) dl->ctrl[j] = tau_l[j];
            mj_step(model_, dl);

            double lpos[3], lvel[3];
            base_state(dl, lpos, lvel);
            c_pos += cost_.pos_x * std::abs(lpos[0] - command().goal_pos[0]);
            c_pos += cost_.pos_y * std::abs(lpos[1] - command().goal_pos[1]);
            c_pos += cost_.pos_z * std::abs(lpos[2] - command().goal_pos[2]);

            const double q_dot_l  = dl->qpos[3]*goal_quat_[0] + dl->qpos[4]*goal_quat_[1]
                                   + dl->qpos[5]*goal_quat_[2] + dl->qpos[6]*goal_quat_[3];
            const double q_dist = 1.0 - std::abs(q_dot_l);
            c_orient += cost_.orientation * q_dist * q_dist;

            if (cost_.vel_x > 0.0 || cost_.vel_y > 0.0 || cost_.vel_z > 0.0) {
                c_vel += cost_.vel_x * (lvel[0] - command().vx)*(lvel[0] - command().vx)
                       + cost_.vel_y * (lvel[1] - command().vy)*(lvel[1] - command().vy)
                       + cost_.vel_z * lvel[2]*lvel[2];
            }

            if (cost_.ang_vel > 0.0) {
                const double wx = dl->qvel[3], wy = dl->qvel[4], wz = dl->qvel[5];
                c_vel += cost_.ang_vel * (wx*wx + wy*wy + wz*wz);
            }

            if (phases_.active_gait()) {
                double gref_q[NUM_JOINTS] = {}, gref_dq[NUM_JOINTS] = {};
                phases_.active_gait()->get_phase(t, gref_q, gref_dq);
                for (int j = 0; j < NUM_JOINTS; ++j) {
                    if (cost_.gait_ref_weights[j] == 0.0) continue;
                    const double e = q_l[j] - gref_q[j];
                    c_gait += cost_.gait_ref_weights[j] * e * e;
                }
                for (int j = 0; j < NUM_JOINTS; ++j) {
                    if (cost_.joint_vel_weights[j] == 0.0) continue;
                    const double e = dq_l[j] - gref_dq[j];
                    c_jvel += cost_.joint_vel_weights[j] * e * e;
                }
            }

            for (int j = 0; j < NUM_JOINTS; ++j) {
                if (cost_.control_effort_weights[j] == 0.0) continue;
                // Same cost-shaping gains + post-step state as step_cost().
                const double u_error = 50.0 * (trajectory_[t * NUM_JOINTS + j]
                                               - dl->qpos[act_qpos_adr_[j]])
                                     - 3.0 * dl->qvel[act_qvel_adr_[j]];
                c_effort += cost_.control_effort_weights[j] * u_error * u_error;
            }
        }

        std::printf("[cost] pos=%6.1f  orient=%6.1f  vel=%6.1f  gait=%6.1f  jvel=%6.1f  effort=%6.1f  | sample_min=%6.1f  dt=%.1fms\n",
                    c_pos, c_orient, c_vel, c_gait, c_jvel, c_effort, cmin, last_compute_ms_);
    }

    // Output: execute step 0 of the weighted-mean trajectory from current state.
    for (int j = 0; j < NUM_JOINTS; ++j) real_q_des_[j] = trajectory_[j];
    for (int j = 0; j < NUM_JOINTS; ++j)
        tau_out[j] = unitree_pd_torque(pd_.kp[j], pd_.kd[j], real_q_des_[j], state.q[j],
                                       /*dq_des=*/0.0, state.dq[j], /*tau_ff=*/0.0);

    if (phases_.active_gait()) phases_.active_gait()->advance();

    last_compute_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_start).count();
}
