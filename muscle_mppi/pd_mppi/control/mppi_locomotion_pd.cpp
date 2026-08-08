#include "mppi_locomotion_pd.h"

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
// original joint-space gait library — see pd_mppi/gaits/ and
// analysis/unit_tests/generate_pd_gaits.py) from the FAST/MED library in
// ../pd_mppi/gaits/. A phase selects a gait by name (TaskPhase::desired_gait)
// or, as an escape hatch, an explicit TSV path (TaskPhase::gait_path) — see
// resolve_gait_key() below. Same 4-gait mapping as the muscle variant
// (control/mppi_locomotion.cpp's kNamedGaits).
static const char* GAIT_INPLACE_PATH   = "../pd_mppi/gaits/FAST/gait_FAST_0_0_10cm.tsv";
static const char* GAIT_WALK_PATH      = "../pd_mppi/gaits/MED/gait_MED_0_1_10cm.tsv";
static const char* GAIT_WALK_FAST_PATH = "../pd_mppi/gaits/FAST/gait_FAST_0_1_10cm.tsv";
static const char* GAIT_TROT_PATH      = "../pd_mppi/gaits/MED/gait_MED_0_5_15cm.tsv";

static const std::unordered_map<std::string, const char*> kNamedGaits = {
    {"in_place",  GAIT_INPLACE_PATH},
    {"walk",      GAIT_WALK_PATH},
    {"walk_fast", GAIT_WALK_FAST_PATH},
    {"trot",      GAIT_TROT_PATH},
};

// Resolves a phase to the key it's loaded/stored under in MPPILocomotionPD::gaits_:
// an explicit gait_path override (if set) is keyed by its own path string;
// otherwise desired_gait must be one of kNamedGaits' keys.
static std::string resolve_gait_key(const TaskPhase& p)
{
    if (!p.gait_path.empty()) return p.gait_path;
    if (!kNamedGaits.count(p.desired_gait))
        throw std::runtime_error("Unknown desired_gait '" + p.desired_gait
                                 + "'. Must be one of: in_place, walk, walk_fast, trot");
    return p.desired_gait;
}

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

    // Load the 4 canonical named gaits up front, plus any per-phase gait_path
    // override not already covered.
    for (const auto& kv : kNamedGaits) gaits_[kv.first].load(kv.second);
    for (const auto& p : task_.phases)
        if (!p.gait_path.empty() && !gaits_.count(p.gait_path))
            gaits_[p.gait_path].load(p.gait_path);

    // Snapshot the YAML-loaded baseline before activate_phase() can overwrite
    // task_.noise_sigma_act with a per-phase override.
    std::memcpy(base_noise_sigma_act_, task_.noise_sigma_act, sizeof(base_noise_sigma_act_));

    if (!task_.phases.empty())
        activate_phase(0);

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
// Phase sequencing
// ============================================================================

void MPPILocomotionPD::activate_phase(int idx)
{
    const TaskPhase& p = task_.phases[idx];
    cmd_.goal_pos[0] = p.goal_pos[0];
    cmd_.goal_pos[1] = p.goal_pos[1];
    cmd_.goal_pos[2] = p.goal_pos[2];
    cmd_.vx = p.cmd_vel[0];
    cmd_.vy = p.cmd_vel[1];
    active_gait_ = &gaits_.at(resolve_gait_key(p));

    // Per-phase noise_sigma_act override, falling back to the task-level
    // baseline.
    if (p.has_noise_sigma_act)
        std::memcpy(task_.noise_sigma_act, p.noise_sigma_act, sizeof(task_.noise_sigma_act));
    else
        std::memcpy(task_.noise_sigma_act, base_noise_sigma_act_, sizeof(task_.noise_sigma_act));
}

void MPPILocomotionPD::advance_phase(const RobotState& state)
{
    if (task_.phases.empty()) return;

    const TaskPhase& cur = task_.phases[phase_index_];
    const double dx = state.pos[0] - cur.goal_pos[0];
    const double dy = state.pos[1] - cur.goal_pos[1];
    const double dz = state.pos[2] - cur.goal_pos[2];
    if (std::sqrt(dx*dx + dy*dy + dz*dz) >= cur.goal_thresh) return;  // distance gate; dwelling_ untouched

    // Dwell gate: counts ticks spent within goal_thresh, not reset when the
    // robot drifts back out in between. Kept running even after task_success_
    // so dwelling_ keeps tracking correctly.
    //
    // <= (not <): RTWholeBodyMPPI's Timer.increment() only flips `done` once
    // elapsed_time (pre-incremented) reaches end_time, and that `done` check
    // happens inside the SAME next_goal() call that performed the increment —
    // so it takes waiting_time+1 in-threshold calls to advance a phase, not
    // waiting_time. Advancing on `dwell_ticks_ < waiting_time` fires one tick
    // early on every phase transition. NOTE: the muscle variant still uses
    // `<`; the two must agree before a controlled comparison. No effect on
    // any task with waiting_time: 0 (i.e. every task except guinea_fowl).
    if (++dwell_ticks_ <= cur.waiting_time) {
        dwelling_ = true;   // mid-dwell: settled, not yet cleared to advance
        return;
    }

    if (phase_index_ + 1 < static_cast<int>(task_.phases.size())) {
        ++phase_index_;
        activate_phase(phase_index_);
        dwell_ticks_ = 0;
        dwelling_ = false;  // resumed traveling toward the new phase's goal
        const TaskPhase& next = task_.phases[phase_index_];
        std::printf("[phase] -> %d (goal=%.2f,%.2f,%.2f gait=%s)\n",
                    phase_index_, next.goal_pos[0], next.goal_pos[1], next.goal_pos[2],
                    next.gait_path.empty() ? next.desired_gait.c_str() : next.gait_path.c_str());
    } else if (!task_success_) {
        task_success_ = true;  // dwelling_ untouched here
        std::printf("[phase] task complete.\n");
    } else {
        dwelling_ = true;  // settled at the final goal, post-success
    }
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
        for (int j = 0; j < NUM_JOINTS; ++j) d->ctrl[JOINT_OFFSET + j] = tau_out[j];

        mj_step(model_, d);

        double gait_ref_q[NUM_JOINTS] = {}, gait_ref_dq[NUM_JOINTS] = {};
        if (active_gait_) active_gait_->get_phase(t, gait_ref_q, gait_ref_dq);
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

    const double* xmat = d->xmat + base_bid_ * 9;
    const double vx = d->qvel[0], vy = d->qvel[1], vz = d->qvel[2];
    vel_body[0] = vx*xmat[0] + vy*xmat[3] + vz*xmat[6];
    vel_body[1] = vx*xmat[1] + vy*xmat[4] + vz*xmat[7];
    vel_body[2] = vx*xmat[2] + vy*xmat[5] + vz*xmat[8];
}

double MPPILocomotionPD::step_cost(mjData* d, const double gait_ref_q[NUM_JOINTS],
                                   const double gait_ref_dq[NUM_JOINTS],
                                   const double q_des[NUM_JOINTS])
{
    const CostWeights& w = cost_;
    double cost = 0.0;

    double pos[3], vel_body[3];
    base_state(d, pos, vel_body);

    cost += w.pos_x * std::abs(pos[0] - cmd_.goal_pos[0]);
    cost += w.pos_y * std::abs(pos[1] - cmd_.goal_pos[1]);
    cost += w.pos_z * std::abs(pos[2] - cmd_.goal_pos[2]);

    const double q_dot  = d->qpos[3]*goal_quat_[0] + d->qpos[4]*goal_quat_[1]
                         + d->qpos[5]*goal_quat_[2] + d->qpos[6]*goal_quat_[3];
    const double q_dist = 1.0 - std::abs(q_dot);
    cost += w.orientation * q_dist * q_dist;

    if (w.vel_x > 0.0 || w.vel_y > 0.0 || w.vel_z > 0.0) {
        const double ex = vel_body[0] - cmd_.vx;
        const double ey = vel_body[1] - cmd_.vy;
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
    // the whole rollout batch below. Only active when far enough from the
    // goal and not settled at a waypoint; otherwise the target is identity
    // (upright, no yaw preference).
    {
        const double dx = cmd_.goal_pos[0] - state.pos[0];
        const double dy = cmd_.goal_pos[1] - state.pos[1];
        const double dz = cmd_.goal_pos[2] - state.pos[2];
        const double goal_delta = std::sqrt(dx*dx + dy*dy + dz*dz);

        if (goal_delta > 0.1 && !dwelling_) {
            const double yaw   = std::atan2(dy, dx);
            const double pitch = -std::atan2(dz, std::sqrt(dx*dx + dy*dy));
            const double z_axis[3] = {0.0, 0.0, 1.0};
            const double y_axis[3] = {0.0, 1.0, 0.0};
            double q_yaw[4], q_pitch[4];
            mju_axisAngle2Quat(q_yaw,   z_axis, yaw);
            mju_axisAngle2Quat(q_pitch, y_axis, pitch);
            mju_mulQuat(goal_quat_, q_yaw, q_pitch);  // matches scipy's yaw_quat * pitch_quat order
        } else {
            goal_quat_[0] = 1.0; goal_quat_[1] = 0.0; goal_quat_[2] = 0.0; goal_quat_[3] = 0.0;
        }
    }

    // Warm-start: shift trajectory_ forward by 1 step.
    const int stride = task_.horizon * NUM_JOINTS;
    std::vector<double> shifted(stride);
    for (int t = 0; t < task_.horizon - 1; ++t)
        for (int j = 0; j < NUM_JOINTS; ++j)
            shifted[t * NUM_JOINTS + j] = trajectory_[(t + 1) * NUM_JOINTS + j];
    for (int j = 0; j < NUM_JOINTS; ++j)
        shifted[(task_.horizon - 1) * NUM_JOINTS + j] =
            trajectory_[(task_.horizon - 1) * NUM_JOINTS + j];
    trajectory_ = shifted;

    sample_actions();

    #pragma omp parallel for schedule(dynamic)
    for (int s = 0; s < task_.n_samples; ++s)
        costs_[s] = rollout(s, state);

    // Softmin weights.
    double cmin   = *std::min_element(costs_.begin(), costs_.end());
    double cmax   = *std::max_element(costs_.begin(), costs_.end());
    double crange = cmax - cmin;  // cmin logged below as a diagnostic only

    std::vector<double> weights(task_.n_samples);
    double wsum = 0.0;
    for (int s = 0; s < task_.n_samples; ++s) {
        double s_hat = (crange > 1e-12) ? (costs_[s] - cmin) / crange : 0.0;
        weights[s]   = std::exp(-s_hat / task_.lambda);
        wsum        += weights[s];
    }

    // Weighted average update. actions_ is already clamped (built in
    // sample_actions()/sample_actions_cubic()), so no per-sample re-clamp
    // here — matches RTWholeBodyMPPI's perturb_action() clipping once and
    // reusing that same array for both rollout and this weighted average.
    std::vector<double> new_traj(stride, 0.0);
    for (int s = 0; s < task_.n_samples; ++s) {
        const double w = weights[s] / wsum;
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
            for (int j = 0; j < NUM_JOINTS; ++j) dl->ctrl[JOINT_OFFSET + j] = tau_l[j];
            mj_step(model_, dl);

            double lpos[3], lvel[3];
            base_state(dl, lpos, lvel);
            c_pos += cost_.pos_x * std::abs(lpos[0] - cmd_.goal_pos[0]);
            c_pos += cost_.pos_y * std::abs(lpos[1] - cmd_.goal_pos[1]);
            c_pos += cost_.pos_z * std::abs(lpos[2] - cmd_.goal_pos[2]);

            const double q_dot_l  = dl->qpos[3]*goal_quat_[0] + dl->qpos[4]*goal_quat_[1]
                                   + dl->qpos[5]*goal_quat_[2] + dl->qpos[6]*goal_quat_[3];
            const double q_dist = 1.0 - std::abs(q_dot_l);
            c_orient += cost_.orientation * q_dist * q_dist;

            if (cost_.vel_x > 0.0 || cost_.vel_y > 0.0 || cost_.vel_z > 0.0) {
                c_vel += cost_.vel_x * (lvel[0] - cmd_.vx)*(lvel[0] - cmd_.vx)
                       + cost_.vel_y * (lvel[1] - cmd_.vy)*(lvel[1] - cmd_.vy)
                       + cost_.vel_z * lvel[2]*lvel[2];
            }

            if (cost_.ang_vel > 0.0) {
                const double wx = dl->qvel[3], wy = dl->qvel[4], wz = dl->qvel[5];
                c_vel += cost_.ang_vel * (wx*wx + wy*wy + wz*wz);
            }

            if (active_gait_) {
                double gref_q[NUM_JOINTS] = {}, gref_dq[NUM_JOINTS] = {};
                active_gait_->get_phase(t, gref_q, gref_dq);
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

    if (active_gait_) active_gait_->advance();

    last_compute_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_start).count();
}
