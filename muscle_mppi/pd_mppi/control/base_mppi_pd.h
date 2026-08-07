#pragma once

#include <mujoco/mujoco.h>
#include <vector>
#include <random>
#include <string>
#include <algorithm>
#include "../utils/tasks_pd.h"

// Unitree's PD+feedforward-torque law, ported verbatim from unitree_mujoco's
// Go2Bridge::run() (unitree_mujoco/simulate/src/unitree_sdk2_bridge.h:183-185)
// — the same computation the real Go2's onboard motor firmware performs in
// mode 0x01, and what mppi_controller.cpp/single_leg_controller.cpp already
// drive via DDS LowCmd_ q/kp/dq/kd/tau fields. Callers pass dq_des=0,
// tau_ff=0 unless they have a specific feedforward term, matching how this
// repo's own DDS controllers already use it (e.g. single_leg_controller.cpp
// sets dq()=0.0) — not a simplification specific to this port.
inline double unitree_pd_torque(double kp, double kd, double q_des, double q,
                                double dq_des, double dq, double tau_ff)
{
    return tau_ff + kp * (q_des - q) + kd * (dq_des - dq);
}

struct RobotState {
    double pos[3]          = {};
    double vel[3]          = {};
    double quat[4]         = {1,0,0,0};  // w, x, y, z
    double gyro[3]         = {};
    double q[NUM_JOINTS]   = {};
    double dq[NUM_JOINTS]  = {};
    bool   valid           = false;
};

// PD-actuated variant of BaseMPPI: action space is one desired joint position
// per joint (NUM_JOINTS-wide), not an interleaved antagonistic muscle pair
// (NUM_MUSCLES-wide). See control/base_mppi.h for the muscle-actuated original.
class BaseMPPIPD {
public:
    explicit BaseMPPIPD(const TaskConfig& task);
    virtual ~BaseMPPIPD();

    double cost_min()  const { return *std::min_element(costs_.begin(), costs_.end()); }
    double cost_mean() const {
        double s = 0.0;
        for (auto c : costs_) s += c;
        return s / static_cast<double>(costs_.size());
    }

protected:
    virtual double rollout(int s, const RobotState& state) = 0;

    // Fills actions_[s][t][j] with this iteration's per-sample, per-timestep
    // desired joint positions — already clamped to [action_lo_, action_hi_],
    // ready to use directly for both rollout and the weighted-average update
    // (mirrors RTWholeBodyMPPI's perturb_action(), which returns this same
    // "already resolved" actions array rather than a separate noise buffer —
    // see sample_actions_cubic()'s comment for why this distinction matters).
    void sample_actions();
    void sample_actions_cubic();
    void set_mj_state(mjData* d, const RobotState& state);

    // Shift trajectory_ forward by n_skip steps, holding the tail constant.
    void warm_start(int n_skip);

    // Single MPPI pass: sample → parallel rollouts → softmin weighted-average update.
    // Subclasses set action_lo_/action_hi_ in their constructor to define per-action clamping.
    void run_mppi_step(const RobotState& state);

    TaskConfig task_;

    mjModel*             model_ = nullptr;
    std::vector<mjData*> data_;        // [n_samples + 1]: rollout slots + prediction

    std::vector<double> trajectory_;
    std::vector<double> actions_;
    std::vector<double> costs_;

    // Knot timesteps (0..horizon-1) used by sample_actions_cubic() when
    // task_.sample_type == "cubic". Built once in the constructor.
    std::vector<double> knot_x_;

    // Per-joint clamp bounds used by run_mppi_step(). Set from model_->jnt_range
    // in the constructor (see base_mppi_pd.cpp) — desired joint positions must
    // stay within the robot's actual joint limits.
    double action_lo_[NUM_JOINTS] = {};
    double action_hi_[NUM_JOINTS] = {};

    // Actuator → MuJoCo DOF addresses (built from JOINT_OFFSET — no hardcoded mapping)
    int  act_qpos_adr_[NUM_JOINTS] = {};
    int  act_qvel_adr_[NUM_JOINTS] = {};
    bool has_freejoint_ = false;

    std::mt19937 rng_;
    std::normal_distribution<double> normal_{0.0, 1.0};
};
