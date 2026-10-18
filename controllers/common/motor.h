#pragma once

// External motor between every controller variant and the robot: the actuator
// torque-bandwidth limit all of them drive through (MotorParams, task_config.h).
//
// Per joint, the controller supplies its torque without velocity feedback
// (τ_other) and that feedback's gain kd (PD's kd, the muscle's driver kd). The
// motor commands τ_cmd = clamp(τ_other − kd·q̇), the driver's torque limit, and
// applies it through one of two lags (MotorLag):
//
// FirstOrder, dτ/dt = b·(τ_cmd − τ), with b = ∞ giving τ = τ_cmd. Per physics
// step, α = filter_alpha(b, Δt):
//
//   Explicit  τ ← (1−α)·τ + α·clamp(τ_other − kd·q̇ₖ),  ctrl = τ
//   Implicit  ctrl = (1−α)·τ + α·(clamp(τ_other − kd·q̇ₖ) + kd·q̇ₖ), with α·kd
//             added to the joint damping MuJoCo integrates implicitly
//             (implicit_damping()), so the torque applied is ctrl − α·kd·q̇ₖ₊₁.
//             Unsaturated, that is the filter fed τ_other − kd·q̇ₖ₊₁; saturated,
//             the + kd·q̇ₖ offset keeps it at the limit to first order.
//             Afterwards τ ← that applied torque (applied()).
//
// ZeroOrderHold, every hold_steps physics steps (1/update_rate s) the motor
// takes clamp(τ_other − kd·q̇ₖ) and applies it unchanged until the next one:
//
//   ctrl = clamp(τ_other − kd·q̇ₖ) on the step a command is taken, τ otherwise;
//   afterwards τ ← ctrl.
//
// The velocity feedback is held with the rest of the command, so damping is
// explicit. hold_steps = 1 takes a command every step: no lag, the same as
// FirstOrder at b = ∞.
//
// ctrl can leave the torque limits under Implicit, so the model's own
// ctrlrange clamp must be off while the motor drives it.

#include <algorithm>

#include "activation_dynamics.h"
#include "task_config.h"

// What the motor carries from one physics step to the next.
struct MotorState {
    double tau[NUM_JOINTS] = {};   // torque applied so far; the held torque under ZeroOrderHold
    int    step = 0;               // ZeroOrderHold: physics steps since the held command was taken
};

class Motor {
public:
    // kd: velocity-feedback gain per joint. lo, hi: torque limits per joint.
    void init(const MotorParams& p, const double kd[NUM_JOINTS],
              const double lo[NUM_JOINTS], const double hi[NUM_JOINTS])
    {
        lag_        = p.lag;
        alpha_      = p.lag == MotorLag::FirstOrder ? filter_alpha(p.bandwidth, p.physics_dt) : 1.0;
        hold_steps_ = p.lag == MotorLag::ZeroOrderHold ? p.hold_steps : 1;
        damping_    = p.damping;
        std::copy(kd, kd + NUM_JOINTS, kd_);
        std::copy(lo, lo + NUM_JOINTS, lo_);
        std::copy(hi, hi + NUM_JOINTS, hi_);
    }

    // Joint damping MuJoCo must integrate on top of the physical damping.
    double implicit_damping(int j) const
    {
        return damping_ == MotorDamping::Implicit ? alpha_ * kd_[j] : 0.0;
    }

    // ctrl for one physics step, from the joint velocity dq at its start and
    // the motor's state s. tau_cmd: the command before the torque limit (for
    // logging).
    void command(const double tau_other[NUM_JOINTS], const double dq[NUM_JOINTS],
                 const MotorState& s, double ctrl[NUM_JOINTS], double tau_cmd[NUM_JOINTS]) const
    {
        for (int j = 0; j < NUM_JOINTS; ++j) {
            tau_cmd[j] = tau_other[j] - kd_[j] * dq[j];
            const double limited = std::clamp(tau_cmd[j], lo_[j], hi_[j]);

            if (lag_ == MotorLag::ZeroOrderHold) {
                ctrl[j] = s.step == 0 ? limited : s.tau[j];
                continue;
            }

            const double input = damping_ == MotorDamping::Implicit ? limited + kd_[j] * dq[j] : limited;
            ctrl[j] = (1.0 - alpha_) * s.tau[j] + alpha_ * input;
        }
    }

    // s ← state after the step, from its ctrl and the joint velocity dq_after
    // at its end.
    void applied(const double ctrl[NUM_JOINTS], const double dq_after[NUM_JOINTS], MotorState& s) const
    {
        for (int j = 0; j < NUM_JOINTS; ++j)
            s.tau[j] = ctrl[j] - implicit_damping(j) * dq_after[j];
        s.step = (s.step + 1) % hold_steps_;
    }

    // s ← applying t clamped to the torque limits (e.g. a settled holding
    // torque), with a new command due on the next step.
    void settle(const double t[NUM_JOINTS], MotorState& s) const
    {
        for (int j = 0; j < NUM_JOINTS; ++j) s.tau[j] = std::clamp(t[j], lo_[j], hi_[j]);
        s.step = 0;
    }

private:
    MotorLag     lag_        = MotorLag::FirstOrder;
    double       alpha_      = 1.0;
    int          hold_steps_ = 1;
    MotorDamping damping_    = MotorDamping::Explicit;
    double       kd_[NUM_JOINTS] = {};
    double       lo_[NUM_JOINTS] = {};
    double       hi_[NUM_JOINTS] = {};
};
