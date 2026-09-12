#pragma once

// Small control/kinematics helpers shared by the muscle-actuated and
// PD-actuated variants.

// Unitree's PD+feedforward-torque law, ported verbatim from unitree_mujoco's
// Go2Bridge::run() (unitree_mujoco/simulate/src/unitree_sdk2_bridge.h:183-185)
// — the same computation the real Go2's onboard motor firmware performs in
// mode 0x01, and what mppi_controller.cpp already drives via DDS LowCmd_
// q/kp/dq/kd/tau fields. Callers pass dq_des=0, tau_ff=0 unless they have a
// specific feedforward term, matching how this repo's own DDS controller
// already uses it (mppi_controller.cpp sets dq()=0.0) — not a simplification
// specific to this port.
inline double unitree_pd_torque(double kp, double kd, double q_des, double q,
                                double dq_des, double dq, double tau_ff)
{
    return tau_ff + kp * (q_des - q) + kd * (dq_des - dq);
}

// Rotate a world-frame vector into body-frame axes. xmat is the body->world
// rotation (row-major 3x3, as in mjData::xmat or mju_quat2Mat), so its
// transpose maps world->body. v_world and v_body must not alias.
inline void world_to_body(const double xmat[9], const double v_world[3], double v_body[3])
{
    v_body[0] = v_world[0]*xmat[0] + v_world[1]*xmat[3] + v_world[2]*xmat[6];
    v_body[1] = v_world[0]*xmat[1] + v_world[1]*xmat[4] + v_world[2]*xmat[7];
    v_body[2] = v_world[0]*xmat[2] + v_world[1]*xmat[5] + v_world[2]*xmat[8];
}
