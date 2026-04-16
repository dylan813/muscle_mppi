#pragma once

#include <cmath>
#include <algorithm>
#include "../utils/tasks.h"

// -----------------------------------------------------------------------
// Hill-type muscle model
// Based on SATA go2_torque.py _compute_torques() (fatigue removed)
//
// Per physics substep, for each joint j:
//
//  1. Activation dynamics (first-order low-pass, SATA tanh filter):
//       a[j] = tanh(act_cmd[j]) * alpha + a_prev[j] * (1 - alpha)
//       SATA: activation_sign = (tanh(cmd) - prev) * 0.6 + prev
//
//  2. Hill force-velocity curve:
//       vel_factor = clamp(1 - sign(a[j]) * dq[j] / dq_max[j],  0, 2)
//       tau[j]     = a[j] * tau_max[j] * vel_factor
//       - dq=0 (isometric):   tau = a * tau_max          (full force)
//       - shortening:         vel_factor < 1, force reduced
//       - lengthening:        vel_factor > 1, force boosted up to 2x
// -----------------------------------------------------------------------

// Persistent activation state per set of muscles
struct MuscleState {
    double activation[NUM_JOINTS] = {};  // filtered activation a[j] in [-1, 1]
};

// Compute Hill-model torques for one physics substep.
// Updates activation[] in-place.
// act_cmd[j] = commanded activation ∈ [-1, 1]
// dq[j]      = current joint velocity (rad/s)
inline void hill_compute_torques(
    const double        act_cmd[NUM_JOINTS],
    const double        dq[NUM_JOINTS],
    const MuscleParams& p,
    double              activation[NUM_JOINTS],  // in/out
    double              tau_out[NUM_JOINTS])      // out
{
    for (int j = 0; j < NUM_JOINTS; ++j) {
        // 1. Activation dynamics: first-order low-pass
        activation[j] = std::tanh(act_cmd[j]) * p.activation_alpha
                      + activation[j]          * (1.0 - p.activation_alpha);

        // 2. Hill force-velocity curve
        double sign_a     = std::copysign(1.0, activation[j]);
        double vel_factor = std::clamp(1.0 - sign_a * dq[j] / p.dq_max[j], 0.0, 2.0);
        tau_out[j]        = activation[j] * p.tau_max[j] * vel_factor;
    }
}
