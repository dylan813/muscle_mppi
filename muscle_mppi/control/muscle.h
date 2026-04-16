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
//       vel_factor = min(1 - sign(a[j]) * dq[j] / dq_max[j],  2)   ← no lower clamp
//       tau[j]     = a[j] * tau_max[j] * vel_factor
//       - dq=0 (isometric):   tau = a * tau_max          (full force)
//       - shortening:         vel_factor < 1, force reduced
//       - beyond dq_max:      vel_factor < 0, braking torque  ← velocity boundary constraint
//       - lengthening:        vel_factor > 1, force boosted up to 2x
//
//  3. Final output clamped to ±tau_max (matches Isaac Lab effort_limit behaviour).
//     This is the torque limit boundary constraint, equivalent to hardware motor limits.
// -----------------------------------------------------------------------

// Persistent activation state per set of muscles
struct MuscleState {
    double activation[NUM_JOINTS] = {};  // filtered activation a[j] in [-1, 1]
};

// Compute Hill-model torques for one physics substep.
// Updates activation[] in-place.
//
// act_cmd[j] = commanded activation ∈ [-1, 1]
// q[j]       = current joint position (rad)
// dq[j]      = current joint velocity (rad/s)
//
// Full torque per joint:
//   tau = tau_active + tau_passive + tau_damp
//
// tau_active  = a * tau_max * phi(dq)        (Hill force-velocity)
// tau_passive = exponential unilateral springs near joint limits (Zajac 1989)
// tau_damp    = -b * dq                      (viscous damping)
inline void hill_compute_torques(
    const double        act_cmd[NUM_JOINTS],
    const double        q[NUM_JOINTS],           // joint positions
    const double        dq[NUM_JOINTS],
    const MuscleParams& p,
    double              activation[NUM_JOINTS],  // in/out
    double              tau_out[NUM_JOINTS])      // out
{
    for (int j = 0; j < NUM_JOINTS; ++j) {
        // 1. Activation dynamics: first-order low-pass
        activation[j] = std::tanh(act_cmd[j]) * p.activation_alpha
                      + activation[j]          * (1.0 - p.activation_alpha);

        // 2. Hill force-velocity curve (active torque)
        //    No lower clamp: vel_factor < 0 when dq exceeds dq_max in the muscle's
        //    shortening direction, producing an automatic braking torque (boundary
        //    constraint, matches SATA's unclamped velocity term + Isaac Lab effort_limit).
        double sign_a     = std::copysign(1.0, activation[j]);
        double vel_factor = std::min(1.0 - sign_a * dq[j] / p.dq_max[j], 2.0);
        double tau_active = activation[j] * p.tau_max[j] * vel_factor;

        // 3. Passive elastic (unilateral exponential springs)
        //    Resists motion past q_plus0 (upper) and below q_minus0 (lower).
        double tau_passive = 0.0;
        double d_plus = q[j] - p.q_plus0[j];
        if (d_plus > 0.0)
            tau_passive -= p.k_plus[j]  * (std::exp(p.alpha_plus[j]  * d_plus)  - 1.0);
        double d_minus = p.q_minus0[j] - q[j];
        if (d_minus > 0.0)
            tau_passive += p.k_minus[j] * (std::exp(p.alpha_minus[j] * d_minus) - 1.0);

        // 4. Viscous damping
        double tau_damp = -p.b_damp[j] * dq[j];

        // 5. Torque limit: clamp to ±tau_max (hardware effort limit)
        tau_out[j] = std::clamp(tau_active + tau_passive + tau_damp,
                                -p.tau_max[j], p.tau_max[j]);
    }
}
