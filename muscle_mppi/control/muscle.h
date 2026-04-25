#pragma once

#include <cmath>
#include <algorithm>
#include "../utils/tasks.h"

// Hill-type muscle model: activation dynamics + force-velocity + passive springs.

struct MuscleState {
    double activation[NUM_JOINTS] = {};
};

inline void hill_compute_torques(
    const double        act_cmd[NUM_JOINTS],
    const double        q[NUM_JOINTS],           // joint positions
    const double        dq[NUM_JOINTS],
    const MuscleParams& p,
    double              activation[NUM_JOINTS],  // in/out
    double              tau_out[NUM_JOINTS])      // out
{
    for (int j = 0; j < NUM_JOINTS; ++j) {
        // activation dynamics: first-order low-pass
        activation[j] = std::tanh(act_cmd[j]) * p.activation_alpha
                      + activation[j]          * (1.0 - p.activation_alpha);

        // Hill force-velocity: no lower clamp so vel_factor < 0 produces braking torque
        double sign_a     = std::copysign(1.0, activation[j]);
        double vel_factor = std::min(1.0 - sign_a * dq[j] / p.dq_max[j], 2.0);
        double tau_active = activation[j] * p.tau_max[j] * vel_factor;

        // passive exponential springs near joint limits
        double tau_passive = 0.0;
        double d_plus = q[j] - p.q_plus0[j];
        if (d_plus > 0.0)
            tau_passive -= p.k_plus[j]  * (std::exp(p.alpha_plus[j]  * d_plus)  - 1.0);
        double d_minus = p.q_minus0[j] - q[j];
        if (d_minus > 0.0)
            tau_passive += p.k_minus[j] * (std::exp(p.alpha_minus[j] * d_minus) - 1.0);

        double tau_damp = -p.b_damp[j] * dq[j];

        tau_out[j] = std::clamp(tau_active + tau_passive + tau_damp,
                                -p.tau_max[j], p.tau_max[j]);
    }
}
