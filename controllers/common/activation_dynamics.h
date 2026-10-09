#pragma once

#include <cmath>

// First-order activation dynamics, shared by both variants: the muscle variant
// filters its activations with it (hill_compute_torques, muscle/control/muscle.h)
// and the PD variant its joint targets (MPPILocomotionPD, when the task has an
// `activation_dynamics:` block). Kept free of either variant's TaskConfig so
// both can include it.
//
// Each step, state moves alpha of the way toward cmd:
//   state += alpha * (cmd - state)
// i.e. an exponential moving average of the commands. With alpha from
// filter_alpha() this is d(state)/dt = bandwidth * (cmd - state) (time constant
// 1/bandwidth) integrated exactly for a command held over the step.
inline void activation_dynamics(const double* cmd, double* state, int n, double alpha)
{
    for (int i = 0; i < n; ++i)
        state[i] += alpha * (cmd[i] - state[i]);
}

// alpha for a first-order filter of rate `bandwidth` (1/s) over a step of dt:
// the exact discretization 1 − e^(−bandwidth·dt), so the step response matches
// the continuous filter at every sample. INFINITY gives exactly 1 (no lag).
inline double filter_alpha(double bandwidth, double dt)
{
    return std::isinf(bandwidth) ? 1.0 : 1.0 - std::exp(-bandwidth * dt);
}
