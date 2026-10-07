#pragma once

// First-order activation dynamics, shared by both variants: the muscle variant
// filters its activations with it (hill_compute_torques, muscle/control/muscle.h)
// and the PD variant its joint targets (MPPILocomotionPD, when the task has an
// `activation_dynamics:` block). Kept free of either variant's TaskConfig so
// both can include it.
//
// Each step, state moves alpha of the way toward cmd:
//   state += alpha * (cmd - state),   alpha = bandwidth * dt
// i.e. an exponential moving average of the commands, the forward-Euler form of
// d(state)/dt = bandwidth * (cmd - state) (time constant 1/bandwidth). alpha = 1
// makes state equal cmd; alpha > 1 overshoots it.
inline void activation_dynamics(const double* cmd, double* state, int n, double alpha)
{
    for (int i = 0; i < n; ++i)
        state[i] += alpha * (cmd[i] - state[i]);
}
