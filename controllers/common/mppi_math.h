#pragma once

// MPPI update building blocks shared by both variants. They work on any
// action width (NUM_MUSCLES for the muscle variant, NUM_JOINTS for PD), with
// trajectories stored row-major as [t * action_dim + a].

#include <algorithm>
#include <cmath>
#include <vector>

// Warm-start: shift a (horizon × action_dim) trajectory forward by one step,
// holding the final step constant (mirrors RTWholeBodyMPPI).
inline void shift_trajectory(std::vector<double>& traj, int horizon, int action_dim)
{
    std::vector<double> shifted(horizon * action_dim);
    for (int t = 0; t < horizon - 1; ++t)
        for (int a = 0; a < action_dim; ++a)
            shifted[t * action_dim + a] = traj[(t + 1) * action_dim + a];
    for (int a = 0; a < action_dim; ++a)
        shifted[(horizon - 1) * action_dim + a] = traj[(horizon - 1) * action_dim + a];
    traj = std::move(shifted);
}

// Softmin sample weights over min-max normalised costs:
//   w_s = exp(-(c_s - c_min) / (c_max - c_min) / lambda),  normalised to sum to 1.
// All weights are equal when every cost is the same (range <= 1e-12).
// Fills weights (resized to costs.size()) and returns c_min, which callers
// log as a diagnostic.
inline double softmin_weights(const std::vector<double>& costs, double lambda,
                              std::vector<double>& weights)
{
    const int n = static_cast<int>(costs.size());
    const double cmin   = *std::min_element(costs.begin(), costs.end());
    const double cmax   = *std::max_element(costs.begin(), costs.end());
    const double crange = cmax - cmin;

    weights.resize(n);
    double wsum = 0.0;
    for (int s = 0; s < n; ++s) {
        const double s_hat = (crange > 1e-12) ? (costs[s] - cmin) / crange : 0.0;
        weights[s] = std::exp(-s_hat / lambda);
        wsum      += weights[s];
    }
    for (int s = 0; s < n; ++s) weights[s] /= wsum;

    return cmin;
}
