#pragma once

#include <cmath>
#include <algorithm>
#include "../utils/tasks.h"

// Antagonistic Hill muscle model matching learningwithmuscles/mjmuscle_geometry_free.pyx.
// Each joint j is driven by two virtual muscles: agonist (2*j) and antagonist (2*j+1).
// Activations are in [0, 1]. Net torque emerges from the imbalance between the pair.

struct MuscleState {
    double activation[NUM_MUSCLES] = {};
};

// MuJoCo's piecewise-quadratic force-length bump.
// Returns a value in [0,1] peaked at 1.0 when length==mid, zero outside [A,B].
static inline double active_force_length(double length, double A, double mid, double B) {
    const double left  = 0.5 * (A + mid);
    const double right = 0.5 * (mid + B);
    if (length <= A || length >= B) return 0.0;
    double temp;
    if (length < left) {
        temp = (length - A) / (left - A);
        return 0.5 * temp * temp;
    } else if (length < mid) {
        temp = (mid - length) / (mid - left);
        return 1.0 - 0.5 * temp * temp;
    } else if (length < right) {
        temp = (length - mid) / (right - mid);
        return 1.0 - 0.5 * temp * temp;
    } else {
        temp = (B - length) / (B - right);
        return 0.5 * temp * temp;
    }
}

// MuJoCo's force-velocity curve.
// Concentric (shortening, vel<0): quadratic rise 0→1.
// Eccentric (lengthening, vel>0): quadratic rise 1→fvmax.
static inline double force_vel(double velocity, double c, double vmax, double fvmax) {
    const double eff_vel = velocity / vmax;
    if (eff_vel < -1.0) return 0.0;
    if (eff_vel <= 0.0) return (eff_vel + 1.0) * (eff_vel + 1.0);
    if (eff_vel <= c)   return fvmax - (c - eff_vel) * (c - eff_vel) / c;
    return fvmax;
}

// MuJoCo's parallel elastic (passive) force.
// Zero below optimal length, cubic then linear above.
static inline double passive_force_length(double length, double fpmax, double b) {
    if (length <= 1.0) return 0.0;
    double temp;
    if (length <= b) {
        temp = (length - 1.0) / (b - 1.0);
        return 0.25 * fpmax * temp * temp * temp;
    }
    temp = (length - b) / (b - 1.0);
    return 0.25 * fpmax * (1.0 + 3.0 * temp);
}

// Compute joint torques from antagonistic muscle pairs.
//
// Action layout (act_cmd, activation): interleaved per joint:
//   [agonist_j0, antagonist_j0, agonist_j1, antagonist_j1, ...]
//
// Activation dynamics: first-order filter at act_bandwidth Hz.
// dt must be the physics timestep (task_.dt), not the control period.
inline void hill_compute_torques(
    const double        act_cmd[NUM_MUSCLES],
    const double        q[NUM_JOINTS],
    const double        dq[NUM_JOINTS],
    const MuscleParams& p,
    double              dt,
    double              activation[NUM_MUSCLES],   // in/out
    double              tau_out[NUM_JOINTS])
{
    const double alpha = p.act_bandwidth * dt;

    for (int j = 0; j < NUM_JOINTS; ++j) {
        // Linear moment-arm parametrization: maps [phi_min, phi_max] → [lce_min, lce_max].
        // Agonist (muscle 1) lengthens as q increases; antagonist (muscle 2) shortens.
        static constexpr double eps = 1e-6;
        const double r1 = (p.lce_max[j] - p.lce_min[j] + eps)
                        / (p.phi_max[j]  - p.phi_min[j]  + eps);   // > 0
        const double r2 = (p.lce_max[j] - p.lce_min[j] + eps)
                        / (p.phi_min[j]  - p.phi_max[j]  + eps);   // < 0  (= -r1)

        const double lce1 = q[j] * r1 + (p.lce_min[j] - r1 * p.phi_min[j]);
        const double lce2 = q[j] * r2 + (p.lce_min[j] - r2 * p.phi_max[j]);

        const double lce_dot1 = r1 * dq[j];
        const double lce_dot2 = r2 * dq[j];

        // First-order activation filter (matches: act_new = bandwidth*(ctrl-act)*dt + act).
        for (int m = 0; m < 2; ++m) {
            const double ctrl = std::clamp(act_cmd[2 * j + m], 0.0, 1.0);
            double& act = activation[2 * j + m];
            act = std::clamp(act + alpha * (ctrl - act), 0.0, 1.0);
        }
        const double act1 = activation[2 * j];
        const double act2 = activation[2 * j + 1];

        // Force-length: primary bell + secondary shoulder at short lengths.
        const double lmin = p.lce_min[j];
        const double lmax = p.lce_max[j];
        const double active_FL1 = active_force_length(lce1, lmin, 1.0, lmax)
                                + 0.15 * active_force_length(lce1, lmin, 0.5 * (lmin + 0.95), 0.95);
        const double active_FL2 = active_force_length(lce2, lmin, 1.0, lmax)
                                + 0.15 * active_force_length(lce2, lmin, 0.5 * (lmin + 0.95), 0.95);

        // Force-velocity.
        const double vmax  = p.vmax[j];
        const double fvmax = p.fvmax[j];
        const double c     = fvmax - 1.0;
        const double FV1   = force_vel(lce_dot1, c, vmax, fvmax);
        const double FV2   = force_vel(lce_dot2, c, vmax, fvmax);

        // Passive parallel elasticity.
        const double b_passive = 0.5 * (lmax + 1.0);
        const double passive_FL1 = passive_force_length(lce1, p.fpmax[j], b_passive);
        const double passive_FL2 = passive_force_length(lce2, p.fpmax[j], b_passive);

        // Total force per muscle (normalized), scaled to Newtons by peak_force.
        const double F1 = (active_FL1 * FV1 * act1 + passive_FL1) * p.peak_force[j];
        const double F2 = (active_FL2 * FV2 * act2 + passive_FL2) * p.peak_force[j];

        // Net joint torque: minus because muscles pull (matching MuJoCo convention).
        tau_out[j] = -(F1 * r1 + F2 * r2);
    }
}
