#pragma once

// Joint torques that carry the planned ground load: the bias torque (gravity +
// Coriolis, qfrc_bias) less what the ground's planned force on each foot
// (gait_gen/gaits/<label>_forces.tsv, gait_forces_path() in common/gait.h)
// does at the joints:
//
//     tau = qfrc_bias − Σ_feet J_footᵀ F_foot
//
// J_foot is the translational Jacobian of the foot's contact point, the bottom
// of its sphere geom: the point gait_gen/grf.py balances the body
// with. Forces are world frame. Swing feet have zero force and are skipped.
//
// Used for both sides of the gait-tracking cost: the activation gait
// (activation_gait.cpp, at the table's state) and each rollout's state
// (mppi_locomotion.cpp), so both invert the same planned load.

#include <stdexcept>
#include <string>

#include <mujoco/mujoco.h>

#include "../../common/log.h"           // FootGeoms, find_foot_geoms()
#include "../../common/task_config.h"   // NUM_JOINTS

// Values per row of a forces file (FR, FL, RR, RL × x, y, z), and per column.
constexpr int NUM_FOOT_FORCES = 12;

// find_foot_geoms(), but every foot must exist and be a sphere.
inline FootGeoms require_foot_geoms(const mjModel* m)
{
    const FootGeoms feet = find_foot_geoms(m);
    for (int i = 0; i < 4; ++i) {
        if (feet.gid[i] < 0)
            throw std::runtime_error("stance_load: model has no foot geom " + std::to_string(i));
        if (m->geom_type[feet.gid[i]] != mjGEOM_SPHERE)
            throw std::runtime_error("stance_load: foot geom " + std::to_string(i) + " is not a sphere");
    }
    return feet;
}

// tau[j] for each actuated joint j (qv: its dof address). d must hold
// positions and qfrc_bias for its state (mj_forward, or refresh_derived() in
// mppi_locomotion.cpp). The Jacobian lives on d's stack.
inline void loaded_joint_torques(const mjModel* m, mjData* d, const FootGeoms& feet,
                                 const int qv[NUM_JOINTS], const double forces[NUM_FOOT_FORCES],
                                 double tau[NUM_JOINTS])
{
    for (int j = 0; j < NUM_JOINTS; ++j) tau[j] = d->qfrc_bias[qv[j]];

    mj_markStack(d);
    mjtNum* jacp = mj_stackAllocNum(d, 3 * m->nv);   // 3 × nv, row-major

    for (int i = 0; i < 4; ++i) {
        const double* F = forces + 3 * i;
        if (F[0] == 0.0 && F[1] == 0.0 && F[2] == 0.0) continue;   // swing

        const int g = feet.gid[i];
        const mjtNum point[3] = {d->geom_xpos[3 * g],
                                 d->geom_xpos[3 * g + 1],
                                 d->geom_xpos[3 * g + 2] - m->geom_size[3 * g]};   // sphere bottom
        mj_jac(m, d, jacp, nullptr, point, m->geom_bodyid[g]);

        for (int j = 0; j < NUM_JOINTS; ++j)
            for (int k = 0; k < 3; ++k)
                tau[j] -= jacp[k * m->nv + qv[j]] * F[k];
    }

    mj_freeStack(d);
}
