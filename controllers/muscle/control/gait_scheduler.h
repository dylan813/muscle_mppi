#pragma once

#include "stance_load.h"   // NUM_FOOT_FORCES
#include "../utils/tasks.h"
#include "../../common/gait.h"

// Cycles through an activation gait TSV (activation_gait.h): rows 0..11 = a1,
// rows 12..23 = a2, rows 24..35 = the planned stance forces the activations
// were generated with (FR, FL, RR, RL × world x, y, z; stance_load.h).
// Loading, advancing and thread-safety rules come from GaitTable (common/gait.h).
class GaitScheduler : public GaitTable {
public:
    GaitScheduler() : GaitTable("GaitScheduler", 2 * NUM_JOINTS + NUM_FOOT_FORCES) {}

    // Fill act_ref[NUM_MUSCLES] with activations at phase (phase_ + t_offset) % n_phases_.
    // Interleaved layout: [a1_j0, a2_j0, a1_j1, a2_j1, ...] matching hill_compute_torques.
    void get_phase(int t_offset, double act_ref[NUM_MUSCLES]) const {
        const int ph = column(t_offset);
        for (int j = 0; j < NUM_JOINTS; ++j) {
            act_ref[2 * j]     = value(j, ph);               // a1 agonist
            act_ref[2 * j + 1] = value(NUM_JOINTS + j, ph);  // a2 antagonist
        }
    }

    // Fill forces[NUM_FOOT_FORCES] with the planned stance forces at the same phase.
    void get_forces(int t_offset, double forces[NUM_FOOT_FORCES]) const {
        const int ph = column(t_offset);
        for (int k = 0; k < NUM_FOOT_FORCES; ++k)
            forces[k] = value(2 * NUM_JOINTS + k, ph);
    }
};
