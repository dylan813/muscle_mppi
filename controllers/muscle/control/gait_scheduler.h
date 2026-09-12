#pragma once

#include "../utils/tasks.h"
#include "../../common/gait.h"

// Cycles through a 24×N activation gait TSV (rows 0..11 = a1, rows 12..23 = a2).
// Loading, advancing and thread-safety rules come from GaitTable (common/gait.h).
class GaitScheduler : public GaitTable {
public:
    GaitScheduler() : GaitTable("GaitScheduler") {}

    // Fill act_ref[NUM_MUSCLES] with activations at phase (phase_ + t_offset) % n_phases_.
    // Interleaved layout: [a1_j0, a2_j0, a1_j1, a2_j1, ...] matching hill_compute_torques.
    void get_phase(int t_offset, double act_ref[NUM_MUSCLES]) const {
        const int ph = column(t_offset);
        for (int j = 0; j < NUM_JOINTS; ++j) {
            act_ref[2 * j]     = value(j, ph);               // a1 agonist
            act_ref[2 * j + 1] = value(NUM_JOINTS + j, ph);  // a2 antagonist
        }
    }
};
