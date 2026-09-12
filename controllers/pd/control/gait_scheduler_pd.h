#pragma once

#include "../utils/tasks_pd.h"
#include "../../common/gait_table.h"

// Cycles through a 2*NUM_JOINTS×N joint-space gait TSV (rows 0..NUM_JOINTS-1
// = joint positions, rows NUM_JOINTS..2*NUM_JOINTS-1 = joint velocities —
// RTWholeBodyMPPI's original raw gait format, unlike the muscle variant's
// GaitScheduler (muscle/control/gait_scheduler.h), which stores an agonist/antagonist
// activation pair per joint instead). Used by MPPILocomotionPD's step_cost()
// for both position-space and velocity-space gait tracking (mirrors
// RTWholeBodyMPPI's Q_diag[7:19] and Q_diag[25:37] terms).
//
// Loading, advancing and thread-safety rules come from GaitTable (common/gait_table.h).
class GaitSchedulerPD : public GaitTable {
public:
    GaitSchedulerPD() : GaitTable("GaitSchedulerPD") {}

    // Fill q_ref[NUM_JOINTS]/dq_ref[NUM_JOINTS] with joint position/velocity
    // references at phase (phase_ + t_offset) % n_phases_.
    void get_phase(int t_offset, double q_ref[NUM_JOINTS], double dq_ref[NUM_JOINTS]) const {
        const int ph = column(t_offset);
        for (int j = 0; j < NUM_JOINTS; ++j) {
            q_ref[j]  = value(j, ph);
            dq_ref[j] = value(NUM_JOINTS + j, ph);
        }
    }
};
