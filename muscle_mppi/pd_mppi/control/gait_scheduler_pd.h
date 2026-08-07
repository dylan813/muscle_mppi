#pragma once

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../utils/tasks_pd.h"

// Cycles through a 2*NUM_JOINTS×N joint-space gait TSV (rows 0..NUM_JOINTS-1
// = joint positions, rows NUM_JOINTS..2*NUM_JOINTS-1 = joint velocities —
// RTWholeBodyMPPI's original raw gait format, unlike the muscle variant's
// GaitScheduler (control/gait_scheduler.h), which stores an agonist/antagonist
// activation pair per joint instead). Used by MPPILocomotionPD's step_cost()
// for both position-space and velocity-space gait tracking (mirrors
// RTWholeBodyMPPI's Q_diag[7:19] and Q_diag[25:37] terms).
//
// Thread-safety: get_phase() is read-only and safe to call from parallel MPPI rollouts.
// advance() must only be called from the main thread after all rollouts complete.
class GaitSchedulerPD {
public:
    GaitSchedulerPD() = default;

    void load(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("GaitSchedulerPD: cannot open: " + path);

        std::vector<std::vector<double>> rows;
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::vector<double> row;
            std::istringstream ss(line);
            double v;
            while (ss >> v) row.push_back(v);
            if (!row.empty()) rows.push_back(std::move(row));
        }

        if ((int)rows.size() != 2 * NUM_JOINTS)
            throw std::runtime_error("GaitSchedulerPD: expected "
                + std::to_string(2 * NUM_JOINTS) + " rows, got "
                + std::to_string(rows.size()));
        n_phases_ = (int)rows[0].size();
        for (auto& r : rows)
            if ((int)r.size() != n_phases_)
                throw std::runtime_error("GaitSchedulerPD: inconsistent row lengths");

        gait_    = std::move(rows);
        phase_   = 0;
        loaded_  = true;
    }

    bool loaded()   const { return loaded_; }
    int  n_phases() const { return n_phases_; }

    // Fill q_ref[NUM_JOINTS]/dq_ref[NUM_JOINTS] with joint position/velocity
    // references at phase (phase_ + t_offset) % n_phases_.
    void get_phase(int t_offset, double q_ref[NUM_JOINTS], double dq_ref[NUM_JOINTS]) const {
        const int ph = (phase_ + t_offset) % n_phases_;
        for (int j = 0; j < NUM_JOINTS; ++j) {
            q_ref[j]  = gait_[j][ph];
            dq_ref[j] = gait_[NUM_JOINTS + j][ph];
        }
    }

    // Advance one phase step. Call once per update() from the main thread.
    void advance() {
        if (loaded_) phase_ = (phase_ + 1) % n_phases_;
    }

private:
    std::vector<std::vector<double>> gait_;
    int  n_phases_ = 0;
    int  phase_    = 0;
    bool loaded_   = false;
};
