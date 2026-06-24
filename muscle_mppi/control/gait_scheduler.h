#pragma once

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../utils/tasks.h"

// Cycles through a 24×N activation gait TSV (rows 0..11 = a1, rows 12..23 = a2).
//
// Thread-safety: get_phase() is read-only and safe to call from parallel MPPI rollouts.
// advance() must only be called from the main thread after all rollouts complete.
class GaitScheduler {
public:
    GaitScheduler() = default;

    void load(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("GaitScheduler: cannot open: " + path);

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
            throw std::runtime_error("GaitScheduler: expected "
                + std::to_string(2 * NUM_JOINTS) + " rows, got "
                + std::to_string(rows.size()));
        n_phases_ = (int)rows[0].size();
        for (auto& r : rows)
            if ((int)r.size() != n_phases_)
                throw std::runtime_error("GaitScheduler: inconsistent row lengths");

        gait_    = std::move(rows);
        phase_   = 0;
        loaded_  = true;
    }

    bool loaded()   const { return loaded_; }
    int  n_phases() const { return n_phases_; }

    // Fill act_ref[NUM_MUSCLES] with activations at phase (phase_ + t_offset) % n_phases_.
    // Interleaved layout: [a1_j0, a2_j0, a1_j1, a2_j1, ...] matching hill_compute_torques.
    void get_phase(int t_offset, double act_ref[NUM_MUSCLES]) const {
        const int ph = (phase_ + t_offset) % n_phases_;
        for (int j = 0; j < NUM_JOINTS; ++j) {
            act_ref[2 * j]     = gait_[j][ph];               // a1 agonist
            act_ref[2 * j + 1] = gait_[NUM_JOINTS + j][ph];  // a2 antagonist
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
