#pragma once

// Gait-reference plumbing shared by both variants: reading and cycling through
// a 2*NUM_JOINTS × N gait TSV, and resolving a task phase to the gait it uses.
// What the rows mean is variant-specific, so each variant derives its own
// scheduler with a get_phase() for its layout (muscle/control/gait_scheduler.h,
// pd/control/gait_scheduler_pd.h).

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "paths.h"
#include "types.h"

// Thread-safety: reading the table (the derived get_phase()) is read-only and
// safe to call from parallel MPPI rollouts. advance() must only be called from
// the main thread after all rollouts complete.
class GaitTable {
public:
    // `name` prefixes load errors, so they say which scheduler failed.
    explicit GaitTable(const char* name) : name_(name) {}

    void load(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error(name_ + ": cannot open: " + path);

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
            throw std::runtime_error(name_ + ": expected "
                + std::to_string(2 * NUM_JOINTS) + " rows, got "
                + std::to_string(rows.size()));
        n_phases_ = (int)rows[0].size();
        for (auto& r : rows)
            if ((int)r.size() != n_phases_)
                throw std::runtime_error(name_ + ": inconsistent row lengths");

        gait_    = std::move(rows);
        phase_   = 0;
        loaded_  = true;
    }

    bool loaded()   const { return loaded_; }
    int  n_phases() const { return n_phases_; }

    // Advance one phase step. Call once per update() from the main thread.
    void advance() {
        if (loaded_) phase_ = (phase_ + 1) % n_phases_;
    }

protected:
    // Column index for `t_offset` steps ahead of the current phase.
    int    column(int t_offset)       const { return (phase_ + t_offset) % n_phases_; }
    double value(int row, int column) const { return gait_[row][column]; }

private:
    std::string name_;
    std::vector<std::vector<double>> gait_;
    int  n_phases_ = 0;
    int  phase_    = 0;
    bool loaded_   = false;
};

// Categorical gait name -> repo-relative TSV path (each variant has its own table).
using NamedGaitPaths = std::unordered_map<std::string, const char*>;

// Resolves a phase to the key its gait is loaded/stored under: an explicit
// gait_path override (if set) is keyed by its own path string; otherwise
// desired_gait must be one of `named`'s keys.
inline std::string resolve_gait_key(const TaskPhase& p, const NamedGaitPaths& named)
{
    if (!p.gait_path.empty()) return p.gait_path;
    if (!named.count(p.desired_gait))
        throw std::runtime_error("Unknown desired_gait '" + p.desired_gait
                                 + "'. Must be one of: in_place, walk, walk_fast, trot");
    return p.desired_gait;
}

// Load every named gait up front (mirrors RTWholeBodyMPPI's self.gaits dict),
// plus any per-phase gait_path override not already covered, keyed as
// resolve_gait_key() expects.
template <class Gait>
void load_gaits(std::unordered_map<std::string, Gait>& gaits, const NamedGaitPaths& named,
                const std::vector<TaskPhase>& phases)
{
    for (const auto& kv : named) gaits[kv.first].load(repo_path(kv.second));
    for (const auto& p : phases)
        if (!p.gait_path.empty() && !gaits.count(p.gait_path))
            gaits[p.gait_path].load(p.gait_path);
}
