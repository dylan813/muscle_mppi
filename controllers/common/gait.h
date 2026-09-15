#pragma once

// Gait-reference plumbing shared by both variants: reading and cycling through
// a 2*NUM_JOINTS × N gait TSV, resolving a task phase to the gait it uses, and
// sequencing through a task's phases (PhaseSequencer). What the gait rows mean
// is variant-specific, so each variant derives its own scheduler with a
// get_phase() for its layout (muscle/control/gait_scheduler.h,
// pd/control/gait_scheduler_pd.h).
//
// Shared on purpose: phase sequencing (goal thresholds, dwell timing) is identical for the muscle and PD variants so their
// results stay comparable, and editing it changes both. If one variant needs
// different behavior, fork that piece into the variant's own folder first
// rather than changing it here (e.g. settle_standing() is muscle-only and PD
// never calls it; nominal_pose lives in PD's task config).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "task_config.h"

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

// Categorical gait name -> TSV path, repo-relative or absolute (each variant has
// its own table).
using NamedGaitPaths = std::unordered_map<std::string, std::string>;

// Resolves a phase to the key its gait is loaded/stored under: an explicit
// gait_path override (if set) is keyed by its own path string; otherwise
// desired_gait must be one of `named`'s keys.
inline std::string resolve_gait_key(const TaskPhase& p, const NamedGaitPaths& named)
{
    if (!p.gait_path.empty()) return p.gait_path;
    if (!named.count(p.desired_gait)) {
        // List the variant's own gait names (sorted, so the message is stable).
        std::vector<std::string> names;
        for (const auto& kv : named) names.push_back(kv.first);
        std::sort(names.begin(), names.end());
        std::string list;
        for (const auto& n : names) list += (list.empty() ? "" : ", ") + n;
        throw std::runtime_error("Unknown desired_gait '" + p.desired_gait
                                 + "'. Must be one of: " + list);
    }
    return p.desired_gait;
}

// Load every gait the task's phases use, once each, keyed as resolve_gait_key()
// expects: named gaits from `named`, per-phase gait_path overrides from their own
// path. Named gaits no phase uses aren't loaded. Resolving every phase here also
// reports an unknown desired_gait at startup rather than when that phase is reached.
template <class Gait>
void load_gaits(std::unordered_map<std::string, Gait>& gaits, const NamedGaitPaths& named,
                const std::vector<TaskPhase>& phases)
{
    for (const auto& p : phases) {
        const std::string key = resolve_gait_key(p, named);
        if (gaits.count(key)) continue;
        gaits[key].load(p.gait_path.empty() ? repo_path(named.at(key)) : p.gait_path);
    }
}

// ============================================================================
// Phase sequencing
// ============================================================================

// Walks a task's ordered phase (waypoint) sequence, matching RTWholeBodyMPPI's
// next_goal(). Owns the task's gaits and exposes, for the controller's cost:
// the current phase and MotionCommand, the active gait, and whether the robot
// is dwelling at a waypoint. Gait is the variant's scheduler (GaitScheduler /
// GaitSchedulerPD). Anything that depends on the variant's action space (e.g.
// the per-phase noise_sigma_act override) is applied by the controller, using
// current_phase() whenever init()/advance() report a new phase.
//
// Thread-safety: the accessors are read-only and safe from parallel rollouts;
// advance() and set_command() must only be called from the main thread.
template <class Gait>
class PhaseSequencer {
public:
    PhaseSequencer() = default;
    // Holds pointers into its own gait map and the controller's task, so a
    // copy would point at the wrong data.
    PhaseSequencer(const PhaseSequencer&)            = delete;
    PhaseSequencer& operator=(const PhaseSequencer&) = delete;

    // Load the gaits and activate phase 0. phases must outlive the sequencer.
    // A task with no phases keeps an all-zero command.
    void init(const std::vector<TaskPhase>& phases, const NamedGaitPaths& named)
    {
        phases_ = &phases;
        named_  = &named;

        // Load every gait the phases use (named gaits and gait_path overrides).
        load_gaits(gaits_, named, phases);

        if (!phases.empty())
            activate(0);
    }

    // Call once per control tick, before the controller's update(). Advances to
    // the next task phase once the robot has been within the current phase's
    // goal_thresh for more than waiting_time ticks (waiting_time + 1 in-threshold
    // ticks; cumulative, not reset if it drifts back out in between — see the
    // dwell gate below). Keeps running (updating dwelling) even after the
    // task's final phase is reached — matches RTWholeBodyMPPI's next_goal(),
    // which the driver keeps calling every in-threshold tick forever. No-op
    // only if the task has no phases at all.
    //
    // Returns true when this call moved to a new phase.
    bool advance(const RobotState& state)
    {
        const std::vector<TaskPhase>& phases = *phases_;
        if (phases.empty()) return false;

        const TaskPhase& cur = phases[phase_index_];
        const double dx = state.pos[0] - cur.goal_pos[0];
        const double dy = state.pos[1] - cur.goal_pos[1];
        const double dz = state.pos[2] - cur.goal_pos[2];
        if (std::sqrt(dx*dx + dy*dy + dz*dz) >= cur.goal_thresh) return false;  // distance gate; dwelling_ untouched

        // Dwell gate: counts ticks spent within goal_thresh, not reset when the
        // robot drifts back out in between — matches RTWholeBodyMPPI's next_goal(),
        // whose Timer only ever increments on calls the driver's distance check let
        // through, and is never reset on a failed check. Kept running even after
        // task_success_ (matches the original driver still calling next_goal()
        // every in-threshold tick forever) so dwelling_ keeps tracking correctly.
        //
        // <= (not <): RTWholeBodyMPPI's Timer.increment() only flips `done` once
        // elapsed_time (pre-incremented) reaches end_time, and that `done` check
        // happens inside the SAME next_goal() call that performed the increment —
        // so it takes waiting_time+1 in-threshold calls to advance a phase, not
        // waiting_time. Advancing on `dwell_ticks_ < waiting_time` fires one tick
        // early on every phase transition. No effect on any task with
        // waiting_time: 0.
        if (++dwell_ticks_ <= cur.waiting_time) {
            dwelling_ = true;   // mid-dwell: settled, not yet cleared to advance
            return false;
        }

        if (phase_index_ + 1 < static_cast<int>(phases.size())) {
            ++phase_index_;
            activate(phase_index_);
            dwell_ticks_ = 0;
            dwelling_ = false;  // resumed traveling toward the new phase's goal
            const TaskPhase& next = phases[phase_index_];
            std::printf("[phase] -> %d (goal=%.2f,%.2f,%.2f gait=%s)\n",
                        phase_index_, next.goal_pos[0], next.goal_pos[1], next.goal_pos[2],
                        next.gait_path.empty() ? next.desired_gait.c_str() : next.gait_path.c_str());
            return true;
        } else if (!task_success_) {
            task_success_ = true;  // dwelling_ untouched here — matches the original
            std::printf("[phase] task complete.\n");
        } else {
            dwelling_ = true;  // settled at the final goal, post-success
        }
        return false;
    }

    // Active phase (nullptr only for a task with no phases).
    const TaskPhase* current_phase() const {
        return phases_->empty() ? nullptr : &(*phases_)[phase_index_];
    }

    const MotionCommand& command() const { return cmd_; }
    void set_command(const MotionCommand& cmd) { cmd_ = cmd; }

    // Active phase's gait (nullptr only for a task with no phases). Non-const
    // so the controller can advance() it once per tick.
    Gait* active_gait() const { return active_gait_; }

    int  phase_index()  const { return phase_index_; }

    // True once the final phase's dwell gate has passed (see advance()).
    bool task_success() const { return task_success_; }

    // True while settled at a waypoint (mid-dwell, or holding after
    // task_success) — mirrors RTWholeBodyMPPI's Timer.waiting. Controllers
    // disable goal-facing heading tracking while true, matching update()'s
    // `not self.timer.waiting` check in the original.
    bool dwelling()     const { return dwelling_; }

private:
    // Point the command and active gait at phases[idx].
    void activate(int idx)
    {
        const TaskPhase& p = (*phases_)[idx];
        cmd_.goal_pos[0] = p.goal_pos[0];
        cmd_.goal_pos[1] = p.goal_pos[1];
        cmd_.goal_pos[2] = p.goal_pos[2];
        cmd_.vx = p.cmd_vel[0];
        cmd_.vy = p.cmd_vel[1];
        active_gait_ = &gaits_.at(resolve_gait_key(p, *named_));
    }

    const std::vector<TaskPhase>* phases_ = nullptr;
    const NamedGaitPaths*         named_  = nullptr;

    // All gaits a task can use, keyed by resolve_gait_key() so both the
    // canonical names and any per-phase gait_path override share one map
    // without key collisions. active_gait_ points at the current phase's entry.
    std::unordered_map<std::string, Gait> gaits_;
    Gait* active_gait_ = nullptr;

    MotionCommand cmd_;

    int  phase_index_  = 0;
    int  dwell_ticks_  = 0;      // cumulative ticks spent within goal_thresh
    bool task_success_ = false;  // true once the final phase's dwell gate passes
    bool dwelling_     = false;
};
