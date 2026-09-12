// Standalone MuJoCo simulation test for MPPILocomotion.
// No DDS, no real-time constraint — MPPI runs as fast as possible against a
// local mjData simulation. Useful for verifying the controller works before
// worrying about latency.
//
// Run from any directory (e.g. controllers/build/):
//   ./mppi_sim [task] [yaml] [output.csv] [--name <run>] [--save <name>] [--no-gif]
//   ./mppi_sim walk_rough --name rough_test1   # -> analysis/data/mppi_sim/rough_test1.csv/_qpos.csv/.gif
// Defaults read controllers/muscle/utils/tasks.yaml and write to
// analysis/data/mppi_sim/mppi_sim.csv (created on first run if missing), both
// resolved against the repo root (common/task_config.h). An explicit [yaml] or
// [output.csv] is relative to the current directory, as usual.
//
// --save copies this run's CSVs, once it finishes, into
// analysis/log/trials/<name>/trial_NNN/ — one directory per run, so
// repeated trials of the same task accumulate instead of overwriting.
//
// After the run, the logged rollout is rendered to <output>.gif next to the CSV
// (copied into the trial with --save). Any previous GIF of that name is always
// deleted first, so --no-gif leaves no GIF there at all. See run_sim() in
// common/harness.h.
//
// Output CSV columns:
//   t, px, py, pz, vx, vy, vz, qw, roll_deg, dq_j0..dq_j{NUM_JOINTS-1}, act_m0..act_m{NUM_MUSCLES-1}
//   px/py/pz are the whole-robot (trunk + legs) center of mass, i.e. the base
//   body's subtree_com — matching what step_cost() scores against goal_pos
//   in mppi_locomotion.cpp, not the trunk frame origin.
//   vx/vy/vz are body-frame linear velocity (rotated from the free joint's
//   world-frame qvel[0:3]), matching the body-frame convention step_cost()
//   uses when comparing against cmd_.vx/vy in mppi_locomotion.cpp.

#include <ostream>

#include "control/mppi_locomotion.h"
#include "../common/harness.h"

int main(int argc, char** argv)
{
    SimSpec<MPPILocomotion> spec;
    spec.default_yaml = kDefaultTasksYaml;
    spec.default_csv  = repo_path("analysis/data/mppi_sim/mppi_sim.csv");

    spec.joint_damping = [](const MPPILocomotion& mppi) { return mppi.task_ref().muscle.kd_sim; };

    // Whole-robot CoM (base body's subtree_com), which step_cost() scores.
    spec.log_position = [](const mjData* d, int base_bid) -> const double* {
        return d->subtree_com + base_bid * 3;
    };

    spec.extra_header = [](std::ostream& csv) {
        for (int m = 0; m < NUM_MUSCLES; ++m) csv << ",act_m" << m;
    };
    spec.extra_row = [](std::ostream& csv, const MPPILocomotion& mppi) {
        const double* act = mppi.activation();
        for (int j = 0; j < NUM_MUSCLES; ++j) csv << "," << act[j];
    };

    return run_sim(argc, argv, spec);
}
