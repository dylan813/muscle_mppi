// Standalone MuJoCo simulation test for MPPILocomotionPD (direct joint-space
// PD actuation — the PD-actuated mirror of ../muscle/mppi_sim.cpp, which drives the
// muscle-actuated MPPILocomotion instead).
// No DDS, no real-time constraint — MPPI runs as fast as possible against a
// local mjData simulation. Useful for verifying the controller works before
// worrying about latency.
//
// Run from any directory (e.g. controllers/build/):
//   ./pd_mppi_sim [task] [yaml] [output.csv] [--name <run>] [--save <name>] [--no-gif]
//   ./pd_mppi_sim walk_rough --name rough_test1   # -> analysis/data/pd_mppi_sim/rough_test1.csv/_qpos.csv/.gif
// Defaults read controllers/pd/utils/tasks_pd.yaml and write to
// analysis/data/pd_mppi_sim/pd_mppi_sim.csv (a dedicated output directory,
// mirroring analysis/data/mppi_sim/ for the muscle-actuated mppi_sim binary,
// created on first run if missing), both resolved against the repo root
// (common/task_config.h). An explicit [yaml] or [output.csv] is relative to the
// current directory, as usual.
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
//   t, px, py, pz, vx, vy, vz, qw, roll_deg, dq_j0..dq_j{NUM_JOINTS-1}, qdes_j0..qdes_j{NUM_JOINTS-1}
//   px/py/pz are the trunk frame origin (free-joint qpos[0:3]) — matching
//   what step_cost() scores against goal_pos in mppi_locomotion_pd.cpp, and
//   matching RTWholeBodyMPPI's own position cost reference.
//   vx/vy/vz are body-frame linear velocity (rotated from the free joint's
//   world-frame qvel[0:3]), matching the body-frame convention step_cost()
//   uses when comparing against cmd_.vx/vy in mppi_locomotion_pd.cpp.
//   qdes_j* are the commanded joint targets (trajectory_[0..NUM_JOINTS-1]
//   after each solve) — the PD-variant analogue of the muscle variant's
//   act_m* activation columns.

#include <ostream>

#include "control/mppi_locomotion_pd.h"
#include "../common/harness.h"

// Sim model contacts — run_sim() loads the "real world" model the robot is
// simulated on, separately from the planner's rollout model.
//
// Deliberately NOT applying the mjENBL_OVERRIDE contact override here.
// RTWholeBodyMPPI's own interface/simulator.py (the "real world" stepper,
// this model's counterpart) sets o_solref but leaves
// `self.model.opt.enableflags = 1` commented out (simulator.py:51) — so
// the override is inert there and the real simulated robot uses each
// geom's own contact tuning, unmodified. Only RTWholeBodyMPPI's MPPI
// planner (base_controller.py, ported in BaseMPPIPD's constructor) has
// the override actually active, since it's only used for cost-evaluation
// rollouts. Applying it here too (an earlier version of this file did)
// would make the actually-simulated robot's contacts diverge from
// RTWholeBodyMPPI's, not match it.

// Note: go2.xml's foot geoms keep their own contact tuning here, on
// purpose. RTWholeBodyMPPI's go1_mppi.xml uses a much softer foot
// (solimp="0.015 1 0.031", friction 0.8) than go2's MuJoCo-default
// solimp/0.4 friction, but that is a property of *their robot model*, not
// of the MPPI code being ported — and an A/B of go1's values on this robot
// showed no measurable change in body bounce or roll. Left alone so the
// simulated robot stays the Go2 that Unitree's model describes.

int main(int argc, char** argv)
{
    SimSpec<MPPILocomotionPD> spec;
    spec.default_yaml = kDefaultTasksPdYaml;
    spec.default_csv  = repo_path("analysis/data/pd_mppi_sim/pd_mppi_sim.csv");

    spec.joint_damping = [](const MPPILocomotionPD& mppi) { return mppi.task_ref().pd.joint_damping; };

    // Trunk origin = free-joint qpos[0:3], which step_cost() scores.
    spec.log_position = [](const mjModel*, mjData* d, int) -> const double* { return d->qpos; };

    spec.extra_header = [](std::ostream& csv) {
        for (int j = 0; j < NUM_JOINTS; ++j) csv << ",qdes_j" << j;
    };
    spec.extra_row = [](std::ostream& csv, const MPPILocomotionPD& mppi) {
        const double* qdes = mppi.q_des();
        for (int j = 0; j < NUM_JOINTS; ++j) csv << "," << qdes[j];
    };

    return run_sim(argc, argv, spec);
}
