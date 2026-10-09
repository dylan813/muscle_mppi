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
//   t, px, py, pz, vx, vy, vz, qw, roll_deg, wx, wy, wz, dq_j0..dq_j{NUM_JOINTS-1},
//   fn_FR..fn_RL, ft_FR..ft_RL, solve_ms, act_m0..act_m{NUM_MUSCLES-1},
//   act_cmd_m0..act_cmd_m{NUM_MUSCLES-1}, fl_active_m*, fv_m*, fl_passive_m*,
//   tau_j0..tau_j{NUM_JOINTS-1}, tau_applied_j0..tau_applied_j{NUM_JOINTS-1},
//   cost_{pos,orient,vel,ang_vel,gait}, plan_{pos,orient,vel,ang_vel,gait},
//   sample_min, ess
//   tau is the motor command (Hill torque minus driver damping) of the row's
//   last physics step, before the torque limit and the motor's lag;
//   tau_applied the torque the motor applied over that step (common/motor.h).
//   With one physics step per control step and no motor lag, tau is computed
//   from the previous row's state and tau_applied is tau within the limits.
//   act_m* is the filtered activation that torque came from; act_cmd_m* is the
//   command it was filtering toward (equal when activation dynamics are off).
//   fl_active_m*, fv_m*, fl_passive_m* are the Hill factors that torque came
//   from (HillFactors in muscle.h): F = (fl_active·fv·act + fl_passive)·peak_force.
//   Ablated components log their neutral value (fl_active = fv = 1, fl_passive = 0).
//   Cost columns, split by cost term (CostTerms in mppi_locomotion.h), from
//   the update() that produced the row; analysis only, computed outside the
//   timed solve:
//     cost_*      cost of the row's state, i.e. what the executed step
//                 incurred. Sums over a run to the cost the robot incurred.
//     plan_*      the chosen plan's predicted cost over the horizon, rolled
//                 out from the state that update() started from. A forecast:
//                 overlapping windows, so not summable across rows.
//     sample_min  lowest of that update()'s sample costs (horizon totals).
//     ess         effective sample size of its softmin weights, 1/Σw²:
//                 1 = one sample decided the plan, n_samples = all counted equally.
//   wx/wy/wz are body-frame angular velocity, fn_*/ft_* the normal and
//   tangential ground reaction force per foot, and solve_ms that update()'s
//   compute time — all written by write_csv_row_base() in common/log.h.
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
    // Recomputed for the post-step state first, as step_cost() does.
    spec.log_position = [](const mjModel* m, mjData* d, int base_bid) -> const double* {
        mj_kinematics(m, d);
        mj_comPos(m, d);
        return d->subtree_com + base_bid * 3;
    };

    spec.extra_header = [](std::ostream& csv) {
        for (int m = 0; m < NUM_MUSCLES; ++m) csv << ",act_m" << m;
        for (int m = 0; m < NUM_MUSCLES; ++m) csv << ",act_cmd_m" << m;
        for (const char* factor : {"fl_active", "fv", "fl_passive"})
            for (int m = 0; m < NUM_MUSCLES; ++m) csv << "," << factor << "_m" << m;
        for (int j = 0; j < NUM_JOINTS; ++j)  csv << ",tau_j" << j;
        for (int j = 0; j < NUM_JOINTS; ++j)  csv << ",tau_applied_j" << j;
        for (const char* prefix : {"cost_", "plan_"})
            for (const char* term : {"pos", "orient", "vel", "ang_vel", "gait"})
                csv << "," << prefix << term;
        csv << ",sample_min,ess";
    };
    spec.extra_row = [](std::ostream& csv, MPPILocomotion& mppi, const RobotState& state) {
        const double* act = mppi.activation();
        for (int j = 0; j < NUM_MUSCLES; ++j) csv << "," << act[j];
        const double* cmd = mppi.act_cmd();
        for (int j = 0; j < NUM_MUSCLES; ++j) csv << "," << cmd[j];
        const HillFactors f = mppi.muscle_factors();
        for (const double* factor : {f.fl_active, f.fv, f.fl_passive})
            for (int m = 0; m < NUM_MUSCLES; ++m) csv << "," << factor[m];
        const double* tau = mppi.torque();
        for (int j = 0; j < NUM_JOINTS; ++j) csv << "," << tau[j];
        double applied[NUM_JOINTS];
        mppi.applied_torque(state, applied);
        for (int j = 0; j < NUM_JOINTS; ++j) csv << "," << applied[j];

        for (const CostTerms& c : {mppi.executed_cost(state), mppi.plan_cost()})
            csv << "," << c.pos << "," << c.orient << "," << c.vel << "," << c.ang_vel << "," << c.gait;
        csv << "," << mppi.sample_min() << "," << mppi.ess();
    };

    return run_sim(argc, argv, spec);
}
