// Standalone MuJoCo simulation test for MPPILocomotionPD (direct joint-space
// PD actuation — the PD-actuated mirror of ../muscle/mppi_sim.cpp, which drives the
// muscle-actuated MPPILocomotion instead).
// No DDS, no real-time constraint — MPPI runs as fast as possible against a
// local mjData simulation. Useful for verifying the controller works before
// worrying about latency.
//
// Run from any directory (e.g. controllers/build/):
//   ./pd_mppi_sim [task] [yaml] [output.csv] [--name <run>] [--save <name>] [--no-gif]
//                 [--contexts <prefix>]
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
// --contexts <prefix> also writes, one row per CSV row, what that row's update()
// planned from and the plan it chose, in float64 (control/context_files_pd.h),
// so pd_landscape_scorer can roll plans out from a row exactly as that update()
// did. Created, with any missing folders, when the first row is logged; the
// prefix's old files are deleted first, at startup. <prefix> is relative to
// the current directory, like [output.csv].
//
// Output CSV columns:
//   t, px, py, pz, vx, vy, vz, qw, roll_deg, wx, wy, wz, dq_j0..dq_j{NUM_JOINTS-1},
//   fn_FR..fn_RL, ft_FR..ft_RL, solve_ms, qdes_j0..qdes_j{NUM_JOINTS-1},
//   qdes_filt_j0..qdes_filt_j{NUM_JOINTS-1}, tau_j0..tau_j{NUM_JOINTS-1},
//   tau_applied_j0..tau_applied_j{NUM_JOINTS-1}, plan_cost
//   and, last, the push columns for a task with a push: block (Pusher in common/harness.h).
//   tau_j* is the motor command (the PD torque) of the row's last physics step,
//   before the torque limit and the motor's lag; tau_applied_j* the torque the
//   motor applied over that step (common/motor.h) — the same convention and
//   column names the muscle variant uses. Logged rather than reconstructed in
//   analysis, which only works while one command lasts exactly one logged step.
//   wx/wy/wz are body-frame angular velocity, fn_*/ft_* the normal and
//   tangential ground reaction force per foot, and solve_ms that update()'s
//   compute time — all written by write_csv_row_base() in common/log.h.
//   px/py/pz are the trunk frame origin (free-joint qpos[0:3]) — matching
//   what step_cost() scores against goal_pos in mppi_locomotion_pd.cpp, and
//   matching RTWholeBodyMPPI's own position cost reference.
//   vx/vy/vz are body-frame linear velocity (rotated from the free joint's
//   world-frame qvel[0:3]), matching the body-frame convention step_cost()
//   uses when comparing against cmd_.vx/vy in mppi_locomotion_pd.cpp.
//   qdes_j* are the commanded joint targets (trajectory_[0..NUM_JOINTS-1]
//   after each solve) — the PD-variant analogue of the muscle variant's
//   act_cmd_m* command columns. qdes_filt_j* are those targets after the
//   activation-dynamics filter, i.e. what the PD law acted on (the analogue of
//   act_m*; equal to qdes_j* when the task has no activation_dynamics block).
//   plan_cost is the chosen plan's predicted cost over the horizon, rolled out
//   from the state that update() started from (MPPILocomotionPD::plan_cost());
//   computed outside the timed solve.

#include <filesystem>
#include <memory>
#include <ostream>
#include <string>
#include <system_error>
#include <vector>

#include "control/context_files_pd.h"
#include "control/mppi_locomotion_pd.h"
#include "../common/harness.h"

// Moves argv into rest, except `<flag> <prefix>` (or `<flag>=<prefix>`), which
// goes to prefix, so run_sim() never sees it. Returns false if the prefix is
// missing. (As ../muscle/mppi_sim.cpp.)
static bool take_prefix_arg(int argc, char** argv, const std::string& flag, std::vector<char*>& rest,
                            std::string& prefix)
{
    for (int i = 0; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool spaced = i > 0 && arg == flag;
        const bool joined = i > 0 && arg.rfind(flag + "=", 0) == 0;
        if (!spaced && !joined) {
            rest.push_back(argv[i]);
            continue;
        }

        prefix = joined ? arg.substr(flag.size() + 1) : (i + 1 < argc ? argv[++i] : "");
        if (prefix.empty()) {
            fprintf(stderr, "%s needs a file prefix\n", flag.c_str());
            return false;
        }
    }
    return true;
}

// Delete files a previous run left at prefix + each suffix.
static void remove_files(const std::string& prefix, const std::vector<std::string>& suffixes)
{
    for (const std::string& suffix : suffixes) {
        std::error_code ec;
        std::filesystem::remove(prefix + suffix, ec);
    }
}

// Create prefix's missing folders, as run_sim() does for the CSV.
static void create_folders(const std::string& prefix)
{
    const std::filesystem::path dir = std::filesystem::path(prefix).parent_path();
    std::error_code ec;
    if (!dir.empty()) std::filesystem::create_directories(dir, ec);
}

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
    std::vector<char*> args;
    std::string contexts_prefix;
    if (!take_prefix_arg(argc, argv, "--contexts", args, contexts_prefix)) return 1;
    std::unique_ptr<ContextWriterPD> contexts;
    if (!contexts_prefix.empty()) remove_files(contexts_prefix, context_file_suffixes());

    SimSpec<MPPILocomotionPD> spec;
    spec.default_yaml = kDefaultTasksPdYaml;
    spec.default_csv  = repo_path("analysis/data/pd_mppi_sim/pd_mppi_sim.csv");

    spec.joint_damping = [](const MPPILocomotionPD& mppi) { return mppi.task_ref().pd.joint_damping; };

    // Trunk origin = free-joint qpos[0:3], which step_cost() scores.
    spec.log_position = [](const mjModel*, mjData* d, int) -> const double* { return d->qpos; };

    spec.extra_header = [](std::ostream& csv) {
        for (int j = 0; j < NUM_JOINTS; ++j) csv << ",qdes_j" << j;
        for (int j = 0; j < NUM_JOINTS; ++j) csv << ",qdes_filt_j" << j;
        for (int j = 0; j < NUM_JOINTS; ++j) csv << ",tau_j" << j;
        for (int j = 0; j < NUM_JOINTS; ++j) csv << ",tau_applied_j" << j;
        csv << ",plan_cost";
    };
    spec.extra_row = [&contexts, &contexts_prefix](std::ostream& csv, MPPILocomotionPD& mppi,
                                                   const RobotState& state) {
        const double* qdes = mppi.q_des();
        for (int j = 0; j < NUM_JOINTS; ++j) csv << "," << qdes[j];
        const double* qfilt = mppi.q_des_filt();
        for (int j = 0; j < NUM_JOINTS; ++j) csv << "," << qfilt[j];
        const double* tau = mppi.torque();
        for (int j = 0; j < NUM_JOINTS; ++j) csv << "," << tau[j];
        double applied[NUM_JOINTS];
        mppi.applied_torque(state, applied);
        for (int j = 0; j < NUM_JOINTS; ++j) csv << "," << applied[j];
        csv << "," << mppi.plan_cost();

        // The files open with the first row: the horizon comes from the
        // controller run_sim() builds.
        if (!contexts_prefix.empty()) {
            if (!contexts) {
                create_folders(contexts_prefix);
                contexts = std::make_unique<ContextWriterPD>(contexts_prefix, mppi.task_ref().horizon);
            }
            contexts->append(mppi.solve_context(), mppi.plan());
        }
    };

    return run_sim(static_cast<int>(args.size()), args.data(), spec);
}
