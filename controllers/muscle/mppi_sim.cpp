// Standalone MuJoCo simulation test for MPPILocomotion.
// No DDS, no real-time constraint — MPPI runs as fast as possible against a
// local mjData simulation. Useful for verifying the controller works before
// worrying about latency.
//
// Run from any directory (e.g. controllers/build/):
//   ./mppi_sim [task] [yaml] [output.csv] [--save <name>]
// Defaults read controllers/muscle/utils/tasks.yaml and write to
// analysis/data/mppi_sim/mppi_sim.csv (created on first run if missing), both
// resolved against the repo root (common/paths.h). An explicit [yaml] or
// [output.csv] is relative to the current directory, as usual.
//
// --save copies this run's CSVs, once it finishes, into
// analysis/log/trials/<name>/trial_NNN/ — one directory per run, so
// repeated trials of the same task accumulate instead of overwriting.
//
// Output CSV columns:
//   t, px, py, pz, vx, vy, vz, qw, roll_deg, dq_j0..dq_j{NUM_JOINTS-1}, act_m0..act_m{NUM_MUSCLES-1}
//   px/py/pz are the whole-robot (trunk + legs) center of mass, i.e. the base
//   body's subtree_com — matching what step_cost() scores against goal_pos
//   in mppi_locomotion.cpp, not the trunk frame origin.
//   vx/vy/vz are body-frame linear velocity (rotated from the free joint's
//   world-frame qvel[0:3]), matching the body-frame convention step_cost()
//   uses when comparing against cmd_.vx/vy in mppi_locomotion.cpp.

#include <mujoco/mujoco.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "control/mppi_locomotion.h"
#include "../common/sim_harness.h"
#include "../common/trial_log.h"

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    // --save is pulled out first so it can sit anywhere on the command line
    // without shifting the positional [task] [yaml] [output.csv] arguments.
    std::string trial_name;
    const std::vector<std::string> args = trial_log::parse_args(argc, argv, trial_name);
    const size_t nargs = args.size();

    const std::string task_name = (nargs >= 2) ? args[1] : "walk";
    const std::string yaml_path = (nargs >= 3) ? args[2] : kDefaultTasksYaml;
    const std::string csv_path  = (nargs >= 4) ? args[3]
                                               : repo_path("analysis/data/mppi_sim/mppi_sim.csv");

    printf("Task: %s  |  YAML: %s  |  CSV: %s\n",
           task_name.c_str(), yaml_path.c_str(), csv_path.c_str());
    if (!trial_name.empty()) printf("Saving trial under: %s\n", trial_name.c_str());

    // ── load MPPI (also loads the model internally) ──────────────────────────
    MPPILocomotion mppi(task_name, yaml_path);

    // ── load a separate sim model/data ───────────────────────────────────────
    const TaskConfig& task = mppi.task_ref();
    char err[1000];
    mjModel* m = mj_loadXML(task.model_path.c_str(), nullptr, err, sizeof(err));
    if (!m) { fprintf(stderr, "mj_loadXML: %s\n", err); return 1; }
    m->opt.timestep = task.dt;
    mjData* d = mj_makeData(m);

    // Resolve the base body (mirrors MPPILocomotion::base_bid_ resolution)
    // so logged position matches the whole-robot CoM the cost function scores.
    const int base_bid = find_base_body(m);

    // set joint damping to match MPPI's internal model
    int qa[NUM_JOINTS], qv[NUM_JOINTS];
    joint_addresses(m, qa, qv);
    for (int j = 0; j < NUM_JOINTS; ++j) m->dof_damping[qv[j]] = task.muscle.kd_sim[j];

    // place robot feet on ground
    place_on_ground(m, d, qa, task.spawn_height_offset);

    // ── output files ─────────────────────────────────────────────────────────
    std::ofstream csv;
    if (!open_output(csv_path, csv)) return 1;
    write_csv_header_base(csv);
    for (int m = 0; m < NUM_MUSCLES; ++m) csv << ",act_m" << m;
    csv << "\n";

    const std::string qpos_path = qpos_path_for(csv_path);
    std::ofstream qpos_log(qpos_path);

    // ── stand-up phase (PD, mirrors mppi_controller.cpp) ─────────────────────
    printf("Standing up (%.1f s)...\n", kStandupSecs + kHoldSecs);
    run_standup(m, d, qa, qv, task.dt);
    printf("Stand-up complete. Body height: %.3f m\n", d->qpos[2]);

    // ── MPPI loop ─────────────────────────────────────────────────────────────
    printf("Running MPPI for %d convergence solves then logging...\n",
           kConvergenceSolves);

    const double sim_duration = task.sim_duration;   // seconds of MPPI control to record
    double sim_t = 0.0;
    int solve_count = 0;
    double solve_sum_ms = 0.0;
    bool converged = false;

    while (sim_t < sim_duration) {
        // --- MPPI solve ---
        RobotState state = read_state(d, qa, qv);

        auto t0 = std::chrono::steady_clock::now();
        double tau[NUM_JOINTS] = {};
        mppi.advance_phase(state);
        mppi.update(state, tau);
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();

        solve_sum_ms += ms;
        ++solve_count;

        if (!converged && solve_count >= kConvergenceSolves) {
            printf("Converged (avg solve %.1f ms). Starting trajectory logging.\n",
                   solve_sum_ms / solve_count);
            converged = true;
        }

        // --- apply torques for one control step ---
        for (int j = 0; j < NUM_JOINTS; ++j)
            d->ctrl[j] = tau[j];
        mj_step(m, d);
        sim_t += task.dt;

        // --- log ---
        if (converged) {
            write_csv_row_base(csv, sim_t, d->subtree_com + base_bid * 3, d, qv);
            const double* act = mppi.activation();
            for (int j = 0; j < NUM_MUSCLES; ++j) csv << "," << act[j];
            csv << "\n";

            // save full qpos for GIF rendering
            write_qpos_row(qpos_log, m, d);
        }

        // --- safety: stop if robot falls ---
        if (has_fallen(d)) {
            printf("Robot fell at t=%.2f s — stopping.\n", sim_t);
            break;
        }

        // --- stop once the task's final phase has been reached and held ---
        // Mirrors pd_mppi_sim.cpp. Without this the muscle variant ran out the
        // full sim_duration on every success while the PD variant stopped, so
        // the two recorded different spans and any per-run average was taken
        // over a different window per controller.
        if (converged && mppi.task_success()) {
            printf("Task complete at t=%.2f s — stopping.\n", sim_t);
            break;
        }
    }

    printf("Done. Logged to %s and %s\n", csv_path.c_str(), qpos_path.c_str());
    printf("Avg MPPI solve: %.1f ms over %d solves\n",
           solve_sum_ms / solve_count, solve_count);

    // Flush before copying — a run that fell still gets its partial logs saved.
    csv.close();
    qpos_log.close();
    if (!trial_name.empty()) {
        const std::string trial_dir = trial_log::save(trial_name, {csv_path, qpos_path});
        if (!trial_dir.empty()) printf("Trial saved to %s\n", trial_dir.c_str());
    }

    mj_deleteData(d);
    mj_deleteModel(m);
    return 0;
}
