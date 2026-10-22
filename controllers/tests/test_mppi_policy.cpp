// MPPILocomotion with the task's policy: block (muscle/control/mppi_locomotion.h).
// Builds the walk task's controller from tasks_yaml, stands the robot up as
// test_mppi_labels.cpp does, then runs `updates` update()s from that state,
// writing for each update u, to out_dir:
//
//   u<u>_plan_before.npy  plan() before the update (before its shift)
//   u<u>_y.npy            observation(): the y the policy conditioned on
//   u<u>_plan.npy         plan() after the update
//   u<u>_costs.npy        sample_min, mppi_min, policy_min
//   u<u>_weight.npy       policy_weight
//   u<u>_terms.npy        mppi_best_terms, policy_best_terms: rows of pos, orient, vel, ang_vel, gait
//   u<u>_ref_dev.npy      ref_dev_plan, ref_dev_mppi_best, ref_dev_policy_best
//
//   test_mppi_policy <tasks_yaml> <out_dir> <updates>
//
// learning/tests/test_mppi_policy.py runs this with the YAMLs and policy files it writes.

#include <algorithm>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <mujoco/mujoco.h>

#include "../common/harness.h"
#include "../common/npy_writer.h"
#include "../muscle/control/mppi_locomotion.h"

namespace fs = std::filesystem;

// One row of row_shape, values, to path.
static void write_npy(const std::string& path, std::vector<int> row_shape, const double* values)
{
    NpyWriter(path, std::move(row_shape)).append(values);
}

int main(int argc, char** argv)
{
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <tasks_yaml> <out_dir> <updates>\n", argv[0]);
        return 2;
    }
    const fs::path out = argv[2];
    const int updates  = std::stoi(argv[3]);

    try {
        MPPILocomotion mppi("walk", argv[1]);
        const TaskConfig& task = mppi.task_ref();

        // Stand the robot up on its own model, as run_sim() does.
        char err[1000];
        mjModel* m = mj_loadXML(task.model_path.c_str(), nullptr, err, sizeof(err));
        if (!m) { std::fprintf(stderr, "mj_loadXML: %s\n", err); return 1; }
        m->opt.timestep = task.motor.physics_dt;
        mjData* d = mj_makeData(m);
        int qa[NUM_JOINTS], qv[NUM_JOINTS];
        joint_addresses(m, qa, qv);
        for (int j = 0; j < NUM_JOINTS; ++j) m->dof_damping[qv[j]] = task.muscle.kd_sim[j];
        place_on_ground(m, d, qa, task.spawn_height_offset);
        run_standup(m, d, qa, qv);
        const RobotState state = read_state(d, qa, qv);
        mj_deleteData(d);
        mj_deleteModel(m);

        fs::create_directories(out);
        const std::vector<int> plan_shape = {task.horizon, NUM_MUSCLES};
        for (int u = 0; u < updates; ++u) {
            const std::string prefix = (out / ("u" + std::to_string(u))).string();
            write_npy(prefix + "_plan_before.npy", plan_shape, mppi.plan().data());

            mppi.advance_phase(state);
            mppi.update(state);

            double y[OBS_DIM];
            mppi.observation(y);
            const double costs[] = {mppi.sample_min(), mppi.mppi_min(), mppi.policy_min()};
            const double weight  = mppi.policy_weight();
            double terms[2][5];
            for (int g = 0; g < 2; ++g) {
                const CostTerms& c = g == 0 ? mppi.mppi_best_terms() : mppi.policy_best_terms();
                const double row[] = {c.pos, c.orient, c.vel, c.ang_vel, c.gait};
                std::copy(row, row + 5, terms[g]);
            }
            const double ref_dev[] = {mppi.ref_dev_plan(), mppi.ref_dev_mppi_best(), mppi.ref_dev_policy_best()};
            write_npy(prefix + "_y.npy",       {OBS_DIM},  y);
            write_npy(prefix + "_plan.npy",    plan_shape, mppi.plan().data());
            write_npy(prefix + "_costs.npy",   {3},        costs);
            write_npy(prefix + "_weight.npy",  {},         &weight);
            write_npy(prefix + "_terms.npy",   {2, 5},     terms[0]);
            write_npy(prefix + "_ref_dev.npy", {3},        ref_dev);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
