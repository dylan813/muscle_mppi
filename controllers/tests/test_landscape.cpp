// What landscape_scorer is built on, after one update() of the walk task from
// the standing state mppi_sim hands over from (as test_mppi_labels.cpp):
//   score_plans() from solve_context() reproduces plan_cost() for the chosen
//     plan, on any thread, and leaves the controller as it was
//   a context written by ContextWriter and read back by ContextReader scores exactly the same
//   plan_from_knots() passes through the knots: knots_from_plan() gives them
//     back, clamped to [0, 1]
// Builds the real controller and stands the robot up, so it takes a second or two.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include <unistd.h>

#include "check.h"
#include "../common/harness.h"
#include "../muscle/control/context_files.h"
#include "../muscle/control/mppi_locomotion.h"

namespace fs = std::filesystem;

int main()
{
    MPPILocomotion mppi("walk");
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

    mppi.advance_phase(state);
    mppi.update(state);

    const RolloutContext& context = mppi.solve_context();
    const std::vector<double> chosen = mppi.plan();
    const double expected = mppi.plan_cost().total();

    // The chosen plan, scored from the solve's context: plan_cost()'s total.
    // score_plans() sums each step's total, plan_cost() each term over the
    // steps: the same numbers added in another order, hence the tolerance.
    double cost = 0.0;
    mppi.score_plans(context, chosen.data(), 1, &cost);
    CHECK_NEAR(cost, expected, 1e-9 * std::abs(expected));

    // The same plan scores the same on every thread.
    constexpr int kCopies = 8;
    const int stride = task.horizon * NUM_MUSCLES;
    std::vector<double> copies;
    for (int i = 0; i < kCopies; ++i) copies.insert(copies.end(), chosen.begin(), chosen.end());
    std::vector<double> copy_costs(kCopies);
    mppi.score_plans(context, copies.data(), kCopies, copy_costs.data());
    for (double c : copy_costs) CHECK_NEAR(c, cost, 0.0);

    // Scoring left the controller as update() left it.
    CHECK(mppi.plan() == chosen);
    CHECK_NEAR(mppi.plan_cost().total(), expected, 0.0);

    // ContextWriter, then ContextReader: the second row back scores exactly as the original.
    const fs::path dir = fs::temp_directory_path() / ("test_landscape_" + std::to_string(getpid()));
    fs::create_directories(dir);
    const std::string prefix = (dir / "run").string();
    {
        ContextWriter writer(prefix, task.horizon);
        writer.append(context, chosen);
        writer.append(context, chosen);
    }
    const ContextReader reader(prefix, task.horizon);
    CHECK(reader.rows() == 2);
    std::vector<double> plan_back;
    const RolloutContext back = reader.read(1, plan_back);
    CHECK(plan_back == chosen);
    double cost_back = 0.0;
    mppi.score_plans(back, plan_back.data(), 1, &cost_back);
    CHECK_NEAR(cost_back, cost, 0.0);

    bool threw = false;
    try {
        reader.read(2, plan_back);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
    fs::remove_all(dir);

    // Knots -> plan -> knots: the cubic passes through its knots, then each
    // value is clamped to [0, 1]. Two knots pushed out of range test the clamp.
    std::vector<double> knots(task.n_knots * NUM_MUSCLES);
    mppi.knots_from_plan(chosen.data(), knots.data());
    knots[0] = -0.3;
    knots[NUM_MUSCLES + 5] = 1.4;
    std::vector<double> plan(stride), knots_back(knots.size());
    mppi.plan_from_knots(knots.data(), plan.data());
    mppi.knots_from_plan(plan.data(), knots_back.data());
    for (size_t i = 0; i < knots.size(); ++i) CHECK_NEAR(knots_back[i], std::clamp(knots[i], 0.0, 1.0), 1e-12);

    mj_deleteData(d);
    mj_deleteModel(m);
    return check::exit_code();
}
