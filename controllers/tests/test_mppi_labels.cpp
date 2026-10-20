// MPPILocomotion's training-data getters, observation() and plan(), after one
// update() of the walk task from the standing state mppi_sim hands over from.
// Builds the real controller and stands the robot up, so it takes a second or two.

#include <algorithm>
#include <cstdio>
#include <vector>

#include <mujoco/mujoco.h>

#include "check.h"
#include "../common/harness.h"
#include "../muscle/control/mppi_locomotion.h"

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

    // plan(): horizon × NUM_MUSCLES activations in [0, 1], step 0 the held command.
    const std::vector<double>& plan = mppi.plan();
    CHECK(plan.size() == static_cast<size_t>(task.horizon * NUM_MUSCLES));
    CHECK(std::all_of(plan.begin(), plan.end(), [](double a) { return a >= 0.0 && a <= 1.0; }));
    for (int k = 0; k < NUM_MUSCLES; ++k) CHECK_NEAR(plan[k], mppi.act_cmd()[k], 0.0);

    // observation(): build_observation() on the state update() planned from,
    // with the CoM computed here on the test's own model. No actuate() has run
    // since, so activation() is still the activation update() planned from.
    mj_forward(m, d);
    const double* com = d->subtree_com + 3 * find_base_body(m);
    double expected[OBS_DIM], y[OBS_DIM];
    build_observation(state, com, mppi.activation(), mppi.command().goal_pos, expected);
    mppi.observation(y);
    for (int i = 0; i < OBS_DIM; ++i) CHECK_NEAR(y[i], expected[i], 1e-9);

    mj_deleteData(d);
    mj_deleteModel(m);
    return check::exit_code();
}
