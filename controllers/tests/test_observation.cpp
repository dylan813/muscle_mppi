// build_observation() (muscle/control/observation.h): frames, the heading
// frame's independence from pitch, and where each term lands in y.

#include <cmath>
#include <limits>

#include "check.h"
#include "../muscle/control/observation.h"

static constexpr double kTol = 1e-12;

// Trunk orientation: yaw about world z, then pitch about the yawed body y.
static void set_orientation(RobotState& state, double yaw, double pitch)
{
    const double z_axis[3] = {0.0, 0.0, 1.0};
    const double y_axis[3] = {0.0, 1.0, 0.0};
    double q_yaw[4], q_pitch[4];
    mju_axisAngle2Quat(q_yaw, z_axis, yaw);
    mju_axisAngle2Quat(q_pitch, y_axis, pitch);
    mju_mulQuat(state.quat, q_yaw, q_pitch);
}

static void observe(const RobotState& state, const double com[3], const double goal[3], double y[OBS_DIM])
{
    const double act[NUM_MUSCLES] = {};
    build_observation(state, com, act, goal, y);
}

// Upright at rest: gravity straight down, no motion, offset = goal − CoM.
static void test_upright_at_rest()
{
    RobotState state;
    const double com[3]  = {0.01, 0.0, 0.32};
    const double goal[3] = {2.0, 0.0, 0.34};
    double y[OBS_DIM];
    observe(state, com, goal, y);

    CHECK_NEAR(y[OBS_GOAL],     1.99, kTol);
    CHECK_NEAR(y[OBS_GOAL + 1], 0.0,  kTol);
    CHECK_NEAR(y[OBS_GOAL + 2], 0.02, kTol);

    CHECK_NEAR(y[OBS_GRAVITY],     0.0,  kTol);
    CHECK_NEAR(y[OBS_GRAVITY + 1], 0.0,  kTol);
    CHECK_NEAR(y[OBS_GRAVITY + 2], -1.0, kTol);

    for (int k = 0; k < 3; ++k) {
        CHECK_NEAR(y[OBS_LIN_VEL + k], 0.0, kTol);
        CHECK_NEAR(y[OBS_ANG_VEL + k], 0.0, kTol);
    }
}

// Yawed 90° (facing world +y): a goal and a velocity along world +x point to
// the robot's right, so they read as left = −1.
static void test_yaw_90()
{
    RobotState state;
    set_orientation(state, M_PI / 2, 0.0);
    state.vel[0] = 1.0;
    const double com[3]  = {0.0, 0.0, 0.3};
    const double goal[3] = {1.0, 0.0, 0.3};
    double y[OBS_DIM];
    observe(state, com, goal, y);

    CHECK_NEAR(y[OBS_GOAL],     0.0,  kTol);
    CHECK_NEAR(y[OBS_GOAL + 1], -1.0, kTol);
    CHECK_NEAR(y[OBS_GOAL + 2], 0.0,  kTol);

    CHECK_NEAR(y[OBS_LIN_VEL],     0.0,  kTol);
    CHECK_NEAR(y[OBS_LIN_VEL + 1], -1.0, kTol);
    CHECK_NEAR(y[OBS_LIN_VEL + 2], 0.0,  kTol);

    CHECK_NEAR(y[OBS_GRAVITY + 2], -1.0, kTol);
}

// Pitched 10°: the goal offset reads the same as upright (heading frame), while
// gravity tilts in the body frame to (sin 10°, 0, −cos 10°). A body-frame
// offset would read (cos 10°, 0, sin 10°) instead.
static void test_pitch_ignored_by_goal_offset()
{
    const double pitch = 10.0 * M_PI / 180.0;
    const double com[3]  = {0.0, 0.0, 0.3};
    const double goal[3] = {1.0, 0.0, 0.3};

    for (double yaw : {0.0, M_PI / 2}) {
        RobotState upright, pitched;
        set_orientation(upright, yaw, 0.0);
        set_orientation(pitched, yaw, pitch);
        double y_up[OBS_DIM], y_pitch[OBS_DIM];
        observe(upright, com, goal, y_up);
        observe(pitched, com, goal, y_pitch);

        for (int k = 0; k < 3; ++k) CHECK_NEAR(y_pitch[OBS_GOAL + k], y_up[OBS_GOAL + k], kTol);

        CHECK_NEAR(y_pitch[OBS_GRAVITY],     std::sin(pitch),  kTol);
        CHECK_NEAR(y_pitch[OBS_GRAVITY + 1], 0.0,              kTol);
        CHECK_NEAR(y_pitch[OBS_GRAVITY + 2], -std::cos(pitch), kTol);
    }
}

// Every index is written, and each term lands at its index.
static void test_layout()
{
    CHECK(OBS_DIM == 60);

    RobotState state;
    for (int k = 0; k < 3; ++k) state.gyro[k] = 0.1 * (k + 1);
    for (int j = 0; j < NUM_JOINTS; ++j) {
        state.q[j]  = 100.0 + j;
        state.dq[j] = 200.0 + j;
    }
    double act[NUM_MUSCLES];
    for (int m = 0; m < NUM_MUSCLES; ++m) act[m] = 0.01 * m;
    const double com[3]  = {0.0, 0.0, 0.3};
    const double goal[3] = {1.0, 0.0, 0.3};

    double y[OBS_DIM];
    for (double& v : y) v = std::numeric_limits<double>::quiet_NaN();
    build_observation(state, com, act, goal, y);

    for (double v : y) CHECK(std::isfinite(v));
    for (int k = 0; k < 3; ++k)           CHECK_NEAR(y[OBS_ANG_VEL + k], 0.1 * (k + 1), kTol);
    for (int j = 0; j < NUM_JOINTS; ++j)  CHECK_NEAR(y[OBS_Q + j],  100.0 + j, kTol);
    for (int j = 0; j < NUM_JOINTS; ++j)  CHECK_NEAR(y[OBS_DQ + j], 200.0 + j, kTol);
    for (int m = 0; m < NUM_MUSCLES; ++m) CHECK_NEAR(y[OBS_ACT + m], 0.01 * m, kTol);
}

int main()
{
    test_upright_at_rest();
    test_yaw_90();
    test_pitch_ignored_by_goal_offset();
    test_layout();
    return check::exit_code();
}
