// Scaffolding test: checks the two MPPI update steps in common/control_utils.h
// that every later MPPI test builds on.

#include <vector>

#include "check.h"
#include "../common/control_utils.h"

// shift_trajectory moves every step one forward and holds the final step.
static void test_shift_trajectory()
{
    // 3 steps × 2 actions, step t = {10t, 10t + 1}.
    std::vector<double> traj = {0, 1,  10, 11,  20, 21};
    shift_trajectory(traj, /*horizon=*/3, /*action_dim=*/2);

    const std::vector<double> expected = {10, 11,  20, 21,  20, 21};
    CHECK(traj == expected);
}

// softmin_weights sums to 1, ranks lower cost higher, and returns the minimum.
static void test_softmin_weights()
{
    std::vector<double> weights;
    const double cmin = softmin_weights({3.0, 1.0, 2.0}, /*lambda=*/0.1, weights);

    CHECK_NEAR(cmin, 1.0, 0.0);
    CHECK_NEAR(weights[0] + weights[1] + weights[2], 1.0, 1e-12);
    CHECK(weights[1] > weights[2] && weights[2] > weights[0]);

    // Equal costs: equal weights.
    softmin_weights({5.0, 5.0}, 0.1, weights);
    CHECK_NEAR(weights[0], 0.5, 1e-12);
    CHECK_NEAR(weights[1], 0.5, 1e-12);
}

int main()
{
    test_shift_trajectory();
    test_softmin_weights();
    return check::exit_code();
}
