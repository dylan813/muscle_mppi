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

// With range_count, the range comes from the first range_count costs (MPPI's
// samples, before the policy's): a far worse later cost gets ~0 weight and
// leaves the others' weights as they were without it.
static void test_softmin_range_from_first_costs()
{
    std::vector<double> alone, with_worse;
    softmin_weights({3.0, 1.0, 2.0}, 0.1, alone);
    softmin_weights({3.0, 1.0, 2.0, 100.0}, 0.1, with_worse, /*range_count=*/3);
    for (int s = 0; s < 3; ++s) CHECK_NEAR(with_worse[s], alone[s], 1e-12);
    CHECK(with_worse[3] < 1e-100);

    // Scaled by the range of all four instead, the first three would flatten.
    std::vector<double> all;
    softmin_weights({3.0, 1.0, 2.0, 100.0}, 0.1, all);
    CHECK(all[0] > 0.25 && all[1] > 0.25 && all[2] > 0.25);   // each about 1/3

    // A later cost below the first ones' still wins.
    softmin_weights({3.0, 2.0, 4.0, 1.0}, 0.1, with_worse, 3);
    CHECK(*std::max_element(with_worse.begin(), with_worse.end()) == with_worse[3]);

    // First costs spanning no range (one sample): the range of all costs.
    softmin_weights({5.0, 1.0, 9.0}, 0.1, alone);
    softmin_weights({5.0, 1.0, 9.0}, 0.1, with_worse, 1);
    for (int s = 0; s < 3; ++s) CHECK_NEAR(with_worse[s], alone[s], 0.0);
}

int main()
{
    test_shift_trajectory();
    test_softmin_weights();
    test_softmin_range_from_first_costs();
    return check::exit_code();
}
