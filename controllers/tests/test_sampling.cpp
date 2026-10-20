// BaseMPPI's sampling around the plan (muscle/control/base_mppi.h), on the
// walk task (horizon 40, knots at steps 0, 13, 26, 39). "cubic": each sample is
// the cubic through the plan at the knots plus noise there (RTWholeBodyMPPI's
// perturb_action). "normal": the plan plus noise at every step. Every sample
// is clamped to [0, 1].

#include <algorithm>
#include <cmath>
#include <vector>

#include "check.h"
#include "../muscle/control/base_mppi.h"

// BaseMPPI with its sampling state open to the checks.
class Sampler : public BaseMPPI {
public:
    explicit Sampler(const TaskConfig& task) : BaseMPPI(task) {}

    using BaseMPPI::sample_actions;
    std::vector<double>&       plan()          { return trajectory_; }
    const std::vector<double>& actions() const { return actions_; }
    const std::vector<double>& knots()   const { return knot_x_; }

private:
    double rollout(int, const RobotState&) override { return 0.0; }
};

// A plan that isn't a cubic: a 17-step-period sine, phase per muscle, in [0.35, 0.65].
static void fill_wavy(Sampler& sampler, int horizon)
{
    for (int t = 0; t < horizon; ++t)
        for (int m = 0; m < NUM_MUSCLES; ++m)
            sampler.plan()[t * NUM_MUSCLES + m] = 0.5 + 0.15 * std::sin(2.0 * M_PI * t / 17.0 + m);
}

// The cubic through (xk[i], yk[i]), i = 0..3, at x. With 4 knots the
// not-a-knot spline is this one cubic.
static double cubic_through(const std::vector<double>& xk, const double yk[4], double x)
{
    double y = 0.0;
    for (int i = 0; i < 4; ++i) {
        double l = 1.0;
        for (int j = 0; j < 4; ++j)
            if (j != i) l *= (x - xk[j]) / (xk[i] - xk[j]);
        y += yk[i] * l;
    }
    return y;
}

// No noise: every sample is the cubic through the plan's values at the knots,
// not the plan itself (which splining only the noise would give).
static void test_cubic_without_noise(TaskConfig task)
{
    std::fill(task.noise_sigma_act, task.noise_sigma_act + NUM_JOINTS, 0.0);
    Sampler sampler(task);
    CHECK((sampler.knots() == std::vector<double>{0, 13, 26, 39}));

    const int H = task.horizon;
    fill_wavy(sampler, H);
    sampler.sample_actions();

    double worst = 0.0, plan_vs_cubic = 0.0;
    for (int m = 0; m < NUM_MUSCLES; ++m) {
        double yk[4];
        for (int k = 0; k < 4; ++k) yk[k] = sampler.plan()[static_cast<int>(sampler.knots()[k]) * NUM_MUSCLES + m];

        for (int t = 0; t < H; ++t) {
            const double cubic = cubic_through(sampler.knots(), yk, t);
            plan_vs_cubic = std::max(plan_vs_cubic, std::abs(sampler.plan()[t * NUM_MUSCLES + m] - cubic));
            for (int s = 0; s < task.n_samples; ++s)
                worst = std::max(worst, std::abs(sampler.actions()[(s * H + t) * NUM_MUSCLES + m] - cubic));
        }
    }
    CHECK_NEAR(worst, 0.0, 1e-9);
    CHECK(plan_vs_cubic > 0.05);   // so the check above tells the two forms apart
}

// The walk task's noise: at the knots, sample - plan has mean 0 and the
// joint's sigma, drawn independently for the two muscles of each pair.
static void test_cubic_noise_at_knots(const TaskConfig& task)
{
    Sampler sampler(task);
    const int H = task.horizon;
    fill_wavy(sampler, H);

    // 100 calls × n_samples × 4 knots = 12000 draws per muscle. One standard
    // error of each estimate is then under 1% of sigma (0.01 for the
    // correlation), so the 5% tolerances are over 5 standard errors.
    std::vector<double> sum(NUM_MUSCLES), sum_sq(NUM_MUSCLES), sum_pair(NUM_JOINTS);
    long n = 0;
    for (int call = 0; call < 100; ++call) {
        sampler.sample_actions();
        for (int s = 0; s < task.n_samples; ++s)
            for (double knot : sampler.knots()) {
                const int i = static_cast<int>(knot) * NUM_MUSCLES;
                double d[NUM_MUSCLES];
                for (int m = 0; m < NUM_MUSCLES; ++m) {
                    d[m] = sampler.actions()[s * H * NUM_MUSCLES + i + m] - sampler.plan()[i + m];
                    sum[m] += d[m];
                    sum_sq[m] += d[m] * d[m];
                }
                for (int j = 0; j < NUM_JOINTS; ++j) sum_pair[j] += d[2 * j] * d[2 * j + 1];
                ++n;
            }
    }

    for (int m = 0; m < NUM_MUSCLES; ++m) {
        const double sigma = task.noise_sigma_act[m / 2];
        const double mean  = sum[m] / n;
        CHECK_NEAR(mean, 0.0, 0.05 * sigma);
        CHECK_NEAR(std::sqrt(sum_sq[m] / n - mean * mean), sigma, 0.05 * sigma);
    }
    for (int j = 0; j < NUM_JOINTS; ++j) {
        const double sigma = task.noise_sigma_act[j];
        CHECK_NEAR(sum_pair[j] / n / (sigma * sigma), 0.0, 0.05);   // correlation of the pair
    }
}

// Every sample is clamped to [0, 1], also around a plan at the edges.
static void test_samples_stay_in_range(const TaskConfig& task)
{
    for (const char* type : {"cubic", "normal"}) {
        TaskConfig t = task;
        t.sample_type = type;
        Sampler sampler(t);
        for (int i = 0; i < t.horizon * NUM_MUSCLES; ++i) sampler.plan()[i] = (i % 2 == 0) ? 0.98 : 0.02;
        sampler.sample_actions();

        const auto [lo, hi] = std::minmax_element(sampler.actions().begin(), sampler.actions().end());
        CHECK_NEAR(*lo, 0.0, 0.0);   // reached, not passed
        CHECK_NEAR(*hi, 1.0, 0.0);
    }
}

// "normal" without noise: every sample is the plan itself, step by step.
static void test_normal_without_noise(TaskConfig task)
{
    task.sample_type = "normal";
    std::fill(task.noise_sigma_act, task.noise_sigma_act + NUM_JOINTS, 0.0);
    Sampler sampler(task);
    const int stride = task.horizon * NUM_MUSCLES;
    fill_wavy(sampler, task.horizon);
    sampler.sample_actions();

    double worst = 0.0;
    for (int s = 0; s < task.n_samples; ++s)
        for (int i = 0; i < stride; ++i)
            worst = std::max(worst, std::abs(sampler.actions()[s * stride + i] - sampler.plan()[i]));
    CHECK_NEAR(worst, 0.0, 0.0);
}

int main()
{
    const TaskConfig walk = load_task("walk");

    test_cubic_without_noise(walk);
    test_cubic_noise_at_knots(walk);
    test_samples_stay_in_range(walk);
    test_normal_without_noise(walk);

    return check::exit_code();
}
