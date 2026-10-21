// FlowPolicy (muscle/control/flow_policy.h) on a file learning/ exported, with
// the walk task's plan sizes.
//
//   test_flow_policy <policy.onnx> <cpu|cuda>            load only; exits 1 with the error if refused
//   test_flow_policy <policy.onnx> <cpu|cuda> <out_dir>  also samples each case below and writes
//                                                        its inputs and plans, <case>_<name>.npy,
//                                                        for np.load
//
// learning/tests/test_flow_policy.py runs this on files it exports, and checks
// the plans against the Python policy.

#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "../common/npy_writer.h"
#include "../muscle/control/flow_policy.h"

namespace fs = std::filesystem;

// The walk task: horizon 40, 4 knots (BaseMPPI's knot_x_).
static constexpr int kHorizon = 40;
static const std::vector<double> kKnotSteps = {0, 13, 26, 39};

// Cases sampled; test_flow_policy.py's CASES mirrors these names. 30 is MPPI's sample count.
struct Case {
    const char* name;
    int         n;
    double      warm_start;
};
static const Case kCases[] = {
    {"n1_warm0",  1,  0.0}, {"n1_warm0.7",  1,  0.7},
    {"n30_warm0", 30, 0.0}, {"n30_warm0.7", 30, 0.7},
};

// Write values as `rows` rows of row_shape to path.
static void write_npy(const fs::path& path, std::vector<int> row_shape, int rows, const std::vector<double>& values)
{
    NpyWriter w(path.string(), std::move(row_shape));
    for (int r = 0; r < rows; ++r) w.append(values.data() + r * w.row_size());
}

static void sample_case(FlowPolicy& policy, const Case& c, const fs::path& dir)
{
    const int K = static_cast<int>(kKnotSteps.size());
    std::mt19937 rng(c.n);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    // Inputs as test_export.py draws them: y ~ N(1, 2²), previous plans in [0, 1].
    std::vector<double> y(OBS_DIM), prev(c.n * kHorizon * NUM_MUSCLES), noise(c.n * K * NUM_MUSCLES);
    for (double& v : y)     v = 2.0 * normal(rng) + 1.0;
    for (double& v : prev)  v = uniform(rng);
    for (double& v : noise) v = normal(rng);

    std::vector<double> plans(prev.size());
    policy.sample(y.data(), c.n, prev.data(), noise.data(), c.warm_start, plans.data());
    for (double v : plans) CHECK(std::isfinite(v));

    const std::string prefix = (dir / c.name).string();
    write_npy(prefix + "_y.npy",          {OBS_DIM},                1,   y);
    write_npy(prefix + "_prev.npy",       {kHorizon, NUM_MUSCLES},  c.n, prev);
    write_npy(prefix + "_noise.npy",      {K, NUM_MUSCLES},         c.n, noise);
    write_npy(prefix + "_warm_start.npy", {},                       1,   {c.warm_start});
    write_npy(prefix + "_plan.npy",       {kHorizon, NUM_MUSCLES},  c.n, plans);
}

int main(int argc, char** argv)
{
    const std::string device = argc > 2 ? argv[2] : "";
    if (device != "cpu" && device != "cuda") {
        std::fprintf(stderr, "usage: %s <policy.onnx> <cpu|cuda> [out_dir]\n", argv[0]);
        return 2;
    }

    try {
        FlowPolicy policy(argv[1], kHorizon, kKnotSteps,
                          device == "cuda" ? PolicyDevice::Cuda : PolicyDevice::Cpu);
        if (argc < 4) return 0;

        fs::create_directories(argv[3]);
        for (const Case& c : kCases) sample_case(policy, c, argv[3]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return check::exit_code();
}
