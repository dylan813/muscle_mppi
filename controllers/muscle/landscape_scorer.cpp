// MPPI's cost at one saved solve, for the landscape toolbox
// (Toolbox-Cost-Analysis): the cost of any knot values, each turned into a
// plan as cubic sampling turns its knots into a sample, and rolled out from
// what one update() of an `mppi_sim --contexts` run planned from.
//
//   ./landscape_scorer <task> <yaml> <contexts prefix> <row> <threads>
//
// <task> and <yaml> must be what the mppi_sim run used: the cost weights,
// muscle parameters and gait the contexts were saved under. <row> is the CSV
// row whose solve to score at (control/context_files.h); <threads> the OpenMP
// threads the rollouts run on.
//
// Speaks binary over stdin/stdout, in this machine's byte order. Everything
// the controller prints goes to stderr instead.
//   at startup  int32 K = n_knots × NUM_MUSCLES; K float64, the knot values of
//               the plan the row's update() chose (knots_from_plan()); float64,
//               MPPI's cost of that plan itself, which mppi_sim logged as the
//               row's plan_* columns
//   request     int32 n, then n × K float64 knot values, each [k * NUM_MUSCLES + m]
//   reply       n float64 costs (MPPILocomotion::score_plans() of plan_from_knots())
// A request with n = 0, or the end of stdin, ends it.

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>
#include <unistd.h>

#include "control/context_files.h"
#include "control/mppi_locomotion.h"

// Read exactly n values of T from stdin; false at the end of stdin.
template <class T>
static bool read_values(T* values, size_t n)
{
    return std::fread(values, sizeof(T), n, stdin) == n;
}

template <class T>
static void write_values(std::FILE* out, const T* values, size_t n)
{
    if (std::fwrite(values, sizeof(T), n, out) != n) throw std::runtime_error("cannot write to stdout");
}

int main(int argc, char** argv)
{
    if (argc != 6) {
        std::fprintf(stderr, "usage: %s <task> <yaml> <contexts prefix> <row> <threads>\n", argv[0]);
        return 2;
    }
    const std::string task_name = argv[1], yaml = argv[2], prefix = argv[3];
    const int row = std::stoi(argv[4]), threads = std::stoi(argv[5]);

    // The replies keep the real stdout; the controller's own prints go to stderr.
    std::FILE* out = fdopen(dup(STDOUT_FILENO), "wb");
    dup2(STDERR_FILENO, STDOUT_FILENO);

    try {
        MPPILocomotion mppi(task_name, yaml);
        omp_set_num_threads(threads);
        const TaskConfig& task = mppi.task_ref();
        const int32_t n_values = task.n_knots * NUM_MUSCLES;
        const int     stride   = task.horizon * NUM_MUSCLES;

        std::vector<double> chosen;
        const RolloutContext context = read_context(prefix, row, task.horizon, chosen);

        std::vector<double> chosen_knots(n_values);
        mppi.knots_from_plan(chosen.data(), chosen_knots.data());
        double chosen_cost = 0.0;
        mppi.score_plans(context, chosen.data(), 1, &chosen_cost);

        write_values(out, &n_values, 1);
        write_values(out, chosen_knots.data(), chosen_knots.size());
        write_values(out, &chosen_cost, 1);
        std::fflush(out);

        std::vector<double> knots, plans, costs;
        int32_t n = 0;
        while (read_values(&n, 1) && n > 0) {
            knots.resize(static_cast<size_t>(n) * n_values);
            if (!read_values(knots.data(), knots.size())) throw std::runtime_error("stdin ended inside a request");

            plans.resize(static_cast<size_t>(n) * stride);
            for (int i = 0; i < n; ++i) mppi.plan_from_knots(&knots[i * n_values], &plans[i * stride]);

            costs.resize(n);
            mppi.score_plans(context, plans.data(), n, costs.data());
            write_values(out, costs.data(), costs.size());
            std::fflush(out);
        }
        if (n < 0) throw std::runtime_error("negative request size " + std::to_string(n));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "landscape_scorer: %s\n", e.what());
        return 1;
    }
    return 0;
}
