// MPPI's cost at saved solves, for the landscape toolbox
// (Toolbox-Cost-Analysis): the cost of any knot values, each turned into a
// plan as cubic sampling turns its knots into a sample, and rolled out from
// what one update() of an `mppi_sim --contexts` run planned from.
//
//   ./landscape_scorer <task> <yaml> <contexts prefix> <threads>
//
// <task> and <yaml> must be what the mppi_sim run used: the cost weights,
// muscle parameters and gait the contexts were saved under
// (control/context_files.h); <threads> the OpenMP threads the rollouts run on.
// The controller is built, and the run's contexts loaded, once.
//
// Speaks binary over stdin/stdout, in this machine's byte order. Everything
// the controller prints goes to stderr instead. Each request starts with an
// int32 Request:
//   End     (also the end of stdin)
//   Select  int32 row: score at that CSV row's solve from now on. Reply: int32
//           K = n_knots × NUM_MUSCLES; K float64, the knot values of the plan
//           that row's update() chose (knots_from_plan()); float64, MPPI's cost
//           of that plan itself, which mppi_sim logged as the row's plan_* columns
//   Score   int32 n, then n × K float64 knot values, each [k * NUM_MUSCLES + m].
//           Reply: n float64 costs, MPPILocomotion::score_plans() of each
//           plan_from_knots(), at the selected row. Needs a Select first.

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>
#include <unistd.h>

#include "control/context_files.h"
#include "control/mppi_locomotion.h"

enum class Request : int32_t { End = 0, Select = 1, Score = 2 };

// Read exactly n values of T from stdin; false at the end of stdin.
template <class T>
static bool read_values(T* values, size_t n)
{
    return std::fread(values, sizeof(T), n, stdin) == n;
}

// As read_values(), where the end of stdin would cut a request short.
template <class T>
static void read_request_values(T* values, size_t n)
{
    if (!read_values(values, n)) throw std::runtime_error("stdin ended inside a request");
}

template <class T>
static void write_values(std::FILE* out, const T* values, size_t n)
{
    if (std::fwrite(values, sizeof(T), n, out) != n) throw std::runtime_error("cannot write to stdout");
}

int main(int argc, char** argv)
{
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s <task> <yaml> <contexts prefix> <threads>\n", argv[0]);
        return 2;
    }
    const std::string task_name = argv[1], yaml = argv[2], prefix = argv[3];
    const int threads = std::stoi(argv[4]);

    // The replies keep the real stdout; the controller's own prints go to stderr.
    std::FILE* out = fdopen(dup(STDOUT_FILENO), "wb");
    dup2(STDERR_FILENO, STDOUT_FILENO);

    try {
        MPPILocomotion mppi(task_name, yaml);
        omp_set_num_threads(threads);
        const TaskConfig& task = mppi.task_ref();
        const int32_t n_values = task.n_knots * NUM_MUSCLES;
        const int     stride   = task.horizon * NUM_MUSCLES;
        const ContextReader contexts(prefix, task.horizon);

        RolloutContext context;
        bool selected = false;
        std::vector<double> chosen, chosen_knots(n_values), knots, plans, costs;

        int32_t request = 0;
        while (read_values(&request, 1)) {
            switch (static_cast<Request>(request)) {
            case Request::End:
                return 0;

            case Request::Select: {
                int32_t row = 0;
                read_request_values(&row, 1);
                context  = contexts.read(row, chosen);
                selected = true;

                mppi.knots_from_plan(chosen.data(), chosen_knots.data());
                double chosen_cost = 0.0;
                mppi.score_plans(context, chosen.data(), 1, &chosen_cost);

                write_values(out, &n_values, 1);
                write_values(out, chosen_knots.data(), chosen_knots.size());
                write_values(out, &chosen_cost, 1);
                std::fflush(out);
                break;
            }

            case Request::Score: {
                if (!selected) throw std::runtime_error("Score before any Select");
                int32_t n = 0;
                read_request_values(&n, 1);
                if (n < 0) throw std::runtime_error("negative request size " + std::to_string(n));

                knots.resize(static_cast<size_t>(n) * n_values);
                read_request_values(knots.data(), knots.size());

                plans.resize(static_cast<size_t>(n) * stride);
                for (int i = 0; i < n; ++i)
                    mppi.plan_from_knots(&knots[static_cast<size_t>(i) * n_values], &plans[static_cast<size_t>(i) * stride]);

                costs.resize(n);
                mppi.score_plans(context, plans.data(), n, costs.data());
                write_values(out, costs.data(), costs.size());
                std::fflush(out);
                break;
            }

            default:
                throw std::runtime_error("unknown request " + std::to_string(request));
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "landscape_scorer: %s\n", e.what());
        return 1;
    }
    return 0;
}
