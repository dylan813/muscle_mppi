#pragma once

// Minimal assertions for the unit tests in controllers/tests/, run by CTest
// (build, then `ctest` in controllers/build). A failed check prints where it
// failed and what it compared, and the test keeps going so one run reports
// every failure. main() returns check::exit_code(), which CTest reads.

#include <cmath>
#include <cstdio>

namespace check {

inline int failures = 0;

inline int exit_code()
{
    if (failures) std::fprintf(stderr, "%d check(s) failed\n", failures);
    return failures ? 1 : 0;
}

}  // namespace check

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            ++check::failures;                                                       \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); \
        }                                                                            \
    } while (0)

// |a - b| <= tol, for doubles.
#define CHECK_NEAR(a, b, tol)                                                        \
    do {                                                                             \
        const double check_a_ = (a), check_b_ = (b);                                 \
        if (!(std::abs(check_a_ - check_b_) <= (tol))) {                             \
            ++check::failures;                                                       \
            std::fprintf(stderr, "%s:%d: CHECK_NEAR(%s, %s, %s) failed: %.17g vs %.17g\n", \
                         __FILE__, __LINE__, #a, #b, #tol, check_a_, check_b_);      \
        }                                                                            \
    } while (0)
