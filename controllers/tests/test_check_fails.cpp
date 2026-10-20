// Checks check.h itself: these checks fail on purpose, so main() must return
// nonzero. CTest runs this with WILL_FAIL, so the test passes only if they do.

#include "check.h"

int main()
{
    CHECK(1 + 1 == 3);
    CHECK_NEAR(1.0, 2.0, 0.1);
    return check::exit_code();
}
