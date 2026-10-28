// read_npy (common/npy_reader.h): float64 files NpyWriter writes read back
// exactly, with their shape; a float32 file and a missing file throw.

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include "check.h"
#include "../common/npy_reader.h"
#include "../common/npy_writer.h"

namespace fs = std::filesystem;

// Not exact in float32, so a float32 round trip would show.
static double value(int r, int i) { return r + i / 3.0; }

static void test_round_trip(const fs::path& dir)
{
    const fs::path path = dir / "plan.npy";
    {
        NpyWriter w(path.string(), {2, 3}, NpyDtype::Float64);
        for (int r = 0; r < 4; ++r) {
            double row[6];
            for (int i = 0; i < 6; ++i) row[i] = value(r, i);
            w.append(row);
        }
    }

    const NpyArray a = read_npy(path.string());
    CHECK((a.shape == std::vector<int>{4, 2, 3}));
    CHECK(a.values.size() == 24);
    for (int r = 0; r < 4; ++r)
        for (int i = 0; i < 6; ++i) CHECK_NEAR(a.values[r * 6 + i], value(r, i), 0.0);
}

static void test_rows_of_one_dimension(const fs::path& dir)
{
    const fs::path path = dir / "state.npy";
    {
        NpyWriter w(path.string(), {5}, NpyDtype::Float64);
        for (int r = 0; r < 3; ++r) {
            double row[5];
            for (int i = 0; i < 5; ++i) row[i] = value(r, i);
            w.append(row);
        }
    }

    const NpyArray a = read_npy(path.string());
    CHECK((a.shape == std::vector<int>{3, 5}));
    CHECK_NEAR(a.values[2 * 5 + 4], value(2, 4), 0.0);
}

static bool throws(const fs::path& path)
{
    try {
        read_npy(path.string());
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

static void test_rejects(const fs::path& dir)
{
    const fs::path float32 = dir / "float32.npy";
    {
        NpyWriter w(float32.string(), {3});
        const double row[3] = {1.0, 2.0, 3.0};
        w.append(row);
    }
    CHECK(throws(float32));
    CHECK(throws(dir / "missing.npy"));
}

int main()
{
    const fs::path dir = fs::temp_directory_path() / ("test_npy_reader_" + std::to_string(getpid()));
    fs::create_directories(dir);

    test_round_trip(dir);
    test_rows_of_one_dimension(dir);
    test_rejects(dir);

    fs::remove_all(dir);
    return check::exit_code();
}
