// NpyWriter (common/npy_writer.h): header and file size after every row, for
// 2-D and 3-D files, an empty file, and an unopenable path.
//
//   test_npy_writer             files go to a temp directory, removed afterwards
//   test_npy_writer <out_dir>   files are kept for np.load
//                               (learning/tests/test_npy_writer.py runs this)

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <unistd.h>

#include "check.h"
#include "../common/npy_writer.h"

namespace fs = std::filesystem;

// Values learning/tests/test_npy_writer.py expects, exact in float32.
static double obs_value(int r, int c)         { return r * 10.0 + c * 0.5; }
static double plan_value(int r, int i, int j) { return r * 100.0 + i * 10.0 + j * 0.25; }

// The file's header text and total header size (preamble + text), or size 0
// if the magic string or version is wrong.
struct NpyHeader {
    size_t      size = 0;
    std::string text;
};

static NpyHeader read_header(const fs::path& path)
{
    std::ifstream in(path, std::ios::binary);
    char preamble[10] = {};
    in.read(preamble, sizeof(preamble));

    NpyHeader h;
    if (std::memcmp(preamble, "\x93NUMPY\x01\x00", 8) != 0) return h;
    const size_t len = static_cast<uint8_t>(preamble[8]) | (static_cast<uint8_t>(preamble[9]) << 8);
    h.text.resize(len);
    in.read(h.text.data(), len);
    h.size = sizeof(preamble) + len;
    return h;
}

// path holds a 64-byte-aligned header declaring `shape`, then exactly `rows` rows.
static void check_file(const fs::path& path, const std::string& shape, long long rows, size_t row_bytes)
{
    const NpyHeader h = read_header(path);
    CHECK(h.size > 0);
    CHECK(h.size % 64 == 0);
    CHECK(h.text.find("'descr': '<f4'") != std::string::npos);
    CHECK(h.text.find("'shape': " + shape) != std::string::npos);
    CHECK(!h.text.empty() && h.text.back() == '\n');
    CHECK(fs::file_size(path) == h.size + rows * row_bytes);
}

static void test_2d(const fs::path& dir)
{
    const fs::path path = dir / "obs.npy";
    constexpr int kCols = 3;
    constexpr size_t kRowBytes = kCols * sizeof(float);
    {
        NpyWriter w(path.string(), {kCols});
        check_file(path, "(0, 3)", 0, kRowBytes);

        for (int r = 0; r < 4; ++r) {
            double row[kCols];
            for (int c = 0; c < kCols; ++c) row[c] = obs_value(r, c);
            w.append(row);
            check_file(path, "(" + std::to_string(r + 1) + ", 3)", r + 1, kRowBytes);
        }
        CHECK(w.rows() == 4);
    }
    check_file(path, "(4, 3)", 4, kRowBytes);   // unchanged by closing
}

static void test_3d(const fs::path& dir)
{
    const fs::path path = dir / "plan.npy";
    {
        NpyWriter w(path.string(), {2, 3});
        CHECK(w.row_size() == 6);

        for (int r = 0; r < 3; ++r) {
            double row[6];
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 3; ++j) row[i * 3 + j] = plan_value(r, i, j);
            w.append(row);
        }
    }
    check_file(path, "(3, 2, 3)", 3, 6 * sizeof(float));
}

// No rows, e.g. a run that falls before logging starts: still a valid (0, 3) file.
static void test_empty(const fs::path& dir)
{
    const fs::path path = dir / "empty.npy";
    { NpyWriter w(path.string(), {3}); }
    check_file(path, "(0, 3)", 0, 3 * sizeof(float));
}

static void test_open_failure(const fs::path& dir)
{
    bool threw = false;
    try {
        NpyWriter w((dir / "missing_dir" / "x.npy").string(), {3});
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

int main(int argc, char** argv)
{
    const bool keep = argc > 1;
    const fs::path dir = keep ? fs::path(argv[1])
                              : fs::temp_directory_path() / ("test_npy_writer_" + std::to_string(getpid()));
    fs::create_directories(dir);

    test_2d(dir);
    test_3d(dir);
    test_empty(dir);
    test_open_failure(dir);

    if (!keep) fs::remove_all(dir);
    return check::exit_code();
}
