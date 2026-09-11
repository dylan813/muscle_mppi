// Trial log saving for the standalone sims (mppi_sim, pd_mppi_sim).
//
// By default both sims overwrite a single working CSV pair each run
// (analysis/data/mppi_sim/mppi_sim.csv + _qpos.csv, and the pd_mppi_sim
// equivalents under analysis/data/pd_mppi_sim/).
// That is fine for a one-off look, but destroys the previous run — no good for
// running the same task N times and analysing the spread.
//
// Passing `--save <name>` copies the run's logs, after the run finishes, into
//   <repo>/analysis/log/trials/<name>/trial_NNN/
// where NNN is the next free index under that name. The working CSVs are still
// written where they always were, so render_gif.py / plot_walk_leg.py keep
// working unchanged against the latest run.
//
// Nothing here touches the positional [task] [yaml] [output.csv] interface —
// analysis/optimize/objective.py drives mppi_sim that way and must keep working.

#pragma once

#include <cstdio>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace trial_log {

namespace fs = std::filesystem;

// Pull "--save <name>" / "--save=<name>" out of argv, leaving the positional
// arguments (argv[0] included, at index 0) so callers can index them exactly as
// they did before this flag existed. `name` is left empty when the flag is
// absent.
inline std::vector<std::string> parse_args(int argc, char** argv, std::string& name)
{
    std::vector<std::string> positional;
    name.clear();

    for (int i = 0; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i > 0 && arg.rfind("--save=", 0) == 0) {
            name = arg.substr(7);
        } else if (i > 0 && arg == "--save") {
            if (i + 1 < argc) {
                name = argv[++i];
            } else {
                fprintf(stderr, "--save needs a trial name; ignoring.\n");
            }
        } else {
            positional.push_back(arg);
        }
    }
    return positional;
}

// Repo root, found by walking up from the running executable until a directory
// holding analysis/log turns up. Resolved from /proc/self/exe rather than the
// CWD so --save behaves the same whether the binary is launched from build/ or
// from a CMA-ES worker's scratch directory.
inline std::string repo_root()
{
    std::error_code ec;
    fs::path exe = fs::canonical("/proc/self/exe", ec);
    if (ec) return {};

    for (fs::path dir = exe.parent_path(); !dir.empty(); dir = dir.parent_path()) {
        if (fs::is_directory(dir / "analysis" / "log", ec)) return dir.string();
        if (dir == dir.root_path()) break;
    }
    return {};
}

// Copy `files` into the next free trial directory under `name`. Returns the
// directory used, or "" if the trial could not be saved (the sim's own output
// is already on disk either way, so callers just warn and carry on).
inline std::string save(const std::string& name, const std::vector<std::string>& files)
{
    if (name.empty()) return {};
    if (name.find("..") != std::string::npos) {
        fprintf(stderr, "Trial name '%s' must not contain '..' — not saving.\n", name.c_str());
        return {};
    }

    const std::string root = repo_root();
    if (root.empty()) {
        fprintf(stderr, "Could not locate the repo root (no analysis/log above the "
                        "executable) — not saving trial.\n");
        return {};
    }

    std::error_code ec;
    const fs::path trial_root = fs::path(root) / "analysis" / "log" / "trials" / name;
    fs::create_directories(trial_root, ec);
    if (ec) {
        fprintf(stderr, "Could not create %s: %s\n", trial_root.c_str(), ec.message().c_str());
        return {};
    }

    // Claim the next index by creating the directory: create_directory returns
    // false for one that already exists, so concurrent trials under the same
    // name (parallel workers) each land on a distinct index instead of
    // overwriting one another.
    fs::path trial_dir;
    for (int i = 1; i <= 9999; ++i) {
        char buf[16];
        snprintf(buf, sizeof(buf), "trial_%03d", i);
        const fs::path candidate = trial_root / buf;
        if (fs::create_directory(candidate, ec) && !ec) {
            trial_dir = candidate;
            break;
        }
    }
    if (trial_dir.empty()) {
        fprintf(stderr, "No free trial index under %s — not saving trial.\n", trial_root.c_str());
        return {};
    }

    for (const std::string& file : files) {
        if (!fs::exists(file, ec)) continue;
        const fs::path dst = trial_dir / fs::path(file).filename();
        fs::copy_file(file, dst, fs::copy_options::overwrite_existing, ec);
        if (ec)
            fprintf(stderr, "Could not copy %s: %s\n", file.c_str(), ec.message().c_str());
    }
    return trial_dir.string();
}

}  // namespace trial_log
