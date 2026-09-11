#pragma once

#include <filesystem>
#include <string>

// Absolute path of the repo root, compiled in by CMake (see
// controllers/CMakeLists.txt). Built-in defaults (task YAMLs, gait TSVs, sim
// output CSVs) and relative paths inside a task YAML (model_path, gait_path)
// all resolve against it, so the binaries behave the same from any working
// directory. Paths given on the command line are left to the caller — they
// stay relative to wherever the user ran the binary from.
#ifndef MUSCLE_MPPI_ROOT
#error "MUSCLE_MPPI_ROOT must be defined by the build (see controllers/CMakeLists.txt)"
#endif

// Repo-relative path -> absolute path. Absolute paths pass through unchanged,
// so a YAML (e.g. analysis/optimize/objective.py's per-candidate temp copy)
// can still point anywhere explicitly.
inline std::string repo_path(const std::string& rel)
{
    if (rel.empty() || std::filesystem::path(rel).is_absolute()) return rel;
    return (std::filesystem::path(MUSCLE_MPPI_ROOT) / rel).string();
}
