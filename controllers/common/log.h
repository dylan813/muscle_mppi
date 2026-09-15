#pragma once

// Run logging for the standalone sims (used by run_sim() in harness.h): output
// file naming, the shared CSV and qpos log rows, and rendering a logged rollout
// to a GIF. The --save trial copying lives separately in trial_log.h.
//
// Output only — nothing here feeds back into the controller or the simulation.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <string>
#include <system_error>

#include <mujoco/mujoco.h>

#include "control_utils.h"
#include "task_config.h"

// ============================================================================
// Output files
// ============================================================================

// Open an output log for writing. The output directory only holds gitignored
// CSVs, so a fresh checkout may not have it — create it rather than let
// ofstream fail silently. Prints an error and returns false on failure.
inline bool open_output(const std::string& path, std::ofstream& out)
{
    const std::filesystem::path dir = std::filesystem::path(path).parent_path();
    if (!dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
    }
    out.open(path);
    if (!out) { fprintf(stderr, "Cannot open %s for writing\n", path.c_str()); return false; }
    return true;
}

// An output CSV path with a guaranteed ".csv" extension (appended if missing),
// so every companion file name below can be derived from it reliably.
inline std::string with_csv_extension(const std::string& path)
{
    return std::filesystem::path(path).extension() == ".csv" ? path : path + ".csv";
}

// An output CSV path without its ".csv" — the base every run file is named from.
inline std::string output_stem(const std::string& csv_path)
{
    std::filesystem::path p(csv_path);
    if (p.extension() == ".csv") p.replace_extension();
    return p.string();
}

// Companion qpos log path: <name>.csv -> <name>_qpos.csv.
inline std::string qpos_path_for(const std::string& csv_path)
{
    return output_stem(csv_path) + "_qpos.csv";
}

// Rollout GIF path for a CSV: <name>.csv -> <name>.gif.
inline std::string gif_path_for(const std::string& csv_path)
{
    return output_stem(csv_path) + ".gif";
}

// ============================================================================
// CSV and qpos rows
// ============================================================================

// Columns every sim logs; each variant appends its own and then writes "\n".
inline void write_csv_header_base(std::ostream& csv)
{
    csv << "t,px,py,pz,vx,vy,vz,qw,roll_deg";
    for (int j = 0; j < NUM_JOINTS; ++j) csv << ",dq_j" << j;
}

// One row of the shared columns. `pos` is whichever position the variant's
// cost scores (whole-robot CoM for muscle, trunk origin for PD); velocity is
// the free joint's linear velocity in body-frame axes.
inline void write_csv_row_base(std::ostream& csv, double t, const double pos[3],
                               const mjData* d, const int qv[NUM_JOINTS])
{
    const double qw  = d->qpos[3];
    const double roll = 2.0 * std::acos(std::clamp(std::abs(qw), 0.0, 1.0))
                        * 180.0 / M_PI;

    // Rotate world-frame free-joint velocity into body frame (xmat
    // is the body->world rotation, so its transpose maps world->body).
    double xmat[9], v_body[3];
    mju_quat2Mat(xmat, d->qpos + 3);
    world_to_body(xmat, d->qvel, v_body);

    csv << t << ","
        << pos[0] << "," << pos[1] << "," << pos[2] << ","
        << v_body[0] << "," << v_body[1] << "," << v_body[2] << ","
        << qw << "," << roll;
    for (int j = 0; j < NUM_JOINTS; ++j) csv << "," << d->qvel[qv[j]];
}

// Full qpos row, used by analysis/render_gif.py.
inline void write_qpos_row(std::ostream& out, const mjModel* m, const mjData* d)
{
    for (int i = 0; i < m->nq; ++i)
        out << d->qpos[i] << (i < m->nq - 1 ? "," : "\n");
}

// ============================================================================
// Rollout GIF
// ============================================================================

// Single-quote a string for /bin/sh.
inline std::string shell_quote(const std::string& s)
{
    std::string out = "'";
    for (char c : s) out += (c == '\'') ? std::string("'\\''") : std::string(1, c);
    return out + "'";
}

// Render a qpos log to a GIF with analysis/render_gif.py, replacing any existing
// GIF at gif_path first so a failed render can't leave a stale one behind under
// the new run's name. Uses $MUSCLE_MPPI_PYTHON if set (a Python with mujoco,
// numpy, pyyaml and Pillow), else python3, and headless EGL rendering unless
// MUJOCO_GL is already set. Returns true if the GIF was written.
inline bool render_rollout_gif(const std::string& qpos_path, const std::string& gif_path,
                               const std::string& task_name, const std::string& yaml_path)
{
    std::error_code ec;
    std::filesystem::remove(gif_path, ec);

    const char* py = std::getenv("MUSCLE_MPPI_PYTHON");
    const std::string cmd =
        std::string("MUJOCO_GL=\"${MUJOCO_GL:-egl}\" ") + shell_quote(py && *py ? py : "python3")
        + " " + shell_quote(repo_path("analysis/render_gif.py"))
        + " " + shell_quote(qpos_path) + " " + shell_quote(gif_path)
        + " " + shell_quote(task_name) + " " + shell_quote(yaml_path);

    std::fflush(stdout);   // keep our console lines ahead of the script's output
    const int rc = std::system(cmd.c_str());
    return rc == 0 && std::filesystem::exists(gif_path, ec);
}
