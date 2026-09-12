#pragma once

// Task description shared by the muscle-actuated (controllers/muscle/) and
// PD-actuated (controllers/pd/) variants: core types, repo-relative path
// resolution, and the YAML loading helpers each variant's load_task() builds on.

#include <filesystem>
#include <string>
#include <vector>

namespace YAML { class Node; }   // full definition only needed in .cpp files that parse YAML

// ============================================================================
// Repo paths
// ============================================================================

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

// ============================================================================
// Core types
// ============================================================================

// Variant-specific types (MuscleParams/NUM_MUSCLES, PDParams, each variant's
// TaskConfig) stay in that variant's utils/.

static constexpr int NUM_JOINTS = 12;   // 4 legs × 3 joints (FR, FL, RR, RL)

struct RobotState {
    double pos[3]          = {};
    double vel[3]          = {};
    double quat[4]         = {1,0,0,0};  // w, x, y, z
    double gyro[3]         = {};
    double q[NUM_JOINTS]   = {};
    double dq[NUM_JOINTS]  = {};
    bool   valid           = false;
};

// One waypoint in a task's phase sequence, matching RTWholeBodyMPPI's per-phase
// goal_pos/cmd_vel/desired_gait/goal_thresh/waiting_times arrays. Each variant's
// locomotion controller (MPPILocomotion / MPPILocomotionPD) advances
// phase_index_ once the robot has stayed within goal_thresh of the current
// phase's goal_pos for waiting_time consecutive in-threshold ticks.
struct TaskPhase {
    double goal_pos[3]  = {};
    double cmd_vel[2]   = {};  // [vx, vy] body-frame velocity command

    // Categorical gait name: "in_place" | "walk" | "walk_fast" | "trot".
    // Resolved to a gait TSV path by each variant's locomotion controller
    // (see mppi_locomotion.cpp / mppi_locomotion_pd.cpp).
    std::string desired_gait;

    // Optional escape hatch: an explicit gait TSV path, overriding desired_gait
    // when non-empty. Used by tooling (e.g. the CMA-ES muscle-parameter sweep in
    // analysis/optimize/objective.py) that regenerates a gait file per candidate
    // and needs mppi_sim to load that exact file rather than a canonical one.
    std::string gait_path;

    double goal_thresh  = 0.2;
    int    waiting_time = 0;   // dwell ticks required within goal_thresh

    // Optional per-phase override of TaskConfig::noise_sigma_act, applied while
    // this phase is active and reverted to the task-level baseline on the next
    // phase that doesn't set one. Mirrors RTWholeBodyMPPI's next_goal(), which
    // doubles thigh/calf exploration noise specifically during trot phases
    // regardless of the task config's declared baseline. has_noise_sigma_act
    // distinguishes "not set" from a legitimate all-zero override.
    double noise_sigma_act[NUM_JOINTS] = {};
    bool   has_noise_sigma_act = false;
};

struct MotionCommand {
    double vx          = 0.0;
    double vy          = 0.0;
    double goal_pos[3] = {};  // world-frame position target [x, y, z]
};

// Task fields common to both variants, parsed by load_task_base()
// (below). Each variant's TaskConfig derives from this and adds
// its own actuator parameters.
struct TaskConfigBase {
    std::string  model_path;   // absolute; a relative YAML value is resolved against the repo root

    // Ordered waypoint sequence for locomotion tasks.
    std::vector<TaskPhase> phases;

    double       nominal_pose[NUM_JOINTS] = {};
    int          n_samples    = 16;
    int          horizon      = 25;
    double       lambda       = 0.1;
    double       dt           = 0.002;

    // Seconds of MPPI control mppi_sim records after stand-up, before it stops
    // and writes the CSV/qpos log. Bump this for tasks whose phases need more
    // time to complete (e.g. a longer walk distance) than the 10s default covers.
    double       sim_duration = 10.0;

    // World-frame z of the ground under the robot's spawn point. mppi_sim's
    // stand-up placement assumes flat ground at z=0 and drops the robot so its
    // lowest foot lands there; set this to the actual terrain/platform height
    // at spawn (e.g. an elevated starting platform) so the robot lands on top
    // of it instead of spawning with its feet embedded in it.
    double       spawn_height_offset = 0.0;

    // OpenMP thread count for the parallel rollout loop. 0 (default) leaves the
    // OpenMP runtime default in place (typically all available cores).
    int          num_threads  = 0;

    // Noise sampling. "normal": iid Gaussian per timestep (default).
    // "cubic": draw n_knots iid Gaussians spread evenly across the horizon and
    // natural-cubic-spline interpolate between them, matching RTWholeBodyMPPI's
    // spline-parameterized sampling (smoother, lower-dimensional search).
    std::string  sample_type  = "normal";
    int          n_knots      = 4;

    // Per-joint exploration noise sigma, in the variant's action units.
    // Muscle: activation — both muscles of each antagonistic pair use this
    // joint's sigma, each with its own independent draw (BaseMPPI::sample_noise()).
    // PD: desired joint position in radians (BaseMPPIPD::sample_actions()).
    double noise_sigma_act[NUM_JOINTS]   = {};
};

// ============================================================================
// YAML task loading (implemented in task_config.cpp)
// ============================================================================

// Read a length-n YAML sequence into dst; throws naming `field` if the node is
// missing, not a sequence, or the wrong length.
void load_doubles(const YAML::Node& node, double* dst, int n, const std::string& field);

// Parse yaml_path and return its `task_name` entry; throws if the file can't
// be parsed or the task isn't in it.
YAML::Node load_task_node(const std::string& task_name, const std::string& yaml_path);

// Fill the fields common to both variants (see TaskConfigBase) from a task node.
void load_task_base(const YAML::Node& t, TaskConfigBase& cfg);
