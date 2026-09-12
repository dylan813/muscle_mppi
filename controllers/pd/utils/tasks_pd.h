#pragma once

#include <string>
#include <vector>

#include "../../common/paths.h"
#include "../../common/types.h"

// Default task file for every PD-variant binary (absolute, see common/paths.h).
inline const std::string kDefaultTasksPdYaml = repo_path("controllers/pd/utils/tasks_pd.yaml");

// Direct joint-space PD gains — replaces MuscleParams in the muscle-actuated
// variant (see muscle/control/muscle.h / muscle/utils/tasks.h). tau[j] = kp[j]*(q_des[j]-q[j])
// - kd[j]*dq[j], matching RTWholeBodyMPPI's actuator-level PD law.
struct PDParams {
    double kp[NUM_JOINTS] = {};
    double kd[NUM_JOINTS] = {};

    // MuJoCo joint damping applied to the sim (model_->dof_damping), matching
    // RTWholeBodyMPPI's go1_mppi.xml default class (2.0, overridden to 1.0 for
    // the abduction/hip-roll joints) — go2.xml's own default (0.1 for every
    // joint) is far lighter, since it was tuned for the muscle-actuated
    // variant's kd_sim instead. Left at go2.xml's default this under-damps
    // the stiff kp=55 PD law and shows up as visible bouncing/jitter.
    double joint_damping[NUM_JOINTS] = {};
};

struct TaskConfig {
    std::string  model_path;   // absolute; a relative YAML value is resolved against the repo root

    // Ordered waypoint sequence for locomotion tasks (MPPILocomotionPD).
    std::vector<TaskPhase> phases;

    double       nominal_pose[NUM_JOINTS] = {};
    PDParams     pd;
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

    // Per-joint noise sigma (radians) applied to the sampled desired joint
    // position. Used by BaseMPPIPD::sample_actions().
    double noise_sigma_act[NUM_JOINTS]   = {};
};

TaskConfig load_task(const std::string& task_name,
                     const std::string& yaml_path = kDefaultTasksPdYaml);
