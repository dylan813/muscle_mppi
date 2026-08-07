#pragma once

#include <string>
#include <vector>

// These are overridable at compile time via -DNUM_JOINTS=N -DJOINT_OFFSET=N.
// Defaults: full quadruped (12 joints, offset 0).
#ifndef NUM_JOINTS
static constexpr int NUM_JOINTS   = 12;   // 4 legs × 3 joints (FR, FL, RR, RL)
static constexpr int JOINT_OFFSET = 0;    // actuators start at index 0
#endif

// Direct joint-space PD gains — replaces MuscleParams in the muscle-actuated
// variant (see control/muscle.h / utils/tasks.h). tau[j] = kp[j]*(q_des[j]-q[j])
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

// One waypoint in a task's phase sequence, matching RTWholeBodyMPPI's per-phase
// goal_pos/cmd_vel/desired_gait/goal_thresh/waiting_times arrays. MPPILocomotionPD
// advances phase_index_ once the robot has stayed within goal_thresh of the
// current phase's goal_pos for waiting_time consecutive in-threshold ticks.
struct TaskPhase {
    double goal_pos[3]  = {};
    double cmd_vel[2]   = {};  // [vx, vy] body-frame velocity command

    // Categorical gait name: "in_place" | "walk" | "walk_fast" | "trot".
    // Resolved to a gait TSV path by MPPILocomotionPD (see mppi_locomotion_pd.cpp).
    std::string desired_gait;

    // Optional escape hatch: an explicit gait TSV path, overriding desired_gait
    // when non-empty.
    std::string gait_path;

    double goal_thresh  = 0.2;
    int    waiting_time = 0;   // dwell ticks required within goal_thresh

    // Optional per-phase override of TaskConfig::noise_sigma_act, applied while
    // this phase is active and reverted to the task-level baseline on the next
    // phase that doesn't set one. has_noise_sigma_act distinguishes "not set"
    // from a legitimate all-zero override.
    double noise_sigma_act[NUM_JOINTS] = {};
    bool   has_noise_sigma_act = false;
};

struct TaskConfig {
    std::string  model_path;

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

    // RNG seed for noise sampling. RTWholeBodyMPPI seeds deterministically
    // (`seed: 42` in every mppi_gait_config_*.yml -> np.random.default_rng),
    // so runs are reproducible; matched here. Set seed: -1 to draw a
    // nondeterministic seed from std::random_device instead.
    int          seed         = 42;

    // Per-joint noise sigma (radians) applied to the sampled desired joint
    // position. Used by BaseMPPIPD::sample_actions().
    double noise_sigma_act[NUM_JOINTS]   = {};
};

struct MotionCommand {
    double vx          = 0.0;
    double vy          = 0.0;
    double goal_pos[3] = {};  // world-frame position target [x, y, z]
};

TaskConfig load_task(const std::string& task_name,
                     const std::string& yaml_path = "../pd_mppi/utils/tasks_pd.yaml");
