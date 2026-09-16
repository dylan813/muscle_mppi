#pragma once

#include <string>
#include <vector>

#include "../../common/task_config.h"

// Default task file for every muscle-variant binary (absolute, see common/task_config.h).
inline const std::string kDefaultTasksYaml = repo_path("controllers/muscle/utils/tasks.yaml");

static constexpr int NUM_MUSCLES = 2 * NUM_JOINTS;  // antagonistic pair per joint

struct MuscleParams {
    double act_bandwidth = 100.0;              // activation filter bandwidth (Hz)
    double peak_force[NUM_JOINTS] = {};        // peak isometric force (N), scales FL*FV output
    double lce_min[NUM_JOINTS]    = {};        // min fiber length (normalized by l_opt)
    double lce_max[NUM_JOINTS]    = {};        // max fiber length (normalized by l_opt)
    double phi_min[NUM_JOINTS]    = {};        // joint angle (rad) mapping to lce_min for agonist
    double phi_max[NUM_JOINTS]    = {};        // joint angle (rad) mapping to lce_max for agonist
    double vmax[NUM_JOINTS]       = {};        // max contraction velocity (fiber lengths / s)
    double FVmax[NUM_JOINTS]      = {};        // eccentric force amplification (>1)
    double pFLmax[NUM_JOINTS]     = {};        // passive force at max extension
    double kd_sim[NUM_JOINTS]     = {};        // MuJoCo joint damping (applied to sim dofs)

    // Co-contraction level in [0, 1]: where on the torque-balance line (all
    // activation pairs producing the required joint torque) activations are
    // chosen. 0.5 = minimum co-contraction, 1.0 = maximum (stiffest). Used for
    // the warm start (holding the settled standing pose), for inverting rollout
    // states in the gait-tracking cost, and for generating the activation gaits
    // (muscle/control/activation_gait.h), which regenerate when it changes.
    double stiffness = 0.75;

    // Ablation switches, set from the task's optional `ablation:` block (all
    // true = the full model; see run_ablation_walk.sh). The Hill switches apply to both
    // hill_compute_torques and hill_invert_torque, so the gait reference, the
    // gait-tracking cost and the warm start use the same muscle model as the
    // rollouts. All Hill switches off leaves a linear actuator:
    // tau = r * peak_force * (a2 - a1).
    bool activation_dynamics = true;   // false: activation = command (no first-order filter)
    bool use_fl              = true;   // false: active force-length = 1
    bool use_fv              = true;   // false: force-velocity = 1
    bool use_passive         = true;   // false: no passive parallel force
};

// Shared task fields (model_path, phases, sampling, …) come from TaskConfigBase
// (common/task_config.h); this adds the muscle variant's own parameters.
struct TaskConfig : TaskConfigBase {
    MuscleParams muscle;
};

TaskConfig load_task(const std::string& task_name,
                     const std::string& yaml_path = kDefaultTasksYaml);
