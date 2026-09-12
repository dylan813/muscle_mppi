#pragma once

#include <string>
#include <vector>

#include "../../common/paths.h"
#include "../../common/types.h"

// Default task file for every muscle-variant binary (absolute, see common/paths.h).
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
};

// Shared task fields (model_path, phases, sampling, …) come from TaskConfigBase
// (common/types.h); this adds the muscle variant's own parameters.
struct TaskConfig : TaskConfigBase {
    double       height_target = 0.0;
    MuscleParams muscle;

    // Normalized gravity torque at the nominal pose: tau_grav/(-r*peak_force) - (P1-P2).
    // Used by MPPILocomotion to seed the trajectory warm-start.
    double posture_bias[NUM_JOINTS] = {};
    double posture_FL1[NUM_JOINTS]  = {};
    double posture_FL2[NUM_JOINTS]  = {};

    // Co-contraction sampling parameters (per joint type: hip=0, thigh=1, calf=2).
};

TaskConfig load_task(const std::string& task_name,
                     const std::string& yaml_path = kDefaultTasksYaml);
