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

// Shared task fields (model_path, phases, sampling, …) come from TaskConfigBase
// (common/types.h); this adds the PD variant's own parameters.
struct TaskConfig : TaskConfigBase {
    PDParams     pd;
};

TaskConfig load_task(const std::string& task_name,
                     const std::string& yaml_path = kDefaultTasksPdYaml);
