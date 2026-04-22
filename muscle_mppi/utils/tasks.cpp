#include "tasks.h"
#include <stdexcept>
#include <cstring>

static const double STAND_POSE[NUM_JOINTS] = {
    0.0,  0.67, -1.3,   // FR
    0.0,  0.67, -1.3,   // FL
    0.0,  0.67, -1.3,   // RR
    0.0,  0.67, -1.3,   // RL
    0.0, 0.0, 0.0, 0.0
};

TaskConfig get_task(const std::string& name) {
    TaskConfig cfg;  // struct defaults applied

    if (name == "stand") {
        std::memcpy(cfg.nominal_pose, STAND_POSE, sizeof(STAND_POSE));
        cfg.cost.vel_cmd    = 0.0;
        cfg.cost.vel_des[0] = 0.0;
        cfg.cost.vel_des[1] = 0.0;
        cfg.cost.vel_des[2] = 0.0;
        // Default weights already set for standing in CostWeights struct.
    }
    else if (name == "walk_forward") {
        std::memcpy(cfg.nominal_pose, STAND_POSE, sizeof(STAND_POSE));
        cfg.cost.height        = 1e2;
        cfg.cost.orientation   = 10.0;
        cfg.cost.posture       = 0.0;   // walking naturally varies joint angles
        cfg.cost.contact_vel   = 0.5;
        cfg.cost.contact_force = 5e-2;
        cfg.cost.terminal      = 2.5e3;
        cfg.cost.vel_cmd       = 1.0;
        cfg.cost.vel_des[0]    = 0.5;  // 0.5 m/s forward
        cfg.cost.vel_des[1]    = 0.0;
        cfg.cost.vel_des[2]    = 0.0;
        // ref_free_mppi uses n_samples=50 with compressed Hermite splines (few sim steps
        // per rollout). Muscle_mppi does full 450-step rollouts, so keep samples low.
        cfg.n_samples          = 16;
    }
    else {
        throw std::runtime_error("Unknown task: " + name);
    }

    return cfg;
}
