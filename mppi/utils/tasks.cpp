#include "tasks.h"
#include <stdexcept>
#include <cstring>

// Standing pose for Go2W (rad)
// Order: FR, FL, RR, RL (hip, thigh, calf per leg), then wheels
static const double STAND_POSE[NUM_JOINTS] = {
     0.00572,  0.6088, -1.2176,   // FR
    -0.00572,  0.6088, -1.2176,   // FL
     0.00572,  0.6088, -1.2176,   // RR
    -0.00572,  0.6088, -1.2176,   // RL
    0.0, 0.0, 0.0, 0.0            // wheels
};

TaskConfig get_task(const std::string& name) {
    TaskConfig cfg;  // filled with struct defaults

    if (name == "stand") {
        std::memcpy(cfg.nominal_pose, STAND_POSE, sizeof(STAND_POSE));
        cfg.goal_pos[0] = 0.0;
        cfg.goal_pos[1] = 0.0;
        cfg.goal_pos[2] = 0.464;
        // cost weights: default CostWeights (standing — no locomotion pull)
    }
    else if (name == "walk_straight") {
        std::memcpy(cfg.nominal_pose, STAND_POSE, sizeof(STAND_POSE));
        cfg.goal_pos[0] = 5.0;
        cfg.goal_pos[1] = 0.0;
        cfg.goal_pos[2] = 0.464;
        // TODO: add locomotion cost weights and gait reference when implementing walking
    }
    else {
        throw std::runtime_error("Unknown task: " + name);
    }

    return cfg;
}
