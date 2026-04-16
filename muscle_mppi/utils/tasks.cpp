#include "tasks.h"
#include <stdexcept>
#include <cstring>

static const double STAND_POSE[NUM_JOINTS] = {
     0.00572,  0.6088, -1.2176,   // FR
    -0.00572,  0.6088, -1.2176,   // FL
     0.00572,  0.6088, -1.2176,   // RR
    -0.00572,  0.6088, -1.2176,   // RL
    0.0, 0.0, 0.0, 0.0
};

TaskConfig get_task(const std::string& name) {
    TaskConfig cfg;

    if (name == "stand") {
        std::memcpy(cfg.nominal_pose, STAND_POSE, sizeof(STAND_POSE));
        cfg.goal_pos[0] = 0.0;
        cfg.goal_pos[1] = 0.0;
        cfg.goal_pos[2] = 0.464;
        // cost and muscle use struct defaults
    }
    else if (name == "walk_straight") {
        std::memcpy(cfg.nominal_pose, STAND_POSE, sizeof(STAND_POSE));
        cfg.goal_pos[0] = 5.0;
        cfg.goal_pos[1] = 0.0;
        cfg.goal_pos[2] = 0.464;
    }
    else {
        throw std::runtime_error("Unknown task: " + name);
    }

    return cfg;
}
