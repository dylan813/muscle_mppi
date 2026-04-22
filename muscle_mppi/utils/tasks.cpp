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
    TaskConfig cfg;

    if (name == "stand") {
        std::memcpy(cfg.nominal_pose, STAND_POSE, sizeof(STAND_POSE));
        cfg.goal_pos[0] = 0.0;
        cfg.goal_pos[1] = 0.0;
    }
    else if (name == "walk_straight") {
        std::memcpy(cfg.nominal_pose, STAND_POSE, sizeof(STAND_POSE));
        cfg.goal_pos[0] = 5.0;
        cfg.goal_pos[1] = 0.0;
    }
    else {
        throw std::runtime_error("Unknown task: " + name);
    }

    return cfg;
}
