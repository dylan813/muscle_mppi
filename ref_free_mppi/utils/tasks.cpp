#include "tasks.h"
#include <stdexcept>
#include <cstring>

static const double STAND_POSE[NUM_JOINTS] = {
    0.0,  0.67, -1.3,   // FR leg
    0.0,  0.67, -1.3,   // FL leg
    0.0,  0.67, -1.3,   // RR leg
    0.0,  0.67, -1.3,   // RL leg
    0.0, 0.0, 0.0, 0.0  // wheels
};

TaskConfig get_task(const std::string& name) {
    TaskConfig cfg;

    if (name == "stand") {
        std::memcpy(cfg.nominal_pose, STAND_POSE, sizeof(STAND_POSE));
        cfg.cost.vel_cmd      = 0.0;
        cfg.cost.vel_des[0]   = 0.0;
        cfg.cost.vel_des[1]   = 0.0;
        cfg.cost.vel_des[2]   = 0.0;
    }
    else if (name == "walk_forward") {
        std::memcpy(cfg.nominal_pose, STAND_POSE, sizeof(STAND_POSE));
        cfg.cost.height       = 1e2;
        cfg.cost.orientation  = 10.0;
        cfg.cost.joint_reg    = 0.0;
        cfg.cost.contact_vel  = 0.5;
        cfg.cost.contact_frc  = 5e-2;
        cfg.cost.terminal     = 2.5e3;
        cfg.cost.vel_cmd      = 1.0;
        cfg.cost.vel_des[0]   = 0.5;  // 0.5 m/s forward
        cfg.cost.vel_des[1]   = 0.0;
        cfg.cost.vel_des[2]   = 0.0;
        cfg.rf.H_time         = 0.9;  // longer horizon for gait discovery
        cfg.n_samples         = 50;
    }
    else {
        throw std::runtime_error("Unknown task: " + name);
    }

    return cfg;
}
