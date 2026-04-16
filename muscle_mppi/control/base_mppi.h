#pragma once

#include <mujoco/mujoco.h>
#include <vector>
#include <random>
#include <string>
#include <algorithm>
#include "../utils/tasks.h"

// Sensor data assembled from rt/lowstate + rt/sportmodestate
struct RobotState {
    double pos[3]  = {};            // base position (x, y, z)
    double vel[3]  = {};            // base linear velocity
    double quat[4] = {1,0,0,0};    // IMU quaternion (w, x, y, z)
    double gyro[3] = {};            // IMU angular velocity
    double q[16]   = {};            // joint positions
    double dq[16]  = {};            // joint velocities
    bool   valid   = false;
};

// Engine base — analogous to MPPI_quad's base_controller.py (BaseMPPI).
// Owns the MuJoCo model, data pool, trajectory/noise/cost buffers, and
// noise sampling.  Does NOT know about cost functions or control strategy;
// those are provided by the derived class via rollout().
class BaseMPPI {
public:
    explicit BaseMPPI(const TaskConfig& task);
    virtual ~BaseMPPI();

    // Overwrite height target at runtime (measured from real robot at handoff)
    void set_height_target(double z) { height_target_ = z; }

    // Cost diagnostics — valid after update() returns
    double cost_min()  const { return *std::min_element(costs_.begin(), costs_.end()); }
    double cost_max()  const { return *std::max_element(costs_.begin(), costs_.end()); }
    double cost_mean() const {
        double s = 0.0;
        for (auto c : costs_) s += c;
        return s / static_cast<double>(costs_.size());
    }

protected:
    // Derived class provides one rollout; returns total cost for sample s.
    virtual double rollout(int s, const RobotState& state) = 0;

    // Helpers available to derived class
    void   sample_noise();
    void   set_mj_state(mjData* d, const RobotState& state);

    TaskConfig task_;

    mjModel*             model_ = nullptr;
    std::vector<mjData*> data_;        // one mjData per sample

    std::vector<double> trajectory_;   // [horizon × NUM_JOINTS] position targets
    std::vector<double> noise_;        // [n_samples × horizon × NUM_JOINTS]
    std::vector<double> costs_;        // [n_samples]

    double height_target_;

    std::mt19937 rng_;
    std::normal_distribution<double> normal_{0.0, 1.0};
};
