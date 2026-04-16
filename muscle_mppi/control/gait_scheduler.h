#pragma once

#include "../utils/tasks.h"
#include "base_mppi.h"   // RobotState

// -----------------------------------------------------------------------
// Operator motion command — set externally (joystick, nav stack, etc.)
// -----------------------------------------------------------------------
struct MotionCommand {
    double vx     = 0.0;    // desired forward  body velocity (m/s)
    double vy     = 0.0;    // desired lateral  body velocity (m/s)
    double wz     = 0.0;    // desired yaw rate (rad/s)
    double height = 0.464;  // desired body height (m)
};

// -----------------------------------------------------------------------
// Reference state for one control step in the MPPI horizon.
// Cost = || simulated_state - StepReference ||_weighted
// -----------------------------------------------------------------------
struct StepReference {
    // Body targets
    double body_pos[3]   = {0, 0, 0.464}; // x, y, z
    double body_quat[4]  = {1, 0, 0, 0};  // w, x, y, z (upright)
    double body_vel[3]   = {};             // linear  (world frame)
    double body_omega[3] = {};             // angular (body frame)

    // Leg joint targets (NUM_LEG_JOINTS = 12)
    double joint_pos[NUM_LEG_JOINTS] = {};

    // Wheel velocity targets (rad/s, NUM_WHEELS = 4)
    double wheel_vel[NUM_WHEELS] = {};

    // Contact schedule: true = stance, false = swing  (FR, FL, RR, RL)
    bool stance[4] = {true, true, true, true};
};

// -----------------------------------------------------------------------
// Full reference trajectory over the MPPI horizon.
// -----------------------------------------------------------------------
struct GaitReference {
    static constexpr int MAX_STEPS = 20;
    StepReference steps[MAX_STEPS];
    int           count = 0;
};

// -----------------------------------------------------------------------
// GaitScheduler — generates body/joint/wheel reference trajectories.
//
// Current implementation: stand + velocity tracking (all feet in stance).
// Produces a smooth reference that follows the current robot state forward
// at the commanded velocity.  This gives MPPI a "moving target" that:
//   - provides gradient toward desired velocity (vs fixed-pose tracking)
//   - adapts the reference to the robot's current position each update
//   - is ready to extend with trot/walk contact phases
//
// Extension points (TODO):
//   - Trot gait: alternate stance/swing every gait_period_/2 based on phase
//   - Swing foot trajectories: Bezier arcs between lift-off and touch-down
//   - Yaw integration: accumulate heading from wz command
// -----------------------------------------------------------------------
class GaitScheduler {
public:
    // nominal_pose: NUM_JOINTS array, standing reference (actuator order)
    // wheel_radius: metres — used for wheel_vel = vx / r
    explicit GaitScheduler(const double nominal_pose[NUM_JOINTS],
                           double wheel_radius = 0.045);

    void set_command(const MotionCommand& cmd) { cmd_ = cmd; }
    const MotionCommand& command() const { return cmd_; }

    // Populate ref_out with `horizon` steps of `dt_ctrl` seconds each.
    // state is the current robot state (used as integration start).
    void generate(const RobotState& state,
                  int horizon, double dt_ctrl,
                  GaitReference& ref_out) const;

private:
    double        nominal_pose_[NUM_JOINTS] = {};
    double        wheel_radius_;
    MotionCommand cmd_;
};
