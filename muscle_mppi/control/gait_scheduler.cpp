#include "gait_scheduler.h"

#include <cstring>
#include <cmath>

GaitScheduler::GaitScheduler(const double nominal_pose[NUM_JOINTS],
                             double wheel_radius)
    : wheel_radius_(wheel_radius)
{
    std::memcpy(nominal_pose_, nominal_pose, NUM_JOINTS * sizeof(double));
}

void GaitScheduler::generate(const RobotState& state,
                              int horizon, double dt_ctrl,
                              GaitReference& ref_out) const
{
    ref_out.count = horizon;

    // Use the robot's current XY position as the integration origin so the
    // reference always starts from where the robot actually is.  This means
    // MPPI never penalises drift that has already happened — it only penalises
    // deviation from the *future* commanded path.
    const double px0 = state.pos[0];
    const double py0 = state.pos[1];

    for (int t = 0; t < horizon; ++t) {
        StepReference& r = ref_out.steps[t];

        // ------------------------------------------------------------------
        // Body position reference: integrate commanded velocity from now
        // ------------------------------------------------------------------
        const double dt = dt_ctrl * (t + 1);
        r.body_pos[0] = px0 + cmd_.vx * dt;
        r.body_pos[1] = py0 + cmd_.vy * dt;
        r.body_pos[2] = cmd_.height;

        // ------------------------------------------------------------------
        // Body orientation: stay upright (yaw tracking is a TODO extension)
        // ------------------------------------------------------------------
        r.body_quat[0] = 1.0;
        r.body_quat[1] = 0.0;
        r.body_quat[2] = 0.0;
        r.body_quat[3] = 0.0;

        // ------------------------------------------------------------------
        // Body velocity: track commanded linear and angular velocity
        // ------------------------------------------------------------------
        r.body_vel[0]   = cmd_.vx;
        r.body_vel[1]   = cmd_.vy;
        r.body_vel[2]   = 0.0;
        r.body_omega[0] = 0.0;
        r.body_omega[1] = 0.0;
        r.body_omega[2] = cmd_.wz;

        // ------------------------------------------------------------------
        // Joint targets: nominal standing pose for all leg joints.
        // (Swing trajectories would modify these for locomotion gaits.)
        // ------------------------------------------------------------------
        for (int j = 0; j < NUM_LEG_JOINTS; ++j)
            r.joint_pos[j] = nominal_pose_[j];

        // ------------------------------------------------------------------
        // Wheel velocity: rigid-body rolling at commanded forward speed.
        // All 4 wheels get the same target for straight driving; differential
        // steering from wz is a TODO extension.
        // ------------------------------------------------------------------
        const double wheel_spd = (wheel_radius_ > 1e-6)
                                  ? cmd_.vx / wheel_radius_ : 0.0;
        for (int w = 0; w < NUM_WHEELS; ++w)
            r.wheel_vel[w] = wheel_spd;

        // ------------------------------------------------------------------
        // Contact: stand gait — all feet always in stance.
        // Trot gait would alternate pairs based on gait phase here.
        // ------------------------------------------------------------------
        for (int leg = 0; leg < 4; ++leg)
            r.stance[leg] = true;
    }
}
