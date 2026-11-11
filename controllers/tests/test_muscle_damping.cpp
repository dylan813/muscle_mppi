// The muscles' force-velocity damping (hill_velocity_damping, muscle/control/
// muscle.h) and its implicit integration through the motor (common/motor.h):
//   - times joint velocity, it is the part of the Hill torque that depends on
//     velocity (the torque's slope at zero velocity);
//   - co-contraction raises it;
//   - on a light joint at a 10 ms step, applying it explicitly (held for the
//     step) makes the velocity flip sign every step, while handing it to
//     MuJoCo as joint damping (motor damping: implicit) decays it smoothly;
//   - with explicit damping the motor's command is exactly what it was before
//     the muscles' damping was passed to it.

#include <algorithm>
#include <cstdio>
#include <vector>

#include <mujoco/mujoco.h>

#include "check.h"
#include "../common/motor.h"
#include "../muscle/control/muscle.h"

// default_muscle_quad in muscle/utils/tasks.yaml (2026-11-10), per joint type.
static MuscleParams walk_muscle()
{
    static constexpr double peak_force[3] = {149.7, 384.1, 255.1};
    static constexpr double phi_min[3]    = {-0.9472, -1.4708, -2.6227};
    static constexpr double phi_max[3]    = { 0.9472,  3.3907, -0.9378};
    static constexpr double lce_min[3]    = {0.7118, 0.8777, 0.8667};
    static constexpr double lce_max[3]    = {1.1104, 1.0504, 1.1446};
    static constexpr double pFLmax[3]     = {0.3260, 1.3035, 0.3589};
    static constexpr double vmax[3]       = {4.77, 1.86, 2.80};
    static constexpr double FVmax[3]      = {1.1459, 1.2520, 1.0597};

    MuscleParams p;
    p.act_bandwidth = 20.0;
    for (int j = 0; j < NUM_JOINTS; ++j) {
        const int t = j % 3;
        p.peak_force[j] = peak_force[t];
        p.phi_min[j]    = phi_min[t];
        p.phi_max[j]    = phi_max[t];
        p.lce_min[j]    = lce_min[t];
        p.lce_max[j]    = lce_max[t];
        p.pFLmax[j]     = pFLmax[t];
        p.vmax[j]       = vmax[t];
        p.FVmax[j]      = FVmax[t];
        p.kd_sim[j]     = 0.1;
    }
    p.activation_dynamics = false;   // activation = command, so a torque is a function of (q, dq)
    return p;
}

static void torque(const MuscleParams& p, const double act[NUM_MUSCLES], const double q[NUM_JOINTS],
                   const double dq[NUM_JOINTS], double tau[NUM_JOINTS])
{
    double activation[NUM_MUSCLES];
    std::copy(act, act + NUM_MUSCLES, activation);
    hill_compute_torques(act, q, dq, p, 0.01, activation, tau);
}

// hill_velocity_damping · q̇ = −(τ(q̇) − τ(0)) for hill_compute_torques' τ, at
// velocities on every piece of the force-velocity curve (shortening,
// lengthening below and on the plateau, past vmax); at q̇ = 0 it is the
// torque's slope, by central differences.
static void test_damping_is_velocity_torque()
{
    const MuscleParams p = walk_muscle();

    double act[NUM_MUSCLES], q[NUM_JOINTS], rest[NUM_JOINTS], tau0[NUM_JOINTS];
    for (int m = 0; m < NUM_MUSCLES; ++m) act[m] = 0.2 + 0.03 * m;
    for (int j = 0; j < NUM_JOINTS; ++j) { q[j] = 0.5 * (p.phi_min[j] + p.phi_max[j]) + 0.1; rest[j] = 0.0; }
    torque(p, act, q, rest, tau0);

    for (double speed : {0.3, -0.3, 2.0, -2.0, 40.0}) {
        double dq[NUM_JOINTS], damping[NUM_JOINTS], tau[NUM_JOINTS];
        for (int j = 0; j < NUM_JOINTS; ++j) dq[j] = speed * (j % 2 ? -1.0 : 1.0);
        hill_velocity_damping(act, q, dq, p, damping);
        torque(p, act, q, dq, tau);

        for (int j = 0; j < NUM_JOINTS; ++j) {
            const double velocity_part = tau[j] - tau0[j];
            CHECK(damping[j] >= 0.0);
            CHECK_NEAR(-damping[j] * dq[j], velocity_part, 1e-9 * std::max(1.0, std::abs(velocity_part)));
        }
    }

    double damping[NUM_JOINTS], dq_hi[NUM_JOINTS], dq_lo[NUM_JOINTS], tau_hi[NUM_JOINTS], tau_lo[NUM_JOINTS];
    const double h = 1e-6;
    for (int j = 0; j < NUM_JOINTS; ++j) { dq_hi[j] = h; dq_lo[j] = -h; }
    hill_velocity_damping(act, q, rest, p, damping);
    torque(p, act, q, dq_hi, tau_hi);
    torque(p, act, q, dq_lo, tau_lo);
    for (int j = 0; j < NUM_JOINTS; ++j) {
        const double slope = -(tau_hi[j] - tau_lo[j]) / (2.0 * h);
        CHECK_NEAR(damping[j], slope, 1e-6 * std::max(1.0, slope));
    }
}

// At zero velocity the damping is proportional to a1 + a2: doubling both
// activations doubles it.
static void test_cocontraction_raises_damping()
{
    const MuscleParams p = walk_muscle();
    double q[NUM_JOINTS], dq[NUM_JOINTS] = {}, low[NUM_JOINTS], high[NUM_JOINTS];
    for (int j = 0; j < NUM_JOINTS; ++j) q[j] = 0.5 * (p.phi_min[j] + p.phi_max[j]);

    double act_low[NUM_MUSCLES], act_high[NUM_MUSCLES];
    std::fill(act_low, act_low + NUM_MUSCLES, 0.4);
    std::fill(act_high, act_high + NUM_MUSCLES, 0.8);
    hill_velocity_damping(act_low, q, dq, p, low);
    hill_velocity_damping(act_high, q, dq, p, high);

    for (int j = 0; j < NUM_JOINTS; ++j) {
        CHECK(low[j] > 0.0);
        CHECK_NEAR(high[j], 2.0 * low[j], 1e-9 * high[j]);
    }
}

// One hinge with the swinging calf's inertia (0.0163 kg·m², go2's mass matrix
// at the stand-up pose), no gravity, go2's joint damping 0.1, driven only by
// the calf's muscle pair through the motor: force-length and passive force off,
// so the torque is force-velocity alone, a damper (k ≈ 6.9 N·m·s/rad at
// activations 0.7, so k·Δt/I ≈ 4.3 at 10 ms: explicit overshoots).
static constexpr int kCalf = 2;

static const char* kHingeXml = R"(
<mujoco>
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <body>
      <joint name="calf" type="hinge" axis="0 1 0" damping="0.1"/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.0163 0.0163 0.0163"/>
    </body>
  </worldbody>
  <actuator><motor joint="calf"/></actuator>
</mujoco>)";

// Joint velocity at the end of each of `steps` 10 ms steps from dq0.
static std::vector<double> swing(MotorDamping damping, double dq0, int steps)
{
    char error[1000] = "";
    mjSpec* spec = mj_parseXMLString(kHingeXml, nullptr, error, sizeof(error));
    mjModel* m = spec ? mj_compile(spec, nullptr) : nullptr;
    CHECK(m != nullptr);
    if (!m) { std::fprintf(stderr, "%s\n", error); return {}; }
    mjData* d = mj_makeData(m);
    d->qvel[0] = dq0;

    MuscleParams p = walk_muscle();
    p.use_fl = false;
    p.use_passive = false;

    MotorParams mp;
    mp.physics_dt = 0.01;
    mp.damping    = damping;
    double no_kd[NUM_JOINTS] = {}, lo[NUM_JOINTS], hi[NUM_JOINTS];
    std::fill(lo, lo + NUM_JOINTS, -1e9);
    std::fill(hi, hi + NUM_JOINTS, 1e9);
    Motor motor;
    motor.init(mp, no_kd, lo, hi);
    MotorState s;

    double act[NUM_MUSCLES], activation[NUM_MUSCLES];
    std::fill(act, act + NUM_MUSCLES, 0.7);
    std::copy(act, act + NUM_MUSCLES, activation);

    std::vector<double> out;
    for (int k = 0; k < steps; ++k) {
        double q[NUM_JOINTS] = {}, dq[NUM_JOINTS] = {}, tau[NUM_JOINTS], c[NUM_JOINTS];
        double ctrl[NUM_JOINTS], tau_cmd[NUM_JOINTS];
        q[kCalf]  = d->qpos[0];
        dq[kCalf] = d->qvel[0];

        hill_compute_torques(act, q, dq, p, mp.physics_dt, activation, tau);
        hill_velocity_damping(activation, q, dq, p, c);
        motor.command(tau, c, dq, s, ctrl, tau_cmd);

        d->ctrl[0] = ctrl[kCalf];
        m->dof_damping[0] = p.kd_sim[kCalf] + motor.implicit_damping(kCalf, c[kCalf]);
        mj_step(m, d);

        dq[kCalf] = d->qvel[0];
        motor.applied(ctrl, c, dq, s);
        out.push_back(d->qvel[0]);
    }

    mj_deleteData(d);
    mj_deleteModel(m);
    mj_deleteSpec(spec);
    return out;
}

static void test_explicit_flips_implicit_decays()
{
    const double dq0 = 1.0;   // rad/s

    // Held for 10 ms, the damping torque reverses the joint on the first step.
    const std::vector<double> held = swing(MotorDamping::Explicit, dq0, 5);
    CHECK(held.size() == 5 && held[0] < 0.0 && held[1] > 0.0);

    // Inside MuJoCo's step: the speed shrinks every step and never flips.
    const std::vector<double> inside = swing(MotorDamping::Implicit, dq0, 5);
    CHECK(inside.size() == 5);
    double prev = dq0;
    for (double v : inside) {
        CHECK(v > 0.0 && v < prev);
        prev = v;
    }
    CHECK(inside.back() < 0.02 * dq0);
}

// With explicit damping the muscles' damping is not used: the command is
// bit-identical to the motor's command without it, and so is its state after.
static void test_explicit_command_unchanged()
{
    MotorParams mp;
    mp.physics_dt = 0.002;
    mp.bandwidth  = 300.0;
    double kd[NUM_JOINTS], lo[NUM_JOINTS], hi[NUM_JOINTS];
    double tau[NUM_JOINTS], c[NUM_JOINTS], dq[NUM_JOINTS], dq_after[NUM_JOINTS];
    for (int j = 0; j < NUM_JOINTS; ++j) {
        kd[j] = 0.5 + 0.1 * j;  lo[j] = -20.0;  hi[j] = 20.0;
        tau[j] = 3.0 * j - 15.0;  c[j] = 4.0 + j;  dq[j] = 0.7 * j - 3.0;  dq_after[j] = dq[j] - 0.2;
    }
    Motor motor;
    motor.init(mp, kd, lo, hi);
    MotorState s;
    for (int j = 0; j < NUM_JOINTS; ++j) s.tau[j] = 1.0 - 0.2 * j;

    double ctrl_with[NUM_JOINTS], ctrl_without[NUM_JOINTS], cmd_with[NUM_JOINTS], cmd_without[NUM_JOINTS];
    motor.command(tau, c, dq, s, ctrl_with, cmd_with);
    motor.command(tau, dq, s, ctrl_without, cmd_without);

    MotorState after_with = s, after_without = s;
    motor.applied(ctrl_with, c, dq_after, after_with);
    motor.applied(ctrl_without, dq_after, after_without);

    for (int j = 0; j < NUM_JOINTS; ++j) {
        CHECK(ctrl_with[j] == ctrl_without[j]);
        CHECK(cmd_with[j] == cmd_without[j]);
        CHECK(after_with.tau[j] == after_without.tau[j]);
        CHECK(motor.implicit_damping(j, c[j]) == 0.0);
    }
}

int main()
{
    test_damping_is_velocity_torque();
    test_cocontraction_raises_damping();
    test_explicit_flips_implicit_decays();
    test_explicit_command_unchanged();
    return check::exit_code();
}
