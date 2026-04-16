#include "base_mppi.h"

#include <cstring>
#include <stdexcept>

static void mujoco_warning_noop(const char*) {}

BaseMPPI::BaseMPPI(const TaskConfig& task)
    : task_(task), height_target_(task.height_target), rng_(std::random_device{}())
{
    mju_user_warning = mujoco_warning_noop;

    char error[1000];
    model_ = mj_loadXML(task_.model_path, nullptr, error, sizeof(error));
    if (!model_) throw std::runtime_error("Failed to load model: " + std::string(error));

    model_->opt.timestep = task_.dt;

    data_.resize(task_.n_samples);
    for (int i = 0; i < task_.n_samples; ++i)
        data_[i] = mj_makeData(model_);

    trajectory_.assign(task_.horizon * NUM_JOINTS, 0.0);
    noise_.assign(task_.n_samples * task_.horizon * NUM_JOINTS, 0.0);
    costs_.resize(task_.n_samples);
}

BaseMPPI::~BaseMPPI() {
    for (auto* d : data_) mj_deleteData(d);
    mj_deleteModel(model_);
}

void BaseMPPI::sample_noise() {
    for (int s = 0; s < task_.n_samples; ++s)
        for (int t = 0; t < task_.horizon; ++t)
            for (int j = 0; j < NUM_JOINTS; ++j) {
                int idx = s * task_.horizon * NUM_JOINTS + t * NUM_JOINTS + j;
                noise_[idx] = task_.noise_sigma[j] * normal_(rng_);
            }
}

// -----------------------------------------------------------------------
// Write a RobotState into a mjData, then call mj_forward.
// qpos: [base_xyz(3), base_quat(4), joints(16)] = 23
// qvel: [base_linvel(3), base_angvel(3), joint_vel(16)] = 22
// -----------------------------------------------------------------------
void BaseMPPI::set_mj_state(mjData* d, const RobotState& state) {
    mj_resetData(model_, d);

    d->qpos[0] = state.pos[0];
    d->qpos[1] = state.pos[1];
    d->qpos[2] = state.pos[2];
    d->qpos[3] = state.quat[0];  // w
    d->qpos[4] = state.quat[1];  // x
    d->qpos[5] = state.quat[2];  // y
    d->qpos[6] = state.quat[3];  // z

    // Use LS_TO_QPOS mapping: LowState/actuator order → MuJoCo DOF tree order
    for (int j = 0; j < NUM_JOINTS; ++j)
        d->qpos[7 + LS_TO_QPOS[j]] = state.q[j];

    d->qvel[0] = state.vel[0];
    d->qvel[1] = state.vel[1];
    d->qvel[2] = state.vel[2];
    d->qvel[3] = state.gyro[0];
    d->qvel[4] = state.gyro[1];
    d->qvel[5] = state.gyro[2];

    for (int j = 0; j < NUM_JOINTS; ++j)
        d->qvel[6 + LS_TO_QPOS[j]] = state.dq[j];

    mj_forward(model_, d);
}
