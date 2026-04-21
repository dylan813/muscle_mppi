#include <iostream>
#include <cstring>
#include <cmath>
#include <chrono>
#include <unistd.h>

#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/common/time/time_tool.hpp>
#include <unitree/common/thread/thread.hpp>

#include "control/mppi_locomotion.h"

using namespace unitree::common;
using namespace unitree::robot;

#define TOPIC_LOWCMD    "rt/lowcmd"
#define TOPIC_LOWSTATE  "rt/lowstate"
#define TOPIC_SPORTMODE "rt/sportmodestate"

// Sentinel values that tell the bridge to ignore position/velocity and use raw torque
constexpr double PosStopF = 2.146E+9f;
constexpr double VelStopF = 16000.0f;

uint32_t crc32_core(uint32_t* ptr, uint32_t len) {
    uint32_t xbit, data, CRC32 = 0xFFFFFFFF;
    const uint32_t poly = 0x04c11db7;
    for (uint32_t i = 0; i < len; ++i) {
        xbit = 1u << 31; data = ptr[i];
        for (int b = 0; b < 32; ++b) {
            if (CRC32 & 0x80000000) { CRC32 <<= 1; CRC32 ^= poly; } else CRC32 <<= 1;
            if (data & xbit) CRC32 ^= poly;
            xbit >>= 1;
        }
    }
    return CRC32;
}

class MPPIController {
public:
    explicit MPPIController(const std::string& task = "stand") : mppi_(task) {}

    void Init() {
        InitLowCmd();

        lowcmd_publisher_.reset(
            new ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
        lowcmd_publisher_->InitChannel();

        lowstate_subscriber_.reset(
            new ChannelSubscriber<unitree_go::msg::dds_::LowState_>(TOPIC_LOWSTATE));
        lowstate_subscriber_->InitChannel(
            std::bind(&MPPIController::LowStateHandler, this, std::placeholders::_1), 1);

        sportmode_subscriber_.reset(
            new ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>(TOPIC_SPORTMODE));
        sportmode_subscriber_->InitChannel(
            std::bind(&MPPIController::SportModeHandler, this, std::placeholders::_1), 1);

        control_thread_ = CreateRecurrentThreadEx(
            "mppi_ctrl", UT_CPU_ID_NONE, 20000, &MPPIController::ControlLoop, this);
    }

private:
    void InitLowCmd() {
        low_cmd_.head()[0]    = 0xFE;
        low_cmd_.head()[1]    = 0xEF;
        low_cmd_.level_flag() = 0xFF;
        low_cmd_.gpio()       = 0;
        for (int i = 0; i < 20; ++i) {
            low_cmd_.motor_cmd()[i].mode() = 0x01;
            low_cmd_.motor_cmd()[i].q()    = PosStopF;
            low_cmd_.motor_cmd()[i].kp()   = 0.0;
            low_cmd_.motor_cmd()[i].dq()   = VelStopF;
            low_cmd_.motor_cmd()[i].kd()   = 0.0;
            low_cmd_.motor_cmd()[i].tau()  = 0.0;
        }
    }

    void LowStateHandler(const void* msg) {
        const auto* s = static_cast<const unitree_go::msg::dds_::LowState_*>(msg);
        for (int i = 0; i < NUM_JOINTS; ++i) {
            state_.q[i]  = s->motor_state()[i].q();
            state_.dq[i] = s->motor_state()[i].dq();
        }
        state_.quat[0] = s->imu_state().quaternion()[0];
        state_.quat[1] = s->imu_state().quaternion()[1];
        state_.quat[2] = s->imu_state().quaternion()[2];
        state_.quat[3] = s->imu_state().quaternion()[3];
        state_.gyro[0] = s->imu_state().gyroscope()[0];
        state_.gyro[1] = s->imu_state().gyroscope()[1];
        state_.gyro[2] = s->imu_state().gyroscope()[2];
    }

    void SportModeHandler(const void* msg) {
        const auto* s = static_cast<const unitree_go::msg::dds_::SportModeState_*>(msg);
        state_.pos[0] = s->position()[0];
        state_.pos[1] = s->position()[1];
        state_.pos[2] = s->position()[2];
        state_.vel[0] = s->velocity()[0];
        state_.vel[1] = s->velocity()[1];
        state_.vel[2] = s->velocity()[2];
        state_.valid  = true;
    }

    void ControlLoop() {
        running_time_ += dt_;

        if (running_time_ < STANDUP_DURATION) {
            // ---- Phase 1: PD stand-up (identical to baseline) ----
            double phase = std::tanh(running_time_ / 1.2);
            for (int i = 0; i < NUM_LEG_JOINTS; ++i) {
                low_cmd_.motor_cmd()[i].q()   = phase * stand_pos_[i] + (1.0 - phase) * crouch_pos_[i];
                low_cmd_.motor_cmd()[i].kp()  = phase * 50.0 + (1.0 - phase) * 20.0;
                low_cmd_.motor_cmd()[i].dq()  = 0.0;
                low_cmd_.motor_cmd()[i].kd()  = 3.5;
                low_cmd_.motor_cmd()[i].tau() = 0.0;
            }
            for (int i = NUM_LEG_JOINTS; i < NUM_JOINTS; ++i) {
                low_cmd_.motor_cmd()[i].q()   = 0.0;
                low_cmd_.motor_cmd()[i].kp()  = 0.0;
                low_cmd_.motor_cmd()[i].dq()  = 0.0;
                low_cmd_.motor_cmd()[i].kd()  = 0.5;
                low_cmd_.motor_cmd()[i].tau() = 0.0;
            }
        } else {
            // ---- Phase 2: MPPI (activation commands) + Hill-model torque control ----
            double activations[NUM_JOINTS] = {};
            mppi_.update(state_, activations);

            // Transition: set motion command, capture height, print diagnostic
            if (running_time_ - dt_ < STANDUP_DURATION) {
                mppi_.set_command({.vx = 0.1});   // Phase 2: slow forward walk
                mppi_.set_height_target(state_.pos[2]);

                auto t0 = std::chrono::steady_clock::now();
                mppi_.update(state_, activations);
                auto t1 = std::chrono::steady_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

                std::cout << "Switching to muscle MPPI. Update: " << ms << " ms\n";
                std::cout << "  base z            = " << state_.pos[2]  << " m\n";
                std::cout << "  quat w            = " << state_.quat[0] << "\n";
                std::cout << "  base vel          = " << state_.vel[0]  << " "
                          << state_.vel[1] << " " << state_.vel[2]  << "\n";
                std::cout << "  activation[0]     = " << mppi_.muscle_state().activation[0] << "\n";
                std::cout << "  cost min/mean/max = "
                          << mppi_.cost_min()  << " / "
                          << mppi_.cost_mean() << " / "
                          << mppi_.cost_max()  << "\n";
                std::cout << "  command (vx,vy,wz) = ("
                          << mppi_.command().vx << ", "
                          << mppi_.command().vy << ", "
                          << mppi_.command().wz << ")\n";

                auto bd = mppi_.diagnose_cost(state_);
                std::cout << "  -- cost breakdown (zero-noise nominal rollout) --\n"
                          << "     height        = " << bd.height        << "\n"
                          << "     orientation   = " << bd.orientation   << "\n"
                          << "     posture       = " << bd.posture       << "\n"
                          << "     contact_vel   = " << bd.contact_vel   << "\n"
                          << "     contact_force = " << bd.contact_force << "\n"
                          << "     act_smooth    = " << bd.act_smooth    << "\n"
                          << "     terminal      = " << bd.terminal      << "\n"
                          << "     TOTAL         = " << bd.total()       << "\n";
            }

            // Hill model feedforward + hardware velocity damping.
            // tau_effective[i] = tau_hill[i] - kd_[i] * actual_dq[i]
            double tau_cmd[NUM_JOINTS] = {};
            mppi_.compute_real_torques(state_, activations, tau_cmd);

            for (int i = 0; i < NUM_JOINTS; ++i) {
                low_cmd_.motor_cmd()[i].q()   = PosStopF;
                low_cmd_.motor_cmd()[i].kp()  = 0.0;
                low_cmd_.motor_cmd()[i].dq()  = 0.0;
                low_cmd_.motor_cmd()[i].kd()  = kd_[i];
                low_cmd_.motor_cmd()[i].tau() = tau_cmd[i];
            }
        }

        low_cmd_.crc() = crc32_core(
            reinterpret_cast<uint32_t*>(&low_cmd_),
            (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
        lowcmd_publisher_->Write(low_cmd_);
    }

    static constexpr double dt_              = 0.02;
    static constexpr double STANDUP_DURATION = 3.0;

    // Hardware velocity damping during Hill torque mode.
    // tau_eff[i] = tau_hill[i] - kd_[i] * actual_dq[i]
    // Start at standup kd for legs — tune down once standing is confirmed stable.
    const double kd_[NUM_JOINTS] = {
        2.0, 3.5, 3.5,   // FR  hip / thigh / calf
        2.0, 3.5, 3.5,   // FL
        2.0, 3.5, 3.5,   // RR
        2.0, 3.5, 3.5,   // RL
        2.0, 2.0, 2.0, 2.0  // wheels
    };

    const double stand_pos_[NUM_LEG_JOINTS] = {
         0.00572,  0.6088, -1.2176,
        -0.00572,  0.6088, -1.2176,
         0.00572,  0.6088, -1.2176,
        -0.00572,  0.6088, -1.2176
    };
    const double crouch_pos_[NUM_LEG_JOINTS] = {
         0.04735,  1.2219, -2.4438,
        -0.04735,  1.2219, -2.4438,
         0.04735,  1.2219, -2.4438,
        -0.04735,  1.2219, -2.4438
    };

    double running_time_ = 0.0;

    MPPILocomotion mppi_;
    RobotState     state_{};
    unitree_go::msg::dds_::LowCmd_ low_cmd_{};

    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_>            lowcmd_publisher_;
    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_>         lowstate_subscriber_;
    ChannelSubscriberPtr<unitree_go::msg::dds_::SportModeState_>   sportmode_subscriber_;

    ThreadPtr control_thread_;
};

int main(int argc, const char** argv) {
    if (argc < 2)
        ChannelFactory::Instance()->Init(1, "lo");
    else
        ChannelFactory::Instance()->Init(0, argv[1]);

    std::cout << "MPPI Controller (Hill muscle model) — press Enter to start\n";
    std::cin.get();

    MPPIController controller("stand");
    controller.Init();

    while (true) sleep(10);
    return 0;
}
