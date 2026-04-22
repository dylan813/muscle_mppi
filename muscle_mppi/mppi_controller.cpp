#include <iostream>
#include <cstring>
#include <cmath>
#include <chrono>
#include <mutex>
#include <thread>
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

        mppi_thread_ = std::thread(&MPPIController::MPPILoop, this);
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
        std::lock_guard<std::mutex> lk(state_mutex_);
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
        std::lock_guard<std::mutex> lk(state_mutex_);
        state_.pos[0] = s->position()[0];
        state_.pos[1] = s->position()[1];
        state_.pos[2] = s->position()[2];
        state_.vel[0] = s->velocity()[0];
        state_.vel[1] = s->velocity()[1];
        state_.vel[2] = s->velocity()[2];
        state_.valid  = true;
    }

    // 50 Hz servo loop: sends cached MPPI torques; runs standup during phase 1
    void ControlLoop() {
        running_time_ += dt_;

        if (running_time_ < STANDUP_DURATION) {
            // Two-phase standup: 0–1.5 s → crouch, 1.5–3.0 s → stand
            double alpha;
            if (running_time_ < STANDUP_DURATION * 0.5) {
                alpha = running_time_ / (STANDUP_DURATION * 0.5);
                alpha = std::min(alpha, 1.0);
                for (int i = 0; i < NUM_LEG_JOINTS; ++i) {
                    low_cmd_.motor_cmd()[i].q()   = crouch_pos_[i];
                    low_cmd_.motor_cmd()[i].kp()  = alpha * 30.0;
                    low_cmd_.motor_cmd()[i].dq()  = 0.0;
                    low_cmd_.motor_cmd()[i].kd()  = 3.5;
                    low_cmd_.motor_cmd()[i].tau() = 0.0;
                }
            } else {
                alpha = (running_time_ - STANDUP_DURATION * 0.5) / (STANDUP_DURATION * 0.5);
                alpha = std::min(alpha, 1.0);
                for (int i = 0; i < NUM_LEG_JOINTS; ++i) {
                    low_cmd_.motor_cmd()[i].q()   = (1.0-alpha)*crouch_pos_[i] + alpha*stand_pos_[i];
                    low_cmd_.motor_cmd()[i].kp()  = 30.0 + alpha * 20.0;
                    low_cmd_.motor_cmd()[i].dq()  = 0.0;
                    low_cmd_.motor_cmd()[i].kd()  = 3.5;
                    low_cmd_.motor_cmd()[i].tau() = 0.0;
                }
            }
            for (int i = NUM_LEG_JOINTS; i < NUM_JOINTS; ++i) {
                low_cmd_.motor_cmd()[i].q()   = 0.0;
                low_cmd_.motor_cmd()[i].kp()  = 0.0;
                low_cmd_.motor_cmd()[i].dq()  = 0.0;
                low_cmd_.motor_cmd()[i].kd()  = 0.5;
                low_cmd_.motor_cmd()[i].tau() = 0.0;
            }
        } else {
            // Send cached Hill torques from background MPPI thread
            double tau_cmd[NUM_JOINTS];
            {
                std::lock_guard<std::mutex> lk(cmd_mutex_);
                std::copy(cached_tau_, cached_tau_ + NUM_JOINTS, tau_cmd);
            }
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

    // Background MPPI solver: starts after standup, runs as fast as it can
    void MPPILoop() {
        while (running_time_ < STANDUP_DURATION)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

        {
            std::lock_guard<std::mutex> lk(state_mutex_);
            mppi_.set_height_target(state_.pos[2]);
            std::cout << "Muscle MPPI started. height target = " << state_.pos[2] << " m\n";
        }

        int solve_count = 0;
        double solve_sum_ms = 0.0;

        while (true) {
            RobotState snap;
            {
                std::lock_guard<std::mutex> lk(state_mutex_);
                snap = state_;
            }
            if (!snap.valid) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            auto t0 = std::chrono::steady_clock::now();
            double activations[NUM_JOINTS] = {};
            mppi_.update(snap, activations);
            double tau_cmd[NUM_JOINTS] = {};
            mppi_.compute_real_torques(snap, activations, tau_cmd);
            double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();

            {
                std::lock_guard<std::mutex> lk(cmd_mutex_);
                std::copy(tau_cmd, tau_cmd + NUM_JOINTS, cached_tau_);
            }

            solve_sum_ms += ms;
            if (++solve_count % 20 == 0)
                std::cout << "Muscle MPPI avg solve: " << solve_sum_ms / solve_count << " ms\n";
        }
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
        0.0,  0.67, -1.3,
        0.0,  0.67, -1.3,
        0.0,  0.67, -1.3,
        0.0,  0.67, -1.3,
    };
    const double crouch_pos_[NUM_LEG_JOINTS] = {
        0.0,  1.36, -2.65,
        0.0,  1.36, -2.65,
        0.0,  1.36, -2.65,
        0.0,  1.36, -2.65,
    };

    double running_time_ = 0.0;

    MPPILocomotion mppi_;

    std::mutex state_mutex_;
    RobotState state_{};

    std::mutex cmd_mutex_;
    double cached_tau_[NUM_JOINTS] = {};

    unitree_go::msg::dds_::LowCmd_ low_cmd_{};

    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_>            lowcmd_publisher_;
    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_>         lowstate_subscriber_;
    ChannelSubscriberPtr<unitree_go::msg::dds_::SportModeState_>   sportmode_subscriber_;

    ThreadPtr   control_thread_;
    std::thread mppi_thread_;
};

int main(int argc, const char** argv) {
    if (argc < 2)
        ChannelFactory::Instance()->Init(1, "lo");
    else
        ChannelFactory::Instance()->Init(1, argv[1]);

    std::cout << "MPPI Controller (Hill muscle model) — press Enter to start\n";
    std::cin.get();

    const std::string task = (argc >= 3) ? argv[2] : "stand";
    MPPIController controller(task);
    controller.Init();

    while (true) sleep(10);
    return 0;
}
