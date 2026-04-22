#include <iostream>
#include <cstring>
#include <cmath>
#include <chrono>
#include <atomic>
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

#include "control/ref_free_mppi.h"

using namespace unitree::common;
using namespace unitree::robot;

#define TOPIC_LOWCMD    "rt/lowcmd"
#define TOPIC_LOWSTATE  "rt/lowstate"
#define TOPIC_SPORTMODE "rt/sportmodestate"

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

class RefFreeMPPIController {
public:
    explicit RefFreeMPPIController(const std::string& task = "stand")
        : mppi_(task) {}

    void Init() {
        InitLowCmd();

        lowcmd_publisher_.reset(
            new ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
        lowcmd_publisher_->InitChannel();

        lowstate_subscriber_.reset(
            new ChannelSubscriber<unitree_go::msg::dds_::LowState_>(TOPIC_LOWSTATE));
        lowstate_subscriber_->InitChannel(
            std::bind(&RefFreeMPPIController::LowStateHandler,
                      this, std::placeholders::_1), 1);

        sportmode_subscriber_.reset(
            new ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>(TOPIC_SPORTMODE));
        sportmode_subscriber_->InitChannel(
            std::bind(&RefFreeMPPIController::SportModeHandler,
                      this, std::placeholders::_1), 1);

        // 50 Hz servo loop — always sends from the latest cached command
        control_thread_ = CreateRecurrentThreadEx(
            "rf_servo", UT_CPU_ID_NONE, 20000,
            &RefFreeMPPIController::ControlLoop, this);

        // MPPI solver loop runs as fast as it can in a background thread.
        // It writes into cached_q_ / cached_dq_ protected by cmd_mutex_.
        mppi_thread_ = std::thread(&RefFreeMPPIController::MPPILoop, this);
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
        const auto* s =
            static_cast<const unitree_go::msg::dds_::LowState_*>(msg);
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
        const auto* s =
            static_cast<const unitree_go::msg::dds_::SportModeState_*>(msg);
        std::lock_guard<std::mutex> lk(state_mutex_);
        state_.pos[0] = s->position()[0];
        state_.pos[1] = s->position()[1];
        state_.pos[2] = s->position()[2];
        state_.vel[0] = s->velocity()[0];
        state_.vel[1] = s->velocity()[1];
        state_.vel[2] = s->velocity()[2];
        state_.valid  = true;
    }

    // ---- 50 Hz servo loop: sends last cached MPPI command ----
    void ControlLoop() {
        running_time_ += dt_;

        if (running_time_ < STANDUP_DURATION) {
            // Two-phase standup matching Unitree go2w_stand_example:
            //   0–1.5 s: current pose → crouch (thigh=1.36, calf=-2.65)
            //   1.5–3.0 s: crouch → stand (thigh=0.67, calf=-1.3)
            double alpha;
            const double* from;
            const double* to;
            if (running_time_ < STANDUP_DURATION * 0.5) {
                alpha = running_time_ / (STANDUP_DURATION * 0.5);
                from  = crouch_pos_;  // start: assume lying flat ≈ crouch target
                to    = crouch_pos_;
                // ramp from whatever into crouch; use phase against zero as from
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
                    low_cmd_.motor_cmd()[i].q()   =
                        (1.0 - alpha) * crouch_pos_[i] + alpha * stand_pos_[i];
                    low_cmd_.motor_cmd()[i].kp()  = 30.0 + alpha * 20.0;  // 30→50
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
            // Phase 2: send cached MPPI command at full 50 Hz
            double q[NUM_JOINTS], dq[NUM_JOINTS];
            {
                std::lock_guard<std::mutex> lk(cmd_mutex_);
                std::copy(cached_q_,  cached_q_  + NUM_JOINTS, q);
                std::copy(cached_dq_, cached_dq_ + NUM_JOINTS, dq);
            }
            for (int i = 0; i < NUM_LEG_JOINTS; ++i) {
                low_cmd_.motor_cmd()[i].q()   = q[i];
                low_cmd_.motor_cmd()[i].kp()  = 50.0;
                low_cmd_.motor_cmd()[i].dq()  = dq[i];
                low_cmd_.motor_cmd()[i].kd()  = 3.5;
                low_cmd_.motor_cmd()[i].tau() = 0.0;
            }
            for (int i = NUM_LEG_JOINTS; i < NUM_JOINTS; ++i) {
                low_cmd_.motor_cmd()[i].q()   = 0.0;
                low_cmd_.motor_cmd()[i].kp()  = 0.0;
                low_cmd_.motor_cmd()[i].dq()  = dq[i];
                low_cmd_.motor_cmd()[i].kd()  = 0.5;
                low_cmd_.motor_cmd()[i].tau() = 0.0;
            }
        }

        low_cmd_.crc() = crc32_core(
            reinterpret_cast<uint32_t*>(&low_cmd_),
            (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
        lowcmd_publisher_->Write(low_cmd_);
    }

    // ---- Background MPPI solver: runs as fast as it can ----
    void MPPILoop() {
        // Wait for stand-up to finish before starting MPPI
        while (running_time_ < STANDUP_DURATION) std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // Capture height target on first entry
        {
            std::lock_guard<std::mutex> lk(state_mutex_);
            mppi_.set_height_target(state_.pos[2]);
            std::cout << "MPPI solver started.  height target = "
                      << state_.pos[2] << " m\n";
        }

        // Seed cached command with standing pose
        {
            std::lock_guard<std::mutex> lk(cmd_mutex_);
            for (int j = 0; j < NUM_LEG_JOINTS; ++j) cached_q_[j] = stand_pos_[j];
            for (int j = NUM_LEG_JOINTS; j < NUM_JOINTS; ++j) cached_q_[j] = 0.0;
            std::fill(cached_dq_, cached_dq_ + NUM_JOINTS, 0.0);
        }

        int solve_count = 0;
        double solve_sum_ms = 0.0;

        while (true) {
            // Snapshot latest state
            RobotState state_snap;
            {
                std::lock_guard<std::mutex> lk(state_mutex_);
                state_snap = state_;
            }
            if (!state_snap.valid) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            auto t0 = std::chrono::steady_clock::now();
            double q_out[NUM_JOINTS], dq_out[NUM_JOINTS];
            mppi_.update(state_snap, q_out, dq_out);
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            {
                std::lock_guard<std::mutex> lk(cmd_mutex_);
                std::copy(q_out,  q_out  + NUM_JOINTS, cached_q_);
                std::copy(dq_out, dq_out + NUM_JOINTS, cached_dq_);
            }

            // Feed measured delay back so state prediction stays accurate
            mppi_.set_predict_delay(ms / 1000.0);

            solve_sum_ms += ms;
            ++solve_count;
            if (solve_count % 20 == 0)
                std::cout << "MPPI avg solve: "
                          << solve_sum_ms / solve_count << " ms  ("
                          << static_cast<int>(std::round(solve_sum_ms / solve_count / 20.0))
                          << " steps predicted)\n";
        }
    }

    static constexpr double dt_              = 0.02;
    static constexpr double STANDUP_DURATION = 3.0;

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

    RefFreeMPPI mppi_;

    // Shared state (DDS callbacks → MPPI solver)
    std::mutex  state_mutex_;
    RobotState  state_{};

    // Shared command (MPPI solver → servo loop)
    std::mutex cmd_mutex_;
    double cached_q_[NUM_JOINTS]  = {};
    double cached_dq_[NUM_JOINTS] = {};

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

    std::cout << "Reference-Free MPPI Controller — press Enter to start\n";
    std::cin.get();

    const std::string task = (argc >= 3) ? argv[2] : "stand";
    RefFreeMPPIController controller(task);
    controller.Init();

    while (true) sleep(10);
    return 0;
}
