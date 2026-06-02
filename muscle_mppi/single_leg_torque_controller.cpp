#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>
#include <unistd.h>

#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/common/time/time_tool.hpp>
#include <unitree/common/thread/thread.hpp>

#include "control/single_leg_torque.h"

using namespace unitree::common;
using namespace unitree::robot;

#define TOPIC_LOWCMD   "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"

constexpr double PosStopF = 2.146E+9f;
constexpr double VelStopF = 16000.0f;

static const double NOMINAL_POSE[12] = {
    0.0, 0.67, -1.3,   // FR
    0.0, 0.67, -1.3,   // FL  ← torque-controlled by MPPI
    0.0, 0.67, -1.3,   // RR
    0.0, 0.67, -1.3,   // RL
};
static const double KP_HOLD = 40.0;
static const double KD_HOLD = 3.5;

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

class TorqueController {
public:
    explicit TorqueController(const std::string& task      = "reach_torque",
                              const std::string& yaml_path = "../utils/tasks.yaml")
        : mppi_(task, yaml_path)
    {
        TaskConfig cfg = load_task(task, yaml_path);

        for (int j = 0; j < NUM_JOINTS; ++j)
            kd_fl_[j] = cfg.muscle.kd_sim[j];

        for (int k = 0; k < 3; ++k)
            foot_target_[k] = cfg.foot_target[k];

        // Lightweight FK model — only used for mj_kinematics to get xpos.
        char err[1000];
        fk_model_ = mj_loadXML(cfg.model_path.c_str(), nullptr, err, sizeof(err));
        fk_data_  = mj_makeData(fk_model_);
        for (int j = 0; j < NUM_JOINTS; ++j) {
            int jid = fk_model_->actuator_trnid[2 * (JOINT_OFFSET + j)];
            fk_qpos_adr_[j] = fk_model_->jnt_qposadr[jid];
        }
        foot_bid_ = mj_name2id(fk_model_, mjOBJ_BODY, "FL_foot");
    }

    ~TorqueController() {
        mj_deleteData(fk_data_);
        mj_deleteModel(fk_model_);
    }

    void Init() {
        InitLowCmd();

        lowcmd_publisher_.reset(
            new ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
        lowcmd_publisher_->InitChannel();

        lowstate_subscriber_.reset(
            new ChannelSubscriber<unitree_go::msg::dds_::LowState_>(TOPIC_LOWSTATE));
        lowstate_subscriber_->InitChannel(
            std::bind(&TorqueController::LowStateHandler, this,
                      std::placeholders::_1), 1);

        control_thread_ = CreateRecurrentThreadEx(
            "torque_ctrl", UT_CPU_ID_NONE, 2000, &TorqueController::ControlLoop, this);

        mppi_thread_ = std::thread(&TorqueController::MPPILoop, this);
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
        for (int i = 0; i < 12; ++i) {
            full_q_[i]  = s->motor_state()[i].q();
            full_dq_[i] = s->motor_state()[i].dq();
        }
        for (int j = 0; j < NUM_JOINTS; ++j) {
            state_.q[j]  = full_q_[JOINT_OFFSET + j];
            state_.dq[j] = full_dq_[JOINT_OFFSET + j];
        }
        state_.valid = true;
    }

    void ControlLoop() {
        double tau_cmd[NUM_JOINTS];
        {
            std::lock_guard<std::mutex> lk(cmd_mutex_);
            std::copy(cached_tau_, cached_tau_ + NUM_JOINTS, tau_cmd);
        }
        double q_snap[12], dq_snap[12];
        {
            std::lock_guard<std::mutex> lk(state_mutex_);
            std::copy(full_q_,  full_q_  + 12, q_snap);
            std::copy(full_dq_, full_dq_ + 12, dq_snap);
        }

        for (int i = 0; i < 12; ++i) {
            if (i >= JOINT_OFFSET && i < JOINT_OFFSET + NUM_JOINTS) {
                int j = i - JOINT_OFFSET;
                low_cmd_.motor_cmd()[i].q()   = PosStopF;
                low_cmd_.motor_cmd()[i].kp()  = 0.0;
                low_cmd_.motor_cmd()[i].dq()  = 0.0;
                low_cmd_.motor_cmd()[i].kd()  = kd_fl_[j];
                low_cmd_.motor_cmd()[i].tau() = mppi_ready_.load() ? tau_cmd[j] : 0.0;
            } else {
                low_cmd_.motor_cmd()[i].q()   = NOMINAL_POSE[i];
                low_cmd_.motor_cmd()[i].kp()  = KP_HOLD;
                low_cmd_.motor_cmd()[i].dq()  = 0.0;
                low_cmd_.motor_cmd()[i].kd()  = KD_HOLD;
                low_cmd_.motor_cmd()[i].tau() = 0.0;
            }
        }

        low_cmd_.crc() = crc32_core(
            reinterpret_cast<uint32_t*>(&low_cmd_),
            (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
        lowcmd_publisher_->Write(low_cmd_);
    }

    void MPPILoop() {
        std::cout << "Waiting for robot state...\n";
        while (true) {
            { std::lock_guard<std::mutex> lk(state_mutex_); if (state_.valid) break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        std::cout << "State received — running torque-space MPPI.\n\n";

        int solve_count = 0;
        double solve_sum_ms = 0.0;

        while (true) {
            RobotState snap;
            { std::lock_guard<std::mutex> lk(state_mutex_); snap = state_; }

            auto t0 = std::chrono::steady_clock::now();
            double tau[NUM_JOINTS] = {};
            mppi_.update(snap, tau);
            double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();

            { std::lock_guard<std::mutex> lk(cmd_mutex_);
              std::copy(tau, tau + NUM_JOINTS, cached_tau_); }

            solve_sum_ms += ms;
            ++solve_count;
            if (solve_count % 20 == 0) {
                // FK: set joint positions and compute xpos only.
                for (int j = 0; j < NUM_JOINTS; ++j)
                    fk_data_->qpos[fk_qpos_adr_[j]] = snap.q[j];
                mj_kinematics(fk_model_, fk_data_);
                const double* fp = fk_data_->xpos + 3 * foot_bid_;
                double dx = fp[0]-foot_target_[0],
                       dy = fp[1]-foot_target_[1],
                       dz = fp[2]-foot_target_[2];
                double err = std::sqrt(dx*dx + dy*dy + dz*dz);
                std::printf("solve %4d | %.1f ms | cost min=%7.3f mean=%7.3f"
                            " | foot [%6.3f %6.3f %6.3f] | err %.4f m\n",
                            solve_count, solve_sum_ms / solve_count,
                            mppi_.cost_min(), mppi_.cost_mean(),
                            fp[0], fp[1], fp[2], err);
            }

            if (!mppi_ready_.load() && solve_count >= 10) {
                std::cout << "Warm — handing over torques to FL leg.\n";
                mppi_ready_.store(true);
            }
        }
    }

    std::atomic<bool> mppi_ready_{false};

    SingleLegTorque mppi_;
    double kd_fl_[NUM_JOINTS]    = {};
    double foot_target_[3]       = {};
    mjModel* fk_model_           = nullptr;
    mjData*  fk_data_            = nullptr;
    int      fk_qpos_adr_[NUM_JOINTS] = {};
    int      foot_bid_           = -1;

    std::mutex state_mutex_;
    RobotState state_{};
    double full_q_[12]  = {};
    double full_dq_[12] = {};

    std::mutex cmd_mutex_;
    double cached_tau_[NUM_JOINTS] = {};

    unitree_go::msg::dds_::LowCmd_ low_cmd_{};
    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_>    lowcmd_publisher_;
    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber_;

    ThreadPtr   control_thread_;
    std::thread mppi_thread_;
};

int main(int argc, const char** argv)
{
    if (argc < 2)
        ChannelFactory::Instance()->Init(1, "lo");
    else
        ChannelFactory::Instance()->Init(1, argv[1]);

    const std::string task      = (argc >= 3) ? argv[2] : "reach_torque";
    const std::string yaml_path = (argc >= 4) ? argv[3] : "../utils/tasks.yaml";

    std::cout << "Stage-1 torque-space MPPI controller\n"
              << "  task: " << task << "\n"
              << "  yaml: " << yaml_path << "\n"
              << "Press Enter to start.\n";
    std::cin.get();

    TorqueController ctrl(task, yaml_path);
    ctrl.Init();

    while (true) sleep(10);
    return 0;
}
