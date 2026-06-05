#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
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
#include <mujoco/mujoco.h>

#include "control/quad_stand.h"

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

class QuadStandController {
public:
    explicit QuadStandController(const std::string& task      = "stand",
                                 const std::string& yaml_path = "../utils/tasks.yaml")
        : mppi_(task, yaml_path)
    {
        TaskConfig cfg = load_task(task, yaml_path);
        for (int j = 0; j < NUM_JOINTS; ++j) {
            kd_[j]        = cfg.muscle.kd_sim[j];
            stand_pos_[j] = cfg.nominal_pose[j];
        }

        // FK model — used to estimate base height from joint angles when
        // SportModeState is unavailable (e.g. robot in low-level control mode).
        char err[1000];
        fk_model_ = mj_loadXML(cfg.model_path.c_str(), nullptr, err, sizeof(err));
        if (!fk_model_)
            throw std::runtime_error("FK model load failed: " + std::string(err));
        fk_data_ = mj_makeData(fk_model_);

        for (int j = 0; j < NUM_JOINTS; ++j) {
            int jid = fk_model_->actuator_trnid[2 * (JOINT_OFFSET + j)];
            fk_qpos_adr_[j] = fk_model_->jnt_qposadr[jid];
        }

        // Foot bodies for height estimate: base z - lowest foot z = height above ground.
        const char* foot_names[] = {"FL_foot", "FR_foot", "RL_foot", "RR_foot"};
        fk_n_feet_ = 0;
        for (int i = 0; i < 4; ++i) {
            int bid = mj_name2id(fk_model_, mjOBJ_BODY, foot_names[i]);
            if (bid >= 0) fk_foot_ids_[fk_n_feet_++] = bid;
        }
    }

    ~QuadStandController() {
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
            std::bind(&QuadStandController::LowStateHandler, this,
                      std::placeholders::_1), 1);

        sportmode_subscriber_.reset(
            new ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>(TOPIC_SPORTMODE));
        sportmode_subscriber_->InitChannel(
            std::bind(&QuadStandController::SportModeHandler, this,
                      std::placeholders::_1), 1);

        control_thread_ = CreateRecurrentThreadEx(
            "qs_ctrl", UT_CPU_ID_NONE, 20000, &QuadStandController::ControlLoop, this);

        mppi_thread_ = std::thread(&QuadStandController::MPPILoop, this);
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
            state_.q[i]  = s->motor_state()[JOINT_OFFSET + i].q();
            state_.dq[i] = s->motor_state()[JOINT_OFFSET + i].dq();
        }
        state_.quat[0] = s->imu_state().quaternion()[0];
        state_.quat[1] = s->imu_state().quaternion()[1];
        state_.quat[2] = s->imu_state().quaternion()[2];
        state_.quat[3] = s->imu_state().quaternion()[3];
        state_.gyro[0] = s->imu_state().gyroscope()[0];
        state_.gyro[1] = s->imu_state().gyroscope()[1];
        state_.gyro[2] = s->imu_state().gyroscope()[2];

        // Estimate base height from FK when SportMode isn't publishing.
        // Place base at origin, compute foot positions, height = -lowest foot z.
        if (!sport_valid_ && fk_n_feet_ > 0) {
            fk_data_->qpos[0] = 0.0; fk_data_->qpos[1] = 0.0; fk_data_->qpos[2] = 0.0;
            fk_data_->qpos[3] = 1.0; fk_data_->qpos[4] = 0.0;
            fk_data_->qpos[5] = 0.0; fk_data_->qpos[6] = 0.0;
            for (int j = 0; j < NUM_JOINTS; ++j)
                fk_data_->qpos[fk_qpos_adr_[j]] = state_.q[j];
            mj_kinematics(fk_model_, fk_data_);

            double min_foot_z = std::numeric_limits<double>::infinity();
            for (int i = 0; i < fk_n_feet_; ++i)
                min_foot_z = std::min(min_foot_z, fk_data_->xpos[fk_foot_ids_[i] * 3 + 2]);
            state_.pos[2] = -min_foot_z;
        }

        state_.valid = true;
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
        sport_valid_  = true;
        state_.valid  = true;
    }

    void ControlLoop() {
        if (!mppi_ready_.load()) {
            running_time_ += 0.02;
            const double phase = std::tanh(running_time_ / 1.2);
            for (int i = 0; i < NUM_JOINTS; ++i) {
                low_cmd_.motor_cmd()[JOINT_OFFSET + i].q()   = phase * stand_pos_[i]
                                                              + (1.0 - phase) * stand_down_pos_[i];
                low_cmd_.motor_cmd()[JOINT_OFFSET + i].kp()  = phase * 50.0 + (1.0 - phase) * 20.0;
                low_cmd_.motor_cmd()[JOINT_OFFSET + i].dq()  = 0.0;
                low_cmd_.motor_cmd()[JOINT_OFFSET + i].kd()  = 3.5;
                low_cmd_.motor_cmd()[JOINT_OFFSET + i].tau() = 0.0;
            }
            if (!stand_up_complete_.load() && running_time_ >= 4.0) {
                stand_up_complete_.store(true);
                std::cout << "Stand-up complete — waiting for MPPI convergence.\n";
            }
        } else {
            double tau_cmd[NUM_JOINTS];
            {
                std::lock_guard<std::mutex> lk(cmd_mutex_);
                std::copy(cached_tau_, cached_tau_ + NUM_JOINTS, tau_cmd);
            }
            for (int i = 0; i < NUM_JOINTS; ++i) {
                low_cmd_.motor_cmd()[JOINT_OFFSET + i].q()   = PosStopF;
                low_cmd_.motor_cmd()[JOINT_OFFSET + i].kp()  = 0.0;
                low_cmd_.motor_cmd()[JOINT_OFFSET + i].dq()  = 0.0;
                low_cmd_.motor_cmd()[JOINT_OFFSET + i].kd()  = kd_[i];
                low_cmd_.motor_cmd()[JOINT_OFFSET + i].tau() = tau_cmd[i];
            }
        }

        low_cmd_.crc() = crc32_core(
            reinterpret_cast<uint32_t*>(&low_cmd_),
            (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
        lowcmd_publisher_->Write(low_cmd_);
    }

    void MPPILoop() {
        std::cout << "QuadStand MPPI started.\n";

        int    solve_count  = 0;
        double solve_sum_ms = 0.0;
        double peak_dq_     = 0.0;   // rolling peak since last print
        double peak_gyro_   = 0.0;

        while (true) {
            RobotState snap;
            {
                std::lock_guard<std::mutex> lk(state_mutex_);
                snap = state_;
            }
            if (!snap.valid || !stand_up_complete_.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            // Track peak motion regardless of print cadence.
            for (int j = 0; j < NUM_JOINTS; ++j)
                peak_dq_ = std::max(peak_dq_, std::abs(snap.dq[j]));
            const double gyro_mag = std::sqrt(snap.gyro[0]*snap.gyro[0]
                                            + snap.gyro[1]*snap.gyro[1]
                                            + snap.gyro[2]*snap.gyro[2]);
            peak_gyro_ = std::max(peak_gyro_, gyro_mag);

            auto t0 = std::chrono::steady_clock::now();
            double tau_cmd[NUM_JOINTS] = {};
            mppi_.update(snap, tau_cmd);
            double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();

            {
                std::lock_guard<std::mutex> lk(cmd_mutex_);
                std::copy(tau_cmd, tau_cmd + NUM_JOINTS, cached_tau_);
            }

            solve_sum_ms += ms;
            ++solve_count;

            if (solve_count % 5 == 0) {
                const double qw       = snap.quat[0];
                const double tilt_deg = 2.0 * std::acos(std::clamp(std::abs(qw), 0.0, 1.0))
                                        * 180.0 / M_PI;

                std::printf("solve %4d | %5.1f ms | cost min=%8.2f traj=%8.2f mean=%8.2f"
                            " | h=%5.3f m | tilt=%4.1f deg\n",
                            solve_count, solve_sum_ms / solve_count,
                            mppi_.best_cost(), mppi_.trajectory_cost(), mppi_.cost_mean(),
                            snap.pos[2], tilt_deg);

                peak_dq_   = 0.0;
                peak_gyro_ = 0.0;
            }

            // Reset solve count when stand-up completes so warm-start solves
            // from the actual standing state before handover.
            if (!stand_phase_done_ && stand_up_complete_.load()) {
                stand_phase_done_ = true;
                solve_count       = 0;
                solve_sum_ms      = 0.0;
                std::cout << "Standing — warming MPPI from standing state.\n";
            }

            if (!mppi_ready_.load() && stand_phase_done_
                    && solve_count >= CONVERGENCE_SOLVES) {
                std::cout << "MPPI warm — handing over to Hill torques after "
                          << solve_count << " solves (avg "
                          << solve_sum_ms / solve_count << " ms)\n";
                mppi_ready_.store(true);
            }
        }
    }

    static constexpr int CONVERGENCE_SOLVES = 40;

    double kd_[NUM_JOINTS]        = {};
    double stand_pos_[NUM_JOINTS] = {};

    // Resting pose from stand_go2.cpp — start of the tanh stand-up interpolation.
    const double stand_down_pos_[NUM_JOINTS] = {
         0.0473455,  1.22187, -2.44375,
        -0.0473455,  1.22187, -2.44375,
         0.0473455,  1.22187, -2.44375,
        -0.0473455,  1.22187, -2.44375,
    };

    double              running_time_     = 0.0;
    std::atomic<bool>   stand_up_complete_{false};
    bool                stand_phase_done_ = false;

    // FK model for base height estimation without SportModeState.
    mjModel* fk_model_             = nullptr;
    mjData*  fk_data_              = nullptr;
    int      fk_qpos_adr_[NUM_JOINTS] = {};
    int      fk_foot_ids_[4]       = {};
    int      fk_n_feet_            = 0;
    bool     sport_valid_          = false;

    std::atomic<bool> mppi_ready_{false};

    QuadStand mppi_;

    std::mutex state_mutex_;
    RobotState state_{};

    std::mutex cmd_mutex_;
    double     cached_tau_[NUM_JOINTS] = {};

    unitree_go::msg::dds_::LowCmd_ low_cmd_{};

    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_>          lowcmd_publisher_;
    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_>       lowstate_subscriber_;
    ChannelSubscriberPtr<unitree_go::msg::dds_::SportModeState_> sportmode_subscriber_;

    ThreadPtr   control_thread_;
    std::thread mppi_thread_;
};

int main(int argc, const char** argv)
{
    if (argc < 2)
        ChannelFactory::Instance()->Init(1, "lo");
    else
        ChannelFactory::Instance()->Init(1, argv[1]);

    const std::string task      = (argc >= 3) ? argv[2] : "stand";
    const std::string yaml_path = (argc >= 4) ? argv[3] : "../utils/tasks.yaml";

    std::cout << "QuadStand Controller (Hill muscle model)\n"
              << "  task: " << task << "\n"
              << "  yaml: " << yaml_path << "\n"
              << "Press Enter to start.\n";
    std::cin.get();

    QuadStandController controller(task, yaml_path);
    controller.Init();

    while (true) sleep(10);
    return 0;
}
