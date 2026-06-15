// Gait activation playback test.
// Mirrors mppi_controller.cpp structure: same DDS setup, same stand-up PD,
// same torque mode — replacing MPPI with a fixed activation gait cycle.
//
// Terminal 1: cd muscle_mppi/unitree_mujoco/simulate/build && ./unitree_mujoco -r go2 -s scene.xml
// Terminal 2 (from muscle_mppi/muscle_mppi/):
//   ../analysis/unit_tests/test_gait/build/test_gait <gait.tsv> [yaml] [net_interface]

#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/common/thread/thread.hpp>
#include <unitree/common/time/time_tool.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "muscle.h"
#include "tasks.h"

using namespace unitree::common;
using namespace unitree::robot;

#define TOPIC_LOWCMD   "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"

static constexpr double PosStopF = 2.146E+9;
static constexpr double VelStopF = 16000.0;

// Tight crouch: symmetric, low CoM, all feet planted before extending.
static const double CROUCH_POS[NUM_JOINTS] = {
    0.0, 1.2, -2.6,   0.0, 1.2, -2.6,
    0.0, 1.2, -2.6,   0.0, 1.2, -2.6
};
// Stand-up target — same as mppi_controller.cpp: crouched intermediate that
// the robot can reliably reach from prone without falling backwards.
// The Hill model takes over from here once gait_ready_ is set.
static const double STAND_POS[NUM_JOINTS] = {
    0.0, 0.67, -1.3,   0.0, 0.67, -1.3,
    0.0, 0.67, -1.3,   0.0, 0.67, -1.3
};

// Velocity damping alongside Hill torques — matches mppi_controller.cpp kd_[]
static const double KD_GAIT[NUM_JOINTS] = {
    2.0, 3.5, 3.5,   2.0, 3.5, 3.5,
    2.0, 3.5, 3.5,   2.0, 3.5, 3.5
};

// Two-phase stand-up (500 Hz loop):
//   Phase 1: curl to tight crouch so all feet are planted symmetrically (2 s)
//   Phase 2: extend to nominal pose (2 s)
static constexpr int CROUCH_TICKS  = 1000;   // 2 s curl-down
static constexpr int STANDUP_TICKS = 2000;   // 2 s additional extension

// Gait phase advances at 100 Hz. Loop at 500 Hz → advance every 5 ticks.
static constexpr int GAIT_TICKS_PER_PHASE = 5;

// --------------------------------------------------------------------------
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

// --------------------------------------------------------------------------
// Gait TSV loader: 24 rows × N cols, rows 0-11 = a1, rows 12-23 = a2
// --------------------------------------------------------------------------
static std::vector<std::vector<double>> load_gait(const std::string& path)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<double> row;
        std::istringstream ss(line);
        double v;
        while (ss >> v) row.push_back(v);
        if (!row.empty()) rows.push_back(std::move(row));
    }
    if ((int)rows.size() != 2 * NUM_JOINTS)
        throw std::runtime_error("Expected " + std::to_string(2 * NUM_JOINTS)
                                 + " rows, got " + std::to_string(rows.size()));
    size_t n = rows[0].size();
    for (auto& r : rows)
        if (r.size() != n) throw std::runtime_error("Inconsistent row lengths");
    return rows;
}

static void fill_act_cmd(const std::vector<std::vector<double>>& g, int phase,
                         double act_cmd[NUM_MUSCLES])
{
    for (int j = 0; j < NUM_JOINTS; ++j) {
        act_cmd[2 * j]     = g[j][phase];             // a1 agonist
        act_cmd[2 * j + 1] = g[NUM_JOINTS + j][phase]; // a2 antagonist
    }
}

// --------------------------------------------------------------------------
class GaitTestController {
public:
    GaitTestController(std::vector<std::vector<double>> gait, const TaskConfig& task)
        : gait_(std::move(gait)), muscle_(task.muscle), n_phases_((int)gait_[0].size())
    {
        for (int i = 0; i < NUM_MUSCLES; ++i) activation_[i] = 0.3;
    }

    void Init() {
        InitLowCmd();

        lowcmd_pub_.reset(
            new ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
        lowcmd_pub_->InitChannel();

        lowstate_sub_.reset(
            new ChannelSubscriber<unitree_go::msg::dds_::LowState_>(TOPIC_LOWSTATE));
        lowstate_sub_->InitChannel(
            std::bind(&GaitTestController::LowStateHandler, this, std::placeholders::_1), 1);

        // 500 Hz control loop — matches PHYSICS_DT = 0.002 s for activation filter
        control_thread_ = CreateRecurrentThreadEx(
            "gait_ctrl", UT_CPU_ID_NONE, 2000, &GaitTestController::ControlLoop, this);
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
        for (int j = 0; j < NUM_JOINTS; ++j) {
            q_[j]  = s->motor_state()[JOINT_OFFSET + j].q();
            dq_[j] = s->motor_state()[JOINT_OFFSET + j].dq();
        }
        state_valid_ = true;
    }

    void ControlLoop() {
        if (!gait_ready_) {
            // PD stand-up: hold STAND_POS until settled — mirrors mppi_controller.cpp
            for (int j = 0; j < NUM_JOINTS; ++j) {
                low_cmd_.motor_cmd()[JOINT_OFFSET + j].q()   = STAND_POS[j];
                low_cmd_.motor_cmd()[JOINT_OFFSET + j].kp()  = 50.0;
                low_cmd_.motor_cmd()[JOINT_OFFSET + j].dq()  = 0.0;
                low_cmd_.motor_cmd()[JOINT_OFFSET + j].kd()  = 3.5;
                low_cmd_.motor_cmd()[JOINT_OFFSET + j].tau() = 0.0;
            }

            if (++standup_tick_ >= STANDUP_TICKS) {
                gait_ready_ = true;
                std::cout << "Stand-up complete — switching to Hill model gait\n";
            }
        } else {
            // Gait playback via Hill muscle model
            double q[NUM_JOINTS], dq[NUM_JOINTS];
            {
                std::lock_guard<std::mutex> lk(state_mutex_);
                if (!state_valid_) return;
                std::copy(q_, q_ + NUM_JOINTS, q);
                std::copy(dq_, dq_ + NUM_JOINTS, dq);
            }

            double act_cmd[NUM_MUSCLES];
            fill_act_cmd(gait_, phase_, act_cmd);

            double tau[NUM_JOINTS] = {};
            hill_compute_torques(act_cmd, q, dq, muscle_, 0.002, activation_, tau);

            for (int j = 0; j < NUM_JOINTS; ++j) {
                low_cmd_.motor_cmd()[JOINT_OFFSET + j].q()   = PosStopF;
                low_cmd_.motor_cmd()[JOINT_OFFSET + j].kp()  = 0.0;
                low_cmd_.motor_cmd()[JOINT_OFFSET + j].dq()  = 0.0;
                low_cmd_.motor_cmd()[JOINT_OFFSET + j].kd()  = KD_GAIT[j];
                low_cmd_.motor_cmd()[JOINT_OFFSET + j].tau() = tau[j];
            }

            // Advance gait phase at 100 Hz (every 5 ticks at 500 Hz)
            if (++phase_tick_ >= GAIT_TICKS_PER_PHASE) {
                phase_tick_ = 0;
                phase_ = (phase_ + 1) % n_phases_;
                if (phase_ == 0)
                    printf("Gait cycle complete\n");
            }
        }

        low_cmd_.crc() = crc32_core(
            reinterpret_cast<uint32_t*>(&low_cmd_),
            (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
        lowcmd_pub_->Write(low_cmd_);
    }

    std::vector<std::vector<double>> gait_;
    const MuscleParams               muscle_;
    const int                        n_phases_;

    std::mutex state_mutex_;
    double     q_[NUM_JOINTS] = {}, dq_[NUM_JOINTS] = {};
    bool       state_valid_ = false;

    double     activation_[NUM_MUSCLES];

    bool       gait_ready_   = false;
    int        standup_tick_ = 0;
    int        phase_        = 0;
    int        phase_tick_   = 0;

    unitree_go::msg::dds_::LowCmd_ low_cmd_{};
    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_>    lowcmd_pub_;
    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_sub_;
    ThreadPtr control_thread_;
};

// --------------------------------------------------------------------------
int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <activation_gait.tsv> [tasks.yaml] [net_interface]\n",
                argv[0]);
        return 1;
    }
    const std::string gait_path = argv[1];
    const std::string yaml_path = (argc >= 3) ? argv[2] : "utils/tasks.yaml";
    const std::string net_iface = (argc >= 4) ? argv[3] : "lo";

    TaskConfig task;
    try { task = load_task("walk", yaml_path); }
    catch (const std::exception& e) { fprintf(stderr, "tasks.yaml: %s\n", e.what()); return 1; }

    std::vector<std::vector<double>> gait;
    try { gait = load_gait(gait_path); }
    catch (const std::exception& e) { fprintf(stderr, "gait: %s\n", e.what()); return 1; }

    printf("Gait:      %s  (%d phases)\n", gait_path.c_str(), (int)gait[0].size());
    printf("Interface: %s\n", net_iface.c_str());
    printf("Stand-up for %.0f s, then Hill model gait.\n",
           STANDUP_TICKS / 500.0);
    printf("Press Enter to start...\n");
    std::cin.get();

    ChannelFactory::Instance()->Init(1, net_iface.c_str());

    GaitTestController controller(std::move(gait), task);
    controller.Init();

    while (true) sleep(10);
    return 0;
}
