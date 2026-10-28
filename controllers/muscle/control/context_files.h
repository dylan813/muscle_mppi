#pragma once

// What one solve planned from (RolloutContext, mppi_locomotion.h) and the plan
// it chose, one row per solve, as float64 .npy files: mppi_sim --contexts
// writes them, landscape_scorer reads a row back and rolls plans out from it
// exactly as that solve's update() did.
//
//   <prefix>_state.npy       (N, 37)                    RobotState: pos 3, quat 4, vel 3, gyro 3, q 12, dq 12
//   <prefix>_activation.npy  (N, NUM_MUSCLES)           filtered activations
//   <prefix>_motor.npy       (N, NUM_JOINTS + 1)        MotorState: tau, then step
//   <prefix>_gait_ref.npy    (N, horizon, NUM_MUSCLES)  gait reference per horizon step
//   <prefix>_goal.npy        (N, 9)                     goal_quat 4, command goal_pos 3, vx, vy
//   <prefix>_plan.npy        (N, horizon, NUM_MUSCLES)  the plan update() chose

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "mppi_locomotion.h"
#include "../../common/npy_reader.h"
#include "../../common/npy_writer.h"

constexpr int kContextStateSize = 3 + 4 + 3 + 3 + 2 * NUM_JOINTS;
constexpr int kContextMotorSize = NUM_JOINTS + 1;
constexpr int kContextGoalSize  = 4 + 3 + 2;

inline const std::vector<std::string>& context_file_suffixes()
{
    static const std::vector<std::string> suffixes = {
        "_state.npy", "_activation.npy", "_motor.npy", "_gait_ref.npy", "_goal.npy", "_plan.npy"};
    return suffixes;
}

class ContextWriter {
public:
    ContextWriter(const std::string& prefix, int horizon)
        : state_(prefix + "_state.npy", {kContextStateSize}, NpyDtype::Float64),
          activation_(prefix + "_activation.npy", {NUM_MUSCLES}, NpyDtype::Float64),
          motor_(prefix + "_motor.npy", {kContextMotorSize}, NpyDtype::Float64),
          gait_ref_(prefix + "_gait_ref.npy", {horizon, NUM_MUSCLES}, NpyDtype::Float64),
          goal_(prefix + "_goal.npy", {kContextGoalSize}, NpyDtype::Float64),
          plan_(prefix + "_plan.npy", {horizon, NUM_MUSCLES}, NpyDtype::Float64) {}

    void append(const RolloutContext& c, const std::vector<double>& plan)
    {
        const RobotState& s = c.state;
        std::vector<double> state;
        auto add = [&state](const double* values, int n) { state.insert(state.end(), values, values + n); };
        add(s.pos, 3);
        add(s.quat, 4);
        add(s.vel, 3);
        add(s.gyro, 3);
        add(s.q, NUM_JOINTS);
        add(s.dq, NUM_JOINTS);

        std::vector<double> motor(c.motor.tau, c.motor.tau + NUM_JOINTS);
        motor.push_back(c.motor.step);

        std::vector<double> goal(c.goal_quat, c.goal_quat + 4);
        goal.insert(goal.end(), c.command.goal_pos, c.command.goal_pos + 3);
        goal.push_back(c.command.vx);
        goal.push_back(c.command.vy);

        state_.append(state.data());
        activation_.append(c.activation);
        motor_.append(motor.data());
        gait_ref_.append(c.gait_ref.data());
        goal_.append(goal.data());
        plan_.append(plan.data());
    }

private:
    NpyWriter state_, activation_, motor_, gait_ref_, goal_, plan_;
};

// Row `row` of the files at prefix: the context it planned from, and in plan
// the plan it chose. Throws if a file is missing, the row doesn't exist, or a
// file's shape doesn't match horizon.
inline RolloutContext read_context(const std::string& prefix, int row, int horizon, std::vector<double>& plan)
{
    // The row's values from one file, after checking its shape.
    auto row_of = [&](const std::string& suffix, const std::vector<int>& row_shape) {
        const NpyArray a = read_npy(prefix + suffix);
        std::vector<int> expected = {a.shape.empty() ? 0 : a.shape[0]};
        expected.insert(expected.end(), row_shape.begin(), row_shape.end());
        if (a.shape != expected)
            throw std::runtime_error("read_context: unexpected shape in " + prefix + suffix);
        if (row < 0 || row >= a.shape[0])
            throw std::runtime_error("read_context: no row " + std::to_string(row) + " in " + prefix + suffix
                                     + " (" + std::to_string(a.shape[0]) + " rows)");
        size_t size = 1;
        for (int d : row_shape) size *= static_cast<size_t>(d);
        const auto first = a.values.begin() + static_cast<long>(row * size);
        return std::vector<double>(first, first + static_cast<long>(size));
    };

    RolloutContext c;

    const std::vector<double> state = row_of("_state.npy", {kContextStateSize});
    const double* next = state.data();
    auto take = [&next](double* values, int n) { std::copy(next, next + n, values); next += n; };
    take(c.state.pos, 3);
    take(c.state.quat, 4);
    take(c.state.vel, 3);
    take(c.state.gyro, 3);
    take(c.state.q, NUM_JOINTS);
    take(c.state.dq, NUM_JOINTS);
    c.state.valid = true;

    const std::vector<double> activation = row_of("_activation.npy", {NUM_MUSCLES});
    std::copy(activation.begin(), activation.end(), c.activation);

    const std::vector<double> motor = row_of("_motor.npy", {kContextMotorSize});
    std::copy(motor.begin(), motor.begin() + NUM_JOINTS, c.motor.tau);
    c.motor.step = static_cast<int>(motor[NUM_JOINTS]);

    c.gait_ref = row_of("_gait_ref.npy", {horizon, NUM_MUSCLES});

    const std::vector<double> goal = row_of("_goal.npy", {kContextGoalSize});
    std::copy(goal.begin(), goal.begin() + 4, c.goal_quat);
    std::copy(goal.begin() + 4, goal.begin() + 7, c.command.goal_pos);
    c.command.vx = goal[7];
    c.command.vy = goal[8];

    plan = row_of("_plan.npy", {horizon, NUM_MUSCLES});
    return c;
}
