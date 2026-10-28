#pragma once

// What one solve planned from (RolloutContext, mppi_locomotion.h) and the plan
// it chose, one row per solve, as float64 .npy files: mppi_sim --contexts
// writes them (ContextWriter), landscape_scorer reads rows back (ContextReader)
// and rolls plans out from them exactly as those solves' update() did.
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

// The files at prefix, loaded once; read() gives any row's context and the
// plan it chose. The constructor throws if a file is missing, or its shape
// doesn't match horizon or the other files' row count.
class ContextReader {
public:
    ContextReader(const std::string& prefix, int horizon)
        : state_(load(prefix, "_state.npy", {kContextStateSize})),
          activation_(load(prefix, "_activation.npy", {NUM_MUSCLES})),
          motor_(load(prefix, "_motor.npy", {kContextMotorSize})),
          gait_ref_(load(prefix, "_gait_ref.npy", {horizon, NUM_MUSCLES})),
          goal_(load(prefix, "_goal.npy", {kContextGoalSize})),
          plan_(load(prefix, "_plan.npy", {horizon, NUM_MUSCLES}))
    {
        for (const NpyArray* a : {&activation_, &motor_, &gait_ref_, &goal_, &plan_})
            if (a->shape[0] != rows())
                throw std::runtime_error("ContextReader: the files at " + prefix + " hold different row counts");
    }

    int rows() const { return state_.shape[0]; }

    // Row `row`'s context, and in plan the plan it chose. Throws if there's no such row.
    RolloutContext read(int row, std::vector<double>& plan) const;

private:
    // The file at prefix + suffix, after checking its rows are row_shape.
    static NpyArray load(const std::string& prefix, const std::string& suffix, const std::vector<int>& row_shape)
    {
        NpyArray a = read_npy(prefix + suffix);
        std::vector<int> expected = {a.shape.empty() ? 0 : a.shape[0]};
        expected.insert(expected.end(), row_shape.begin(), row_shape.end());
        if (a.shape != expected)
            throw std::runtime_error("ContextReader: unexpected shape in " + prefix + suffix);
        return a;
    }

    // Values in one row of file a, and where row `row` starts (load() checked
    // every row has the shape the layout above says).
    static size_t row_size(const NpyArray& a) { return a.values.size() / static_cast<size_t>(a.shape[0]); }
    static const double* row_start(const NpyArray& a, int row)
    {
        return a.values.data() + static_cast<size_t>(row) * row_size(a);
    }

    NpyArray state_, activation_, motor_, gait_ref_, goal_, plan_;
};

inline RolloutContext ContextReader::read(int row, std::vector<double>& plan) const
{
    if (row < 0 || row >= rows())
        throw std::runtime_error("ContextReader: no row " + std::to_string(row) + " (" + std::to_string(rows())
                                 + " rows)");

    RolloutContext c;

    const double* next = row_start(state_, row);
    auto take = [&next](double* values, int n) { std::copy(next, next + n, values); next += n; };
    take(c.state.pos, 3);
    take(c.state.quat, 4);
    take(c.state.vel, 3);
    take(c.state.gyro, 3);
    take(c.state.q, NUM_JOINTS);
    take(c.state.dq, NUM_JOINTS);
    c.state.valid = true;

    const double* activation = row_start(activation_, row);
    std::copy(activation, activation + NUM_MUSCLES, c.activation);

    const double* motor = row_start(motor_, row);
    std::copy(motor, motor + NUM_JOINTS, c.motor.tau);
    c.motor.step = static_cast<int>(motor[NUM_JOINTS]);

    const double* gait_ref = row_start(gait_ref_, row);
    c.gait_ref.assign(gait_ref, gait_ref + row_size(gait_ref_));

    const double* goal = row_start(goal_, row);
    std::copy(goal, goal + 4, c.goal_quat);
    std::copy(goal + 4, goal + 7, c.command.goal_pos);
    c.command.vx = goal[7];
    c.command.vy = goal[8];

    const double* chosen = row_start(plan_, row);
    plan.assign(chosen, chosen + row_size(plan_));
    return c;
}
