#include "task_config.h"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <stdexcept>
#include <utility>


void load_doubles(const YAML::Node& node, double* dst, int n, const std::string& field)
{
    if (!node || !node.IsSequence() || static_cast<int>(node.size()) != n)
        throw std::runtime_error("Field '" + field + "': expected sequence of length "
                                 + std::to_string(n));
    for (int i = 0; i < n; ++i)
        dst[i] = node[i].as<double>();
}

YAML::Node load_task_node(const std::string& task_name, const std::string& yaml_path)
{
    YAML::Node root;
    try {
        root = YAML::LoadFile(yaml_path);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("load_task: cannot parse '" + yaml_path + "': " + e.what());
    }

    if (!root[task_name])
        throw std::runtime_error("load_task: task '" + task_name
                                 + "' not found in " + yaml_path);

    return root[task_name];
}

// Optional `motor:` block (MotorParams). Unknown keys throw, so a typo can't
// silently run the joints without the motor.
static void load_motor(const YAML::Node& node, double dt, MotorParams& motor)
{
    motor.physics_dt = dt;
    if (!node) return;

    for (const auto& kv : node) {
        const std::string key = kv.first.as<std::string>();
        if (key != "lag" && key != "bandwidth" && key != "update_rate" && key != "driver_kd"
            && key != "physics_dt" && key != "damping")
            throw std::runtime_error("Field 'motor': unknown key '" + key
                                     + "' (expected lag, bandwidth, update_rate, driver_kd, physics_dt, damping)");
    }

    if (node["lag"]) {
        const std::string lag = node["lag"].as<std::string>();
        if      (lag == "first_order")     motor.lag = MotorLag::FirstOrder;
        else if (lag == "zero_order_hold") motor.lag = MotorLag::ZeroOrderHold;
        else throw std::runtime_error("Field 'motor.lag': expected first_order or zero_order_hold, got '" + lag + "'");
    }

    // Each lag has its own rate key; the other one would be silently ignored.
    const bool hold = motor.lag == MotorLag::ZeroOrderHold;
    if (hold && node["bandwidth"])
        throw std::runtime_error("Field 'motor.bandwidth': first_order only; zero_order_hold takes update_rate");
    if (!hold && node["update_rate"])
        throw std::runtime_error("Field 'motor.update_rate': zero_order_hold only; first_order takes bandwidth");
    if (hold && !node["update_rate"])
        throw std::runtime_error("Field 'motor.update_rate': required for zero_order_hold");

    if (node["bandwidth"]) {
        motor.bandwidth = node["bandwidth"].as<double>();
        if (!(motor.bandwidth > 0.0))
            throw std::runtime_error("Field 'motor.bandwidth': expected > 0 (.inf for no lag)");
    }

    if (node["driver_kd"])
        load_doubles(node["driver_kd"], motor.driver_kd, NUM_JOINTS, "motor.driver_kd");

    if (node["physics_dt"]) {
        const double physics_dt = node["physics_dt"].as<double>();
        const double ratio      = dt / physics_dt;
        motor.substeps = static_cast<int>(std::lround(ratio));
        if (!(physics_dt > 0.0) || motor.substeps < 1 || std::abs(ratio - motor.substeps) > 1e-9)
            throw std::runtime_error("Field 'motor.physics_dt': expected dt divided by a whole number of steps, got "
                                     + std::to_string(physics_dt));
        motor.physics_dt = dt / motor.substeps;
    }

    if (node["damping"]) {
        const std::string damping = node["damping"].as<std::string>();
        if      (damping == "explicit") motor.damping = MotorDamping::Explicit;
        else if (damping == "implicit") motor.damping = MotorDamping::Implicit;
        else throw std::runtime_error("Field 'motor.damping': expected explicit or implicit, got '" + damping + "'");
    }

    if (!hold) return;

    // The hold samples the whole command, velocity feedback included, so there
    // is no share left for MuJoCo to integrate implicitly.
    if (motor.damping == MotorDamping::Implicit)
        throw std::runtime_error("Field 'motor.damping': zero_order_hold needs explicit damping");

    motor.update_rate = node["update_rate"].as<double>();
    const double steps = 1.0 / (motor.update_rate * motor.physics_dt);
    motor.hold_steps = static_cast<int>(std::lround(steps));
    if (!(motor.update_rate > 0.0) || motor.hold_steps < 1 || std::abs(steps - motor.hold_steps) > 1e-9)
        throw std::runtime_error("Field 'motor.update_rate': expected 1/(a whole number of physics steps), i.e. "
                                 + std::to_string(1.0 / motor.physics_dt) + " Hz divided by a whole number, got "
                                 + std::to_string(motor.update_rate));
}

// Optional `push:` block (PushParams). Unknown keys throw, as for `motor:`.
static std::optional<PushParams> load_push(const YAML::Node& node, double dt)
{
    if (!node) return std::nullopt;

    for (const auto& kv : node) {
        const std::string key = kv.first.as<std::string>();
        if (key != "interval" && key != "max_vel_xy")
            throw std::runtime_error("Field 'push': unknown key '" + key + "' (expected interval, max_vel_xy)");
    }
    if (!node["interval"] || !node["max_vel_xy"])
        throw std::runtime_error("Field 'push': needs interval and max_vel_xy");

    PushParams push;
    push.interval = node["interval"].as<double>();
    const double steps = push.interval / dt;
    push.steps = static_cast<int>(std::lround(steps));
    if (!(push.interval > 0.0) || push.steps < 1 || std::abs(steps - push.steps) > 1e-9)
        throw std::runtime_error("Field 'push.interval': expected a whole number of control steps (dt), got "
                                 + std::to_string(push.interval));

    push.max_vel_xy = node["max_vel_xy"].as<double>();
    if (!(push.max_vel_xy > 0.0))
        throw std::runtime_error("Field 'push.max_vel_xy': expected > 0");

    return push;
}

static TaskEnd load_end_on(const YAML::Node& node)
{
    if (!node) return TaskEnd::Success;

    const std::string end_on = node.as<std::string>();
    if (end_on == "success")  return TaskEnd::Success;
    if (end_on == "duration") return TaskEnd::Duration;
    throw std::runtime_error("Field 'end_on': expected success or duration, got '" + end_on + "'");
}

void load_task_base(const YAML::Node& t, TaskConfigBase& cfg)
{
    cfg.model_path    = repo_path(t["model_path"].as<std::string>());
    cfg.n_samples     = t["n_samples"].as<int>();
    cfg.horizon       = t["horizon"].as<int>();
    cfg.lambda        = t["lambda"].as<double>();
    cfg.dt            = t["dt"].as<double>();
    cfg.sample_type   = t["sample_type"] ? t["sample_type"].as<std::string>() : "normal";
    cfg.n_knots       = t["n_knots"]     ? t["n_knots"].as<int>()             : 4;
    cfg.num_threads   = t["num_threads"] ? t["num_threads"].as<int>()        : 0;
    cfg.sim_duration  = t["sim_duration"] ? t["sim_duration"].as<double>()   : 10.0;
    cfg.end_on        = load_end_on(t["end_on"]);
    cfg.spawn_height_offset =
        t["spawn_height_offset"] ? t["spawn_height_offset"].as<double>()    : 0.0;

    if (t["noise_sigma_act"])
        load_doubles(t["noise_sigma_act"], cfg.noise_sigma_act, NUM_JOINTS, "noise_sigma_act");

    load_motor(t["motor"], cfg.dt, cfg.motor);
    cfg.push = load_push(t["push"], cfg.dt);

    if (t["phases"]) {
        const YAML::Node& phases = t["phases"];
        if (!phases.IsSequence())
            throw std::runtime_error("Field 'phases': expected a sequence");
        for (const auto& p : phases) {
            TaskPhase phase;
            load_doubles(p["goal_pos"], phase.goal_pos, 3, "phases[].goal_pos");
            if (p["cmd_vel"])
                load_doubles(p["cmd_vel"], phase.cmd_vel, 2, "phases[].cmd_vel");
            phase.desired_gait = p["desired_gait"] ? p["desired_gait"].as<std::string>() : "";
            phase.gait_path    = p["gait_path"]    ? repo_path(p["gait_path"].as<std::string>()) : "";
            if (phase.desired_gait.empty() && phase.gait_path.empty())
                throw std::runtime_error("phases[]: needs desired_gait or gait_path");
            phase.goal_thresh  = p["goal_thresh"]  ? p["goal_thresh"].as<double>()  : 0.2;
            phase.waiting_time = p["waiting_time"] ? p["waiting_time"].as<int>()    : 0;
            if (p["noise_sigma_act"]) {
                load_doubles(p["noise_sigma_act"], phase.noise_sigma_act, NUM_JOINTS,
                             "phases[].noise_sigma_act");
                phase.has_noise_sigma_act = true;
            }
            cfg.phases.push_back(std::move(phase));
        }
    }
}
