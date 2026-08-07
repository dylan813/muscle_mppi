#include "tasks_pd.h"

#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <string>
#include <utility>

static void load_doubles(const YAML::Node& node, double* dst, int n,
                         const std::string& field)
{
    if (!node || !node.IsSequence() || static_cast<int>(node.size()) != n)
        throw std::runtime_error("Field '" + field + "': expected sequence of length "
                                 + std::to_string(n));
    for (int i = 0; i < n; ++i)
        dst[i] = node[i].as<double>();
}

TaskConfig load_task(const std::string& task_name, const std::string& yaml_path)
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

    const YAML::Node& t = root[task_name];
    TaskConfig cfg;

    cfg.model_path    = t["model_path"].as<std::string>();
    cfg.n_samples     = t["n_samples"].as<int>();
    cfg.horizon       = t["horizon"].as<int>();
    cfg.lambda        = t["lambda"].as<double>();
    cfg.dt            = t["dt"].as<double>();
    cfg.sample_type   = t["sample_type"] ? t["sample_type"].as<std::string>() : "normal";
    cfg.n_knots       = t["n_knots"]     ? t["n_knots"].as<int>()             : 4;
    cfg.seed          = t["seed"]        ? t["seed"].as<int>()                : 42;
    cfg.num_threads   = t["num_threads"] ? t["num_threads"].as<int>()        : 0;
    cfg.sim_duration  = t["sim_duration"] ? t["sim_duration"].as<double>()   : 10.0;
    cfg.spawn_height_offset =
        t["spawn_height_offset"] ? t["spawn_height_offset"].as<double>()    : 0.0;

    load_doubles(t["nominal_pose"], cfg.nominal_pose, NUM_JOINTS, "nominal_pose");
    if (t["noise_sigma_act"])
        load_doubles(t["noise_sigma_act"], cfg.noise_sigma_act, NUM_JOINTS, "noise_sigma_act");

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
            phase.gait_path    = p["gait_path"]    ? p["gait_path"].as<std::string>()    : "";
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

    const YAML::Node& pd = t["pd"];
    load_doubles(pd["kp"], cfg.pd.kp, NUM_JOINTS, "pd.kp");
    load_doubles(pd["kd"], cfg.pd.kd, NUM_JOINTS, "pd.kd");
    load_doubles(pd["joint_damping"], cfg.pd.joint_damping, NUM_JOINTS, "pd.joint_damping");

    return cfg;
}
