#include "tasks.h"

#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <string>

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
    cfg.height_target = t["height_target"] ? t["height_target"].as<double>() : 0.0;
    cfg.n_samples     = t["n_samples"].as<int>();
    cfg.horizon       = t["horizon"].as<int>();
    cfg.substeps      = t["substeps"].as<int>();
    cfg.n_iterations  = t["n_iterations"].as<int>();
    cfg.lambda        = t["lambda"].as<double>();
    cfg.beta1         = t["beta1"].as<double>();
    cfg.beta2         = t["beta2"].as<double>();
    cfg.dt            = t["dt"].as<double>();

    load_doubles(t["nominal_pose"], cfg.nominal_pose, NUM_JOINTS,  "nominal_pose");
    load_doubles(t["noise_sigma"],  cfg.noise_sigma,  NUM_MUSCLES, "noise_sigma");
    load_doubles(t["foot_target"],  cfg.foot_target,  3,           "foot_target");

    const YAML::Node& c = t["cost"];
    cfg.cost.height        = c["height"].as<double>();
    cfg.cost.orientation   = c["orientation"].as<double>();
    cfg.cost.posture       = c["posture"].as<double>();
    cfg.cost.contact_vel   = c["contact_vel"].as<double>();
    cfg.cost.contact_force = c["contact_force"].as<double>();
    cfg.cost.terminal      = c["terminal"].as<double>();
    cfg.cost.act_effort    = c["act_effort"].as<double>();
    cfg.cost.vel_cmd       = c["vel_cmd"].as<double>();
    load_doubles(c["vel_des"], cfg.cost.vel_des, 3, "vel_des");
    cfg.cost.foot_pos      = c["foot_pos"].as<double>();

    const YAML::Node& m = t["muscle"];
    cfg.muscle.act_bandwidth = m["act_bandwidth"].as<double>();
    load_doubles(m["peak_force"], cfg.muscle.peak_force, NUM_JOINTS, "peak_force");
    load_doubles(m["lce_min"],    cfg.muscle.lce_min,    NUM_JOINTS, "lce_min");
    load_doubles(m["lce_max"],    cfg.muscle.lce_max,    NUM_JOINTS, "lce_max");
    load_doubles(m["phi_min"],    cfg.muscle.phi_min,    NUM_JOINTS, "phi_min");
    load_doubles(m["phi_max"],    cfg.muscle.phi_max,    NUM_JOINTS, "phi_max");
    load_doubles(m["vmax"],       cfg.muscle.vmax,       NUM_JOINTS, "vmax");
    load_doubles(m["fvmax"],      cfg.muscle.fvmax,      NUM_JOINTS, "fvmax");
    load_doubles(m["fpmax"],      cfg.muscle.fpmax,      NUM_JOINTS, "fpmax");
    load_doubles(m["kd_sim"],     cfg.muscle.kd_sim,     NUM_JOINTS, "kd_sim");

    return cfg;
}
