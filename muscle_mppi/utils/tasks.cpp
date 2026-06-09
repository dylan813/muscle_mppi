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
    cfg.dt            = t["dt"].as<double>();

    load_doubles(t["nominal_pose"], cfg.nominal_pose, NUM_JOINTS, "nominal_pose");
    if (t["noise_sigma_act"])
        load_doubles(t["noise_sigma_act"], cfg.noise_sigma_act, NUM_JOINTS, "noise_sigma_act");
    if (t["noise_sigma_final"])
        load_doubles(t["noise_sigma_final"], cfg.noise_sigma_final, NUM_JOINTS, "noise_sigma_final");
    else
        std::copy(cfg.noise_sigma_act, cfg.noise_sigma_act + NUM_JOINTS, cfg.noise_sigma_final);

    if (t["gravity_act"])
        load_doubles(t["gravity_act"], cfg.gravity_act, NUM_MUSCLES, "gravity_act");
    if (t["posture_bias"])
        load_doubles(t["posture_bias"], cfg.posture_bias, NUM_JOINTS, "posture_bias");
    if (t["posture_FL1"])
        load_doubles(t["posture_FL1"],  cfg.posture_FL1,  NUM_JOINTS, "posture_FL1");
    if (t["posture_FL2"])
        load_doubles(t["posture_FL2"],  cfg.posture_FL2,  NUM_JOINTS, "posture_FL2");

    const YAML::Node& m = t["muscle"];
    cfg.muscle.act_bandwidth = m["act_bandwidth"].as<double>();
    load_doubles(m["peak_force"], cfg.muscle.peak_force, NUM_JOINTS, "peak_force");
    load_doubles(m["lce_min"],    cfg.muscle.lce_min,    NUM_JOINTS, "lce_min");
    load_doubles(m["lce_max"],    cfg.muscle.lce_max,    NUM_JOINTS, "lce_max");
    load_doubles(m["phi_min"],    cfg.muscle.phi_min,    NUM_JOINTS, "phi_min");
    load_doubles(m["phi_max"],    cfg.muscle.phi_max,    NUM_JOINTS, "phi_max");
    load_doubles(m["vmax"],       cfg.muscle.vmax,       NUM_JOINTS, "vmax");
    load_doubles(m["FVmax"],      cfg.muscle.FVmax,      NUM_JOINTS, "FVmax");
    load_doubles(m["pFLmax"],     cfg.muscle.pFLmax,     NUM_JOINTS, "pFLmax");
    load_doubles(m["kd_sim"],     cfg.muscle.kd_sim,     NUM_JOINTS, "kd_sim");

    return cfg;
}
