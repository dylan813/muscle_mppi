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

    load_doubles(t["nominal_pose"], cfg.nominal_pose, NUM_JOINTS, "nominal_pose");
    if (t["foot_targets"]) {
        const YAML::Node& fts = t["foot_targets"];
        cfg.n_foot_targets = std::min((int)fts.size(), TaskConfig::MAX_FOOT_TARGETS);
        for (int i = 0; i < cfg.n_foot_targets; ++i)
            load_doubles(fts[i], cfg.foot_targets[i], 3, "foot_targets");
        for (int k = 0; k < 3; ++k) cfg.foot_target[k] = cfg.foot_targets[0][k];
    } else {
        if (t["foot_target"]) load_doubles(t["foot_target"], cfg.foot_target, 3, "foot_target");
        cfg.n_foot_targets = 1;
        for (int k = 0; k < 3; ++k) cfg.foot_targets[0][k] = cfg.foot_target[k];
    }
    if (t["tau_max"])
        load_doubles(t["tau_max"], cfg.tau_max, NUM_JOINTS, "tau_max");
    else
        for (int j = 0; j < NUM_JOINTS; ++j) cfg.tau_max[j] = 1e6;
    if (t["noise_sigma"])
        load_doubles(t["noise_sigma"], cfg.noise_sigma, NUM_MUSCLES, "noise_sigma");

    // Spline parameterisation fields (optional — only used by MPPILocomotion).
    cfg.n_nodes = t["n_nodes"] ? t["n_nodes"].as<int>() : 10;
    if (t["noise_sigma_q"]) load_doubles(t["noise_sigma_q"], cfg.noise_sigma_q, NUM_JOINTS, "noise_sigma_q");
    if (t["noise_sigma_v"]) load_doubles(t["noise_sigma_v"], cfg.noise_sigma_v, NUM_JOINTS, "noise_sigma_v");
    if (t["kp_muscle"])     load_doubles(t["kp_muscle"],     cfg.kp_muscle,     NUM_JOINTS, "kp_muscle");
    if (t["kd_muscle"])     load_doubles(t["kd_muscle"],     cfg.kd_muscle,     NUM_JOINTS, "kd_muscle");

    const YAML::Node& c = t["cost"];
    cfg.cost.height        = c["height"]        ? c["height"].as<double>()        : 0.0;
    cfg.cost.orientation   = c["orientation"]   ? c["orientation"].as<double>()   : 0.0;
    cfg.cost.posture       = c["posture"]       ? c["posture"].as<double>()       : 0.0;
    cfg.cost.contact_vel   = c["contact_vel"]   ? c["contact_vel"].as<double>()   : 0.0;
    cfg.cost.contact_force = c["contact_force"] ? c["contact_force"].as<double>() : 0.0;
    cfg.cost.terminal      = c["terminal"]      ? c["terminal"].as<double>()      : 0.0;
    cfg.cost.act_effort    = c["act_effort"]    ? c["act_effort"].as<double>()    : 0.0;
    cfg.cost.vel_cmd       = c["vel_cmd"]       ? c["vel_cmd"].as<double>()       : 0.0;
    if (c["vel_des"]) load_doubles(c["vel_des"], cfg.cost.vel_des, 3, "vel_des");
    cfg.cost.foot_pos      = c["foot_pos"]      ? c["foot_pos"].as<double>()      : 0.0;
    cfg.cost.torque        = c["torque"]        ? c["torque"].as<double>()        : 0.0;
    cfg.cost.torque_rate   = c["torque_rate"]   ? c["torque_rate"].as<double>()   : 0.0;
    cfg.cost.joint_vel     = c["joint_vel"]     ? c["joint_vel"].as<double>()     : 0.0;
    cfg.cost.base_drift    = c["base_drift"]    ? c["base_drift"].as<double>()    : 0.0;

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
