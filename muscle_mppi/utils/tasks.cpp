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
    load_doubles(t["noise_sigma"],  cfg.noise_sigma,  NUM_JOINTS, "noise_sigma");

    const YAML::Node& c = t["cost"];
    cfg.cost.height        = c["height"].as<double>();
    cfg.cost.orientation   = c["orientation"].as<double>();
    cfg.cost.posture       = c["posture"].as<double>();
    cfg.cost.contact_vel   = c["contact_vel"].as<double>();
    cfg.cost.contact_force = c["contact_force"].as<double>();
    cfg.cost.terminal      = c["terminal"].as<double>();
    cfg.cost.act_effort    = c["act_effort"].as<double>();
    cfg.cost.act_reference = c["act_reference"].as<double>();
    cfg.cost.vel_cmd       = c["vel_cmd"].as<double>();
    load_doubles(c["vel_des"], cfg.cost.vel_des, 3, "vel_des");

    const YAML::Node& m = t["muscle"];
    cfg.muscle.activation_alpha = m["activation_alpha"].as<double>();
    load_doubles(m["tau_max"],     cfg.muscle.tau_max,     NUM_JOINTS, "tau_max");
    load_doubles(m["dq_max"],      cfg.muscle.dq_max,      NUM_JOINTS, "dq_max");
    load_doubles(m["b_damp"],      cfg.muscle.b_damp,      NUM_JOINTS, "b_damp");
    load_doubles(m["kd_sim"],      cfg.muscle.kd_sim,      NUM_JOINTS, "kd_sim");
    load_doubles(m["k_plus"],      cfg.muscle.k_plus,      NUM_JOINTS, "k_plus");
    load_doubles(m["k_minus"],     cfg.muscle.k_minus,     NUM_JOINTS, "k_minus");
    load_doubles(m["alpha_plus"],  cfg.muscle.alpha_plus,  NUM_JOINTS, "alpha_plus");
    load_doubles(m["alpha_minus"], cfg.muscle.alpha_minus, NUM_JOINTS, "alpha_minus");
    load_doubles(m["q_plus0"],     cfg.muscle.q_plus0,     NUM_JOINTS, "q_plus0");
    load_doubles(m["q_minus0"],    cfg.muscle.q_minus0,    NUM_JOINTS, "q_minus0");

    return cfg;
}
