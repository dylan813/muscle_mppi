#include "tasks.h"

#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <string>

#include "../../common/task_config.h"

TaskConfig load_task(const std::string& task_name, const std::string& yaml_path)
{
    const YAML::Node t = load_task_node(task_name, yaml_path);
    TaskConfig cfg;
    load_task_base(t, cfg);

    cfg.height_target = t["height_target"] ? t["height_target"].as<double>() : 0.0;

    cfg.stiffness = t["stiffness"] ? t["stiffness"].as<double>() : 0.75;
    if (!(cfg.stiffness >= 0.0 && cfg.stiffness <= 1.0))
        throw std::runtime_error("Field 'stiffness': expected a value in [0, 1], got "
                                 + std::to_string(cfg.stiffness));
    if (t["posture_bias"] || t["posture_FL1"] || t["posture_FL2"]
        || (t["cost"] && t["cost"]["gait_stiffness"]))
        throw std::runtime_error("Task '" + task_name + "': posture_bias/posture_FL1/posture_FL2 "
                                 "and cost.gait_stiffness are no longer used — the warm start is "
                                 "computed at startup and the co-contraction level is the "
                                 "task-level 'stiffness'. Remove them from " + yaml_path);

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
