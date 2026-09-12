#include "tasks.h"

#include <yaml-cpp/yaml.h>
#include <string>

#include "../../common/task_loader.h"

TaskConfig load_task(const std::string& task_name, const std::string& yaml_path)
{
    const YAML::Node t = load_task_node(task_name, yaml_path);
    TaskConfig cfg;
    load_task_base(t, cfg);

    cfg.height_target = t["height_target"] ? t["height_target"].as<double>() : 0.0;

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
