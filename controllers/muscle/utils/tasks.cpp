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

    if (t["posture_bias"] || t["posture_FL1"] || t["posture_FL2"]
        || (t["cost"] && t["cost"]["gait_stiffness"]) || t["stiffness"] || t["nominal_pose"]
        || t["height_target"])
        throw std::runtime_error("Task '" + task_name + "': posture_bias/posture_FL1/posture_FL2, "
                                 "cost.gait_stiffness, a task-level 'stiffness', nominal_pose and "
                                 "height_target are no longer used — the warm start is computed "
                                 "at startup from the settled stand-up, the co-contraction level "
                                 "is muscle.stiffness and the height target is each phase's "
                                 "goal_pos z. Update " + yaml_path);

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

    cfg.muscle.stiffness = m["stiffness"] ? m["stiffness"].as<double>() : 0.75;
    if (!(cfg.muscle.stiffness >= 0.0 && cfg.muscle.stiffness <= 1.0))
        throw std::runtime_error("Field 'muscle.stiffness': expected a value in [0, 1], got "
                                 + std::to_string(cfg.muscle.stiffness));

    // Ablation switches live at task level rather than inside `muscle:`, so one
    // task can be switched without touching the shared *default_muscle_quad
    // anchor (yaml-cpp has no merge keys, <<:). Each is optional and defaults to
    // true; run_ablation_walk.sh writes this block into per-condition YAMLs.
    // Unknown keys throw, so a typo can't silently run the full model.
    if (const YAML::Node& a = t["ablation"]) {
        for (const auto& kv : a) {
            const std::string key = kv.first.as<std::string>();
            if (key != "activation_dynamics" && key != "use_fl" && key != "use_fv" && key != "use_passive")
                throw std::runtime_error("Task '" + task_name + "': unknown ablation key '" + key
                                         + "' (expected activation_dynamics, use_fl, use_fv, use_passive)");
        }
        auto flag = [&a](const char* key, bool& dst) { if (a[key]) dst = a[key].as<bool>(); };
        flag("activation_dynamics", cfg.muscle.activation_dynamics);
        flag("use_fl",              cfg.muscle.use_fl);
        flag("use_fv",              cfg.muscle.use_fv);
        flag("use_passive",         cfg.muscle.use_passive);
    }

    return cfg;
}
