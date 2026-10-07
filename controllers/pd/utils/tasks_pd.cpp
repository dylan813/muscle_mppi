#include "tasks_pd.h"

#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <string>

#include "../../common/task_config.h"

TaskConfig load_task(const std::string& task_name, const std::string& yaml_path)
{
    const YAML::Node t = load_task_node(task_name, yaml_path);
    TaskConfig cfg;
    load_task_base(t, cfg);

    load_doubles(t["nominal_pose"], cfg.nominal_pose, NUM_JOINTS, "nominal_pose");

    const YAML::Node& pd = t["pd"];
    load_doubles(pd["kp"], cfg.pd.kp, NUM_JOINTS, "pd.kp");
    load_doubles(pd["kd"], cfg.pd.kd, NUM_JOINTS, "pd.kd");
    load_doubles(pd["joint_damping"], cfg.pd.joint_damping, NUM_JOINTS, "pd.joint_damping");

    // Activation dynamics lives at task level rather than inside `pd:`, so one
    // task can switch it on without touching the shared *default_pd_quad anchor
    // (yaml-cpp has no merge keys, <<:), as the muscle variant's `ablation:`
    // block does. Unknown keys throw, so a typo can't silently run plain PD.
    if (const YAML::Node& a = t["activation_dynamics"]) {
        for (const auto& kv : a) {
            const std::string key = kv.first.as<std::string>();
            if (key != "act_bandwidth")
                throw std::runtime_error("Task '" + task_name + "': unknown activation_dynamics key '"
                                         + key + "' (expected act_bandwidth)");
        }
        if (!a["act_bandwidth"])
            throw std::runtime_error("Task '" + task_name + "': activation_dynamics needs act_bandwidth");
        cfg.pd.activation_dynamics = true;
        cfg.pd.act_bandwidth       = a["act_bandwidth"].as<double>();

        // alpha > 1 overshoots the target every step (and oscillates past 2).
        const double alpha = cfg.pd.act_bandwidth * cfg.dt;
        if (!(alpha > 0.0 && alpha <= 1.0))
            throw std::runtime_error("Task '" + task_name + "': activation_dynamics.act_bandwidth * dt must be "
                                     "in (0, 1], got " + std::to_string(alpha));
    }

    return cfg;
}
