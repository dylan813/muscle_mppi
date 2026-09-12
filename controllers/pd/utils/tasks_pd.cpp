#include "tasks_pd.h"

#include <yaml-cpp/yaml.h>
#include <string>

#include "../../common/task_config.h"

TaskConfig load_task(const std::string& task_name, const std::string& yaml_path)
{
    const YAML::Node t = load_task_node(task_name, yaml_path);
    TaskConfig cfg;
    load_task_base(t, cfg);

    const YAML::Node& pd = t["pd"];
    load_doubles(pd["kp"], cfg.pd.kp, NUM_JOINTS, "pd.kp");
    load_doubles(pd["kd"], cfg.pd.kd, NUM_JOINTS, "pd.kd");
    load_doubles(pd["joint_damping"], cfg.pd.joint_damping, NUM_JOINTS, "pd.joint_damping");

    return cfg;
}
