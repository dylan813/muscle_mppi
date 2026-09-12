#pragma once

// YAML task-loading helpers shared by both variants' load_task().

#include <string>

#include <yaml-cpp/yaml.h>

#include "types.h"

// Read a length-n YAML sequence into dst; throws naming `field` if the node is
// missing, not a sequence, or the wrong length.
void load_doubles(const YAML::Node& node, double* dst, int n, const std::string& field);

// Parse yaml_path and return its `task_name` entry; throws if the file can't
// be parsed or the task isn't in it.
YAML::Node load_task_node(const std::string& task_name, const std::string& yaml_path);

// Fill the fields common to both variants (see TaskConfigBase) from a task node.
void load_task_base(const YAML::Node& t, TaskConfigBase& cfg);
