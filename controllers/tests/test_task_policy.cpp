// The muscle task's optional policy: block (PolicyParams, muscle/utils/tasks.h),
// as load_task() reads it from copies of tasks.yaml with the walk task edited.

#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>

#include <unistd.h>
#include <yaml-cpp/yaml.h>

#include "check.h"
#include "../muscle/utils/tasks.h"

namespace fs = std::filesystem;

using Edit = std::function<void(YAML::Node& walk)>;

static const fs::path kDir = fs::temp_directory_path() / ("test_task_policy_" + std::to_string(getpid()));

// The walk task from a copy of tasks.yaml, after edit.
static TaskConfig load_walk(const Edit& edit)
{
    YAML::Node root = YAML::LoadFile(kDefaultTasksYaml);
    YAML::Node walk = root["walk"];
    edit(walk);

    const fs::path path = kDir / "tasks.yaml";
    std::ofstream(path) << YAML::Dump(root);
    return load_task("walk", path.string());
}

// load_walk(edit) throws a std::runtime_error whose message contains text.
static bool refused(const Edit& edit, const std::string& text)
{
    try {
        load_walk(edit);
    } catch (const std::runtime_error& e) {
        return std::string(e.what()).find(text) != std::string::npos;
    }
    return false;
}

static void add_policy(YAML::Node& walk)
{
    walk["policy"]["path"]        = "analysis/log/gpc/run/policy.onnx";
    walk["policy"]["num_samples"] = 30;
    walk["policy"]["warm_start"]  = 0.25;
    walk["policy"]["device"]      = "cuda";
}

static void test_no_block_no_policy()
{
    const TaskConfig task = load_walk([](YAML::Node&) {});
    CHECK(task.policy.path.empty());
    CHECK(task.policy.num_samples == 0);
}

// The path resolves against the repo root, as model_path does; an absolute one is kept.
static void test_block_is_read()
{
    const TaskConfig task = load_walk(add_policy);
    CHECK(task.policy.path == repo_path("analysis/log/gpc/run/policy.onnx"));
    CHECK(fs::path(task.policy.path).is_absolute());
    CHECK(task.policy.num_samples == 30);
    CHECK_NEAR(task.policy.warm_start, 0.25, 0.0);
    CHECK(task.policy.device == PolicyDevice::Cuda);

    const TaskConfig cpu = load_walk([](YAML::Node& walk) { add_policy(walk); walk["policy"]["device"] = "cpu"; });
    CHECK(cpu.policy.device == PolicyDevice::Cpu);

    const TaskConfig absolute = load_walk([](YAML::Node& walk) {
        add_policy(walk);
        walk["policy"]["path"] = "/elsewhere/policy.onnx";
    });
    CHECK(absolute.policy.path == "/elsewhere/policy.onnx");
}

static void test_bad_block_refused()
{
    for (const char* key : {"path", "num_samples", "warm_start", "device"})
        CHECK(refused([key](YAML::Node& walk) { add_policy(walk); walk["policy"].remove(key); },
                      std::string("policy needs '") + key + "'"));

    CHECK(refused([](YAML::Node& walk) { add_policy(walk); walk["policy"]["warmstart"] = 0.0; },
                  "unknown policy key 'warmstart'"));
    CHECK(refused([](YAML::Node& walk) { add_policy(walk); walk["policy"]["num_samples"] = 0; },
                  "policy.num_samples"));
    CHECK(refused([](YAML::Node& walk) { add_policy(walk); walk["policy"]["device"] = "gpu"; },
                  "policy.device"));
    for (double warm : {-0.1, 1.5})
        CHECK(refused([warm](YAML::Node& walk) { add_policy(walk); walk["policy"]["warm_start"] = warm; },
                      "policy.warm_start"));
}

// The policy outputs the plan's knots, so it needs cubic sampling.
static void test_policy_needs_cubic_sampling()
{
    CHECK(refused([](YAML::Node& walk) { add_policy(walk); walk["sample_type"] = "normal"; },
                  "policy needs sample_type cubic"));
}

// n_samples 0 leaves the policy's samples alone; with no policy it leaves none.
static void test_needs_a_sample()
{
    const TaskConfig task = load_walk([](YAML::Node& walk) { add_policy(walk); walk["n_samples"] = 0; });
    CHECK(task.n_samples == 0 && task.policy.num_samples == 30);

    CHECK(refused([](YAML::Node& walk) { walk["n_samples"] = 0; }, "must be at least 1"));
}

int main()
{
    fs::create_directories(kDir);

    test_no_block_no_policy();
    test_block_is_read();
    test_bad_block_refused();
    test_policy_needs_cubic_sampling();
    test_needs_a_sample();

    fs::remove_all(kDir);
    return check::exit_code();
}
