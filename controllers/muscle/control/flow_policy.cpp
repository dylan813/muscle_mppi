#include "flow_policy.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

#include <onnxruntime_session_options_config_keys.h>

// The file's input and output names (learning/policy.py: export_onnx).
static const char* const kInputNames[]  = {"y", "prev_plan", "noise", "warm_start"};
static const char* const kOutputNames[] = {"plan"};

// One CPU thread, the caller's, without spinning: ONNX Runtime's default pool
// keeps one thread per core spinning between calls, which competed with the
// rollouts' OpenMP threads (solves 20× slower in the first training run).
// With Cuda, the network runs on GPU 0 instead, in full float32: TF32, ONNX
// Runtime's default, rounds its products to about 3 significant digits.
static Ort::SessionOptions session_options(PolicyDevice device)
{
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.AddConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning, "0");
    if (device == PolicyDevice::Cpu) return options;

    const OrtApi& api = Ort::GetApi();
    OrtCUDAProviderOptionsV2* created = nullptr;
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&created));
    const std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(api.ReleaseCUDAProviderOptions)>
        cuda(created, api.ReleaseCUDAProviderOptions);

    const char* keys[]   = {"device_id", "use_tf32"};
    const char* values[] = {"0",         "0"};
    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(cuda.get(), keys, values, 2));
    options.AppendExecutionProvider_CUDA_V2(*cuda);
    return options;
}

FlowPolicy::FlowPolicy(const std::string& path, int horizon, const std::vector<double>& knot_steps,
                       PolicyDevice device)
    : horizon_(horizon), n_knots_(static_cast<int>(knot_steps.size())),
      session_(env_, path.c_str(), session_options(device)),
      memory_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    std::string steps;   // as export_onnx writes them: "0,13,26,39"
    for (double k : knot_steps)
        steps += (steps.empty() ? "" : ",") + std::to_string(static_cast<int>(k));

    const std::pair<const char*, std::string> expected[] = {
        {"horizon",     std::to_string(horizon)},
        {"knot_steps",  steps},
        {"obs_dim",     std::to_string(OBS_DIM)},
        {"num_muscles", std::to_string(NUM_MUSCLES)},
    };

    const Ort::ModelMetadata meta = session_.GetModelMetadata();
    Ort::AllocatorWithDefaultOptions allocator;
    for (const auto& [key, value] : expected) {
        const Ort::AllocatedStringPtr label = meta.LookupCustomMetadataMapAllocated(key, allocator);
        if (!label)
            throw std::runtime_error("FlowPolicy: " + path + ": no " + key + " label");
        if (value != label.get())
            throw std::runtime_error("FlowPolicy: " + path + ": label " + key + " is " + label.get()
                                     + ", the controller's is " + value);
    }
}

void FlowPolicy::sample(const double y[OBS_DIM], int n, const double* prev_plans, const double* noise,
                        double warm_start, double* plans)
{
    if (n < 1) throw std::invalid_argument("FlowPolicy::sample: n must be at least 1");

    const size_t plan_size  = static_cast<size_t>(horizon_) * NUM_MUSCLES;
    const size_t noise_size = static_cast<size_t>(n_knots_) * NUM_MUSCLES;

    // One observation for every sample.
    y_.resize(static_cast<size_t>(n) * OBS_DIM);
    for (int s = 0; s < n; ++s) std::copy(y, y + OBS_DIM, y_.begin() + s * OBS_DIM);
    prev_.assign(prev_plans, prev_plans + n * plan_size);
    noise_.assign(noise, noise + n * noise_size);
    plans_.resize(n * plan_size);
    float warm = static_cast<float>(warm_start);

    const int64_t y_shape[]     = {n, OBS_DIM};
    const int64_t plan_shape[]  = {n, horizon_, NUM_MUSCLES};
    const int64_t noise_shape[] = {n, n_knots_, NUM_MUSCLES};
    const int64_t warm_shape[]  = {1};
    const Ort::Value inputs[] = {
        Ort::Value::CreateTensor<float>(memory_, y_.data(),     y_.size(),     y_shape,     2),
        Ort::Value::CreateTensor<float>(memory_, prev_.data(),  prev_.size(),  plan_shape,  3),
        Ort::Value::CreateTensor<float>(memory_, noise_.data(), noise_.size(), noise_shape, 3),
        Ort::Value::CreateTensor<float>(memory_, &warm,         1,             warm_shape,  1),
    };
    Ort::Value output = Ort::Value::CreateTensor<float>(memory_, plans_.data(), plans_.size(), plan_shape, 3);

    session_.Run(Ort::RunOptions{nullptr}, kInputNames, inputs, 4, kOutputNames, &output, 1);
    std::copy(plans_.begin(), plans_.end(), plans);
}
