#pragma once

// The learned policy (learning/), run from the file training exports
// (learning/policy.py: export_onnx) with ONNX Runtime. One sample() flows n
// plans from noise, all conditioned on one observation, as gpc's Policy.apply
// does per sample (gpc/policy.py:56-106). No state is kept between calls: the
// caller keeps each sample's previous plan and draws the noise, as gpc's
// caller does. Runs on the caller's thread alone, leaving the cores to the
// rollouts.
//
// Layouts (H = horizon, K = number of knots, A = NUM_MUSCLES):
//   y           OBS_DIM raw values (control/observation.h)
//   prev_plans  n × H × A activations, [s * H * A + t * A + m], as trajectory_
//   noise       n × K × A standard normal draws, [s * K * A + k * A + m]
//   plans       n × H × A activations, as prev_plans; not clamped to [0, 1]
//
// warm_start in [0, 1]: each flow starts at warm_start · its previous plan's
// knots + (1 - warm_start) · its noise. 0 starts from noise alone, as gpc
// trains.

#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "observation.h"

class FlowPolicy {
public:
    // Loads path. Throws std::runtime_error if a label the file's inputs and
    // output are sized by is missing or differs from this controller's:
    // horizon, knot_steps (BaseMPPI::knot_x_, whole timesteps), obs_dim,
    // num_muscles. The task label isn't checked.
    FlowPolicy(const std::string& path, int horizon, const std::vector<double>& knot_steps);

    void sample(const double y[OBS_DIM], int n, const double* prev_plans, const double* noise,
                double warm_start, double* plans);

private:
    int horizon_;
    int n_knots_;

    Ort::Env        env_;
    Ort::Session    session_{nullptr};
    Ort::MemoryInfo memory_{nullptr};

    // The file's inputs and output in float32, as it was trained.
    std::vector<float> y_, prev_, noise_, plans_;
};
