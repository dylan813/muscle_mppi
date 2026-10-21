"""FlowPolicy (controllers/muscle/control/flow_policy.h), the C++ controller's
runner of the exported policy: it computes what the Python policy does, on the
CPU and on the GPU, and refuses a file whose labels don't fit the controller's
plans.

Runs the C++ test binary on files exported here, so build it first (make in
controllers/build).
"""

import subprocess
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from learning.mppi import NUM_MUSCLES, OBS_DIM, KnotSpline
from learning.policy import ModelConfig, PlanSampler, Policy, export_onnx
from learning.training import FitConfig, fit_round, make_optimizer

REPO    = Path(__file__).resolve().parents[2]
BINARY  = REPO / "controllers/build/test_flow_policy"
HORIZON = 40
SPLINE  = KnotSpline(HORIZON, 4)   # the walk task's knots, which the binary loads with

# kCases in controllers/tests/test_flow_policy.cpp.
CASES = ["n1_warm0", "n1_warm0.7", "n30_warm0", "n30_warm0.7"]

# Where the binary runs the policy (PolicyDevice); cuda needs a GPU. The GPU
# sums in another order: on the trained pilot policy it differs from the CPU
# by up to 1.6e-5.
TOLERANCE = {"cpu": 1e-5, "cuda": 1e-4}
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(),
                                                                reason="no CUDA GPU"))]

pytestmark = pytest.mark.skipif(not BINARY.exists(),
                                reason=f"{BINARY} not built (make in controllers/build)")


def run(*args) -> subprocess.CompletedProcess:
    return subprocess.run([str(BINARY), *map(str, args)], capture_output=True, text=True)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """A briefly fitted policy, so its normalizer statistics aren't the starting
    ones, and its exported file."""
    torch.manual_seed(0)
    policy = Policy(ModelConfig(feature_dims=[8], timestep_embedding_dim=4, kernel_size=3))
    obs, plans = torch.randn(256, OBS_DIM) * 2 + 1, torch.rand(256, HORIZON, NUM_MUSCLES)
    fit_round(policy, make_optimizer(policy, 1e-3), obs, SPLINE.from_plan(plans), FitConfig(1e-3, 128, 2))

    path = tmp_path_factory.mktemp("export") / "policy.onnx"
    export_onnx(policy, SPLINE, path, "walk")
    return policy, path


@pytest.fixture(scope="module", params=DEVICES)
def cpp_run(exported, tmp_path_factory, request):
    """The device and the folder the binary wrote each case's inputs and
    plans to, running the policy there."""
    _, path = exported
    out = tmp_path_factory.mktemp("cpp")
    result = run(path, request.param, out)
    assert result.returncode == 0, result.stderr
    return request.param, out


@pytest.mark.parametrize("case", CASES)
def test_cpp_plans_match_python(exported, cpp_run, case):
    policy, _ = exported
    device, out = cpp_run
    y, prev, noise, warm_start, plan = (torch.from_numpy(np.load(out / f"{case}_{name}.npy"))
                                        for name in ["y", "prev", "noise", "warm_start", "plan"])

    expected = PlanSampler(policy, SPLINE).eval()(y.expand(len(prev), -1), prev, noise, warm_start)
    np.testing.assert_allclose(plan.numpy(), expected.numpy(), atol=TOLERANCE[device])


def relabeled(path: Path, out: Path, key: str, value: str | None) -> Path:
    """A copy of the file at path with label key set to value, or removed if value is None."""
    model = onnx.load(str(path))
    labels = {entry.key: entry.value for entry in model.metadata_props}
    if value is None:
        del labels[key]
    else:
        labels[key] = value

    del model.metadata_props[:]
    for k, v in labels.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = k, v
    onnx.save(model, str(out))
    return out


def test_matching_file_loads(exported):
    _, path = exported
    result = run(path, "cpu")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("key, wrong", [("horizon", "41"), ("knot_steps", "0,13,26,38"),
                                        ("obs_dim", "61"), ("num_muscles", "25")])
@pytest.mark.parametrize("change", ["wrong", "missing"])
def test_mislabeled_file_is_refused(exported, tmp_path, key, wrong, change):
    _, path = exported
    value = wrong if change == "wrong" else None
    result = run(relabeled(path, tmp_path / "relabeled.onnx", key, value), "cpu")

    assert result.returncode == 1
    message = f"label {key} is {wrong}" if change == "wrong" else f"no {key} label"
    assert message in result.stderr


@pytest.mark.parametrize("value", ["walk_rough", None])
def test_task_label_is_not_checked(exported, tmp_path, value):
    _, path = exported
    result = run(relabeled(path, tmp_path / "relabeled.onnx", "task", value), "cpu")
    assert result.returncode == 0, result.stderr
