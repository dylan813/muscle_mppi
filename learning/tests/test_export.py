"""Export for the C++ controller (learning/policy.py: PlanSampler, export_onnx):
the file is valid ONNX with the specified inputs, outputs and labels, and ONNX
Runtime computes what the Python policy does for the same noise, at any batch
size (the number of policy samples) and warm-start level."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from learning.mppi import NUM_MUSCLES, OBS_DIM, KnotSpline
from learning.policy import ModelConfig, PlanSampler, Policy, export_onnx
from learning.training import FitConfig, fit_round, make_optimizer

HORIZON = 40
SPLINE = KnotSpline(HORIZON, 4)   # the walk task's knots


def session(path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


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


def test_file_is_valid_with_the_specified_interface(exported):
    _, path = exported
    onnx.checker.check_model(onnx.load(str(path)))

    s = session(path)
    assert [(i.name, i.shape) for i in s.get_inputs()] == [
        ("y",          ["batch", OBS_DIM]),
        ("prev_plan",  ["batch", HORIZON, NUM_MUSCLES]),
        ("noise",      ["batch", SPLINE.n_knots, NUM_MUSCLES]),
        ("warm_start", [1]),
    ]
    assert [(o.name, o.shape) for o in s.get_outputs()] == [("plan", ["batch", HORIZON, NUM_MUSCLES])]


def test_labels_describe_the_walk_task(exported):
    _, path = exported
    assert session(path).get_modelmeta().custom_metadata_map == {
        "task": "walk", "horizon": "40", "knot_steps": "0,13,26,39",
        "obs_dim": str(OBS_DIM), "num_muscles": str(NUM_MUSCLES), "flow_steps": "10",
    }


@pytest.mark.parametrize("warm_start", [0.0, 0.7])
@pytest.mark.parametrize("batch", [1, 2, 16])
def test_runtime_matches_python(exported, batch, warm_start):
    policy, path = exported
    torch.manual_seed(batch)
    inputs = {
        "y":          torch.randn(batch, OBS_DIM) * 2 + 1,
        "prev_plan":  torch.rand(batch, HORIZON, NUM_MUSCLES),
        "noise":      torch.randn(batch, SPLINE.n_knots, NUM_MUSCLES),
        "warm_start": torch.tensor([warm_start]),
    }

    (plan,) = session(path).run(None, {name: value.numpy() for name, value in inputs.items()})
    expected = PlanSampler(policy, SPLINE).eval()(**inputs)
    np.testing.assert_allclose(plan, expected.numpy(), atol=1e-5)
