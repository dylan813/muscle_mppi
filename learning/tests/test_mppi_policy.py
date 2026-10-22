"""Policy samples in MPPI (controllers/muscle/control/mppi_locomotion.h, with the
task's policy: block): the policy's plans join MPPI's samples, clamped, and are
rolled out and averaged with them. The policy is given observation()'s y, the
shifted plan MPPI samples around, and the warm start, on the CPU or the GPU.

Most tests use probe policies: small ONNX files built here with the exported
policy's inputs, output and labels (policy.export_onnx), computing something
checkable. They run with n_samples 0, so the update is the policy's alone.

Runs the C++ test binary on the walk task, so build it first (make in
controllers/build). Each run stands the robot up, which takes a second or two.
"""

import subprocess
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
import yaml
from onnx import TensorProto, helper, numpy_helper

from learning.mppi import NUM_MUSCLES, OBS_ACT, OBS_DIM, KnotSpline
from learning.policy import ONNX_OPSET, ModelConfig, Policy, export_onnx

REPO    = Path(__file__).resolve().parents[2]
BINARY  = REPO / "controllers/build/test_mppi_policy"
TASKS   = REPO / "controllers/muscle/utils/tasks.yaml"
HORIZON = 40
SPLINE  = KnotSpline(HORIZON, 4)   # the walk task's knots

# Where the controller runs the policy (the policy: block's device); cuda needs a GPU.
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(),
                                                                reason="no CUDA GPU"))]

pytestmark = pytest.mark.skipif(not BINARY.exists(),
                                reason=f"{BINARY} not built (make in controllers/build)")


def run_controller(tmp_path: Path, n_samples: int, policy: dict | None = None,
                   updates: int = 1) -> list[dict[str, np.ndarray]]:
    """What test_mppi_policy wrote per update, for the walk task with n_samples
    and the given policy: block."""
    tasks = yaml.safe_load(TASKS.read_text())
    tasks["walk"]["n_samples"] = n_samples
    if policy is not None:
        tasks["walk"]["policy"] = policy
    path = tmp_path / "tasks.yaml"
    path.write_text(yaml.safe_dump(tasks))

    out = tmp_path / "out"
    result = subprocess.run([str(BINARY), str(path), str(out), str(updates)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    names = ["plan_before", "y", "plan", "costs", "weight", "terms", "ref_dev"]
    return [{name: np.load(out / f"u{u}_{name}.npy")[0] for name in names} for u in range(updates)]


def shifted(plan: np.ndarray) -> np.ndarray:
    """The plan one step on, its last step held (control_utils.h: shift_trajectory)."""
    return np.concatenate([plan[1:], plan[-1:]])


# ── Probe policies ───────────────────────────────────────────────────────────

def probe(path: Path, nodes: list, constants: dict[str, np.ndarray]) -> Path:
    """An ONNX file with the exported policy's inputs, output and labels whose
    plan is computed by nodes, from the inputs and constants."""
    inputs = [helper.make_tensor_value_info("y",          TensorProto.FLOAT, ["batch", OBS_DIM]),
              helper.make_tensor_value_info("prev_plan",  TensorProto.FLOAT, ["batch", HORIZON, NUM_MUSCLES]),
              helper.make_tensor_value_info("noise",      TensorProto.FLOAT, ["batch", SPLINE.n_knots, NUM_MUSCLES]),
              helper.make_tensor_value_info("warm_start", TensorProto.FLOAT, [1])]
    output = helper.make_tensor_value_info("plan", TensorProto.FLOAT, ["batch", HORIZON, NUM_MUSCLES])
    graph = helper.make_graph(nodes, "probe", inputs, [output],
                              [numpy_helper.from_array(value, name) for name, value in constants.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", ONNX_OPSET)], ir_version=8)

    labels = {"task": "walk", "horizon": HORIZON, "knot_steps": ",".join(map(str, SPLINE.steps)),
              "obs_dim": OBS_DIM, "num_muscles": NUM_MUSCLES, "flow_steps": 10}
    for key, value in labels.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, str(value)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


# plan = CONSTANT for every sample, with values outside [0, 1].
CONSTANT = np.linspace(-0.5, 1.5, HORIZON * NUM_MUSCLES, dtype=np.float32).reshape(1, HORIZON, NUM_MUSCLES)


def constant_probe(path: Path, plan: np.ndarray = CONSTANT) -> Path:
    """plan [1, HORIZON, NUM_MUSCLES] for every sample."""
    return probe(path, [helper.make_node("Mul", ["prev_plan", "zero"], ["unused"]),
                        helper.make_node("Add", ["unused", "constant"], ["plan"])],
                 {"zero": np.array(0.0, np.float32), "constant": plan})


# plan = warm_start · prev_plan + (1 - warm_start) · (y's activations + RAMP),
# RAMP rising over the horizon so plans differ from step to step.
RAMP = (0.005 * np.arange(HORIZON, dtype=np.float32)).reshape(1, HORIZON, 1)


def mix_probe(path: Path) -> Path:
    nodes = [helper.make_node("Slice", ["y", "act_start", "act_end", "axis_1"], ["act"]),
             helper.make_node("Unsqueeze", ["act", "axis_1"], ["act_per_step"]),
             helper.make_node("Add", ["act_per_step", "ramp"], ["target"]),
             helper.make_node("Mul", ["warm_start", "prev_plan"], ["from_prev"]),
             helper.make_node("Sub", ["one", "warm_start"], ["target_share"]),
             helper.make_node("Mul", ["target_share", "target"], ["from_target"]),
             helper.make_node("Add", ["from_prev", "from_target"], ["plan"])]
    return probe(path, nodes, {"act_start": np.array([OBS_ACT], np.int64),
                               "act_end":   np.array([OBS_DIM], np.int64),
                               "axis_1":    np.array([1], np.int64),
                               "ramp":      RAMP,
                               "one":       np.array(1.0, np.float32)})


def mix_expected(update: dict, warm_start: float, prev: np.ndarray) -> np.ndarray:
    target = update["y"][OBS_ACT:] + RAMP[0]
    return np.clip(warm_start * prev + (1 - warm_start) * target, 0.0, 1.0)


# ── Tests ────────────────────────────────────────────────────────────────────

def test_without_policy_only_mppi_samples(tmp_path):
    (update,) = run_controller(tmp_path, n_samples=30)
    sample_min, mppi_min, policy_min = update["costs"]
    assert np.isnan(policy_min)
    assert mppi_min == sample_min
    assert np.isnan(update["weight"])

    # The best samples' terms add up to their costs; the policy has none.
    mppi_terms, policy_terms = update["terms"]
    assert mppi_terms.sum() == pytest.approx(mppi_min, rel=1e-5)
    assert np.isnan(policy_terms).all()
    plan_dev, mppi_dev, policy_dev = update["ref_dev"]
    assert np.isfinite([plan_dev, mppi_dev]).all() and np.isnan(policy_dev)


@pytest.mark.parametrize("device", DEVICES)
def test_policy_plans_join_the_update_clamped(tmp_path, device):
    path = constant_probe(tmp_path / "constant.onnx")
    (update,) = run_controller(tmp_path, n_samples=0,
                               policy={"path": str(path), "num_samples": 2, "warm_start": 0.0, "device": device})

    np.testing.assert_allclose(update["plan"], np.clip(CONSTANT[0], 0.0, 1.0), atol=1e-6)
    sample_min, mppi_min, policy_min = update["costs"]
    assert np.isnan(mppi_min)
    assert policy_min == sample_min
    assert update["weight"] == pytest.approx(1.0)   # the policy's samples are all there are

    # MPPI has no samples, and the plan is the policy's sample, as far from the gait reference.
    mppi_terms, policy_terms = update["terms"]
    assert np.isnan(mppi_terms).all()
    assert policy_terms.sum() == pytest.approx(policy_min, rel=1e-5)
    plan_dev, mppi_dev, policy_dev = update["ref_dev"]
    assert np.isnan(mppi_dev)
    assert plan_dev == pytest.approx(policy_dev, abs=1e-6)


def test_far_worse_policy_gets_no_weight(tmp_path):
    """Every agonist at full activation and every antagonist off throws the
    robot (about 3× the cost of MPPI's samples around its standing plan).
    Weighted by the range of MPPI's own samples (control_utils.h:
    softmin_weights), the policy's samples get ~0 weight. A plan only a few
    percent worse, such as no activation at all, still gets some."""
    one_way = np.zeros((1, HORIZON, NUM_MUSCLES), np.float32)
    one_way[..., 0::2] = 1.0   # agonists, the even muscles (observation.h's act[] layout)
    path = constant_probe(tmp_path / "one_way.onnx", one_way)
    (update,) = run_controller(tmp_path, n_samples=30,
                               policy={"path": str(path), "num_samples": 30, "warm_start": 0.0, "device": "cpu"})

    sample_min, mppi_min, policy_min = update["costs"]
    assert policy_min > mppi_min
    assert update["weight"] < 1e-3


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("warm_start", [0.0, 1.0])
def test_policy_gets_y_and_warm_start(tmp_path, warm_start, device):
    """warm_start 0: the plan is y's activations (+ RAMP), so the policy saw
    observation()'s y. 1: the plan is MPPI's plan, shifted."""
    path = mix_probe(tmp_path / "mix.onnx")
    (update,) = run_controller(tmp_path, n_samples=0,
                               policy={"path": str(path), "num_samples": 2, "warm_start": warm_start,
                                       "device": device})
    expected = mix_expected(update, warm_start, shifted(update["plan_before"]))
    np.testing.assert_allclose(update["plan"], expected, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_policy_starts_from_the_shifted_plan(tmp_path, device):
    """Second update, warm start 0.5: the plan before it varies over the
    horizon (the first update's RAMP), so the shift is visible."""
    path = mix_probe(tmp_path / "mix.onnx")
    _, second = run_controller(tmp_path, n_samples=0, updates=2,
                               policy={"path": str(path), "num_samples": 2, "warm_start": 0.5, "device": device})

    np.testing.assert_allclose(second["plan"], mix_expected(second, 0.5, shifted(second["plan_before"])),
                               atol=1e-6)
    unshifted = mix_expected(second, 0.5, second["plan_before"])
    assert np.abs(second["plan"] - unshifted).max() > 1e-3


@pytest.mark.parametrize("device", DEVICES)
def test_exported_policy_with_mppi_samples(tmp_path, device):
    """A real exported policy (untrained) next to MPPI's 30 samples, as gpc
    samples as many from the policy."""
    torch.manual_seed(0)
    path = tmp_path / "policy.onnx"
    export_onnx(Policy(ModelConfig(feature_dims=[8], timestep_embedding_dim=4, kernel_size=3)),
                SPLINE, path, "walk")
    (update,) = run_controller(tmp_path, n_samples=30,
                               policy={"path": str(path), "num_samples": 30, "warm_start": 0.0, "device": device})

    sample_min, mppi_min, policy_min = update["costs"]
    assert np.isfinite(mppi_min) and np.isfinite(policy_min)
    assert sample_min == min(mppi_min, policy_min)
    assert 0.0 <= update["weight"] <= 1.0
    mppi_terms, policy_terms = update["terms"]
    assert mppi_terms.sum() == pytest.approx(mppi_min, rel=1e-5)
    assert policy_terms.sum() == pytest.approx(policy_min, rel=1e-5)
    assert np.isfinite(update["ref_dev"]).all() and (update["ref_dev"] >= 0).all()
    assert update["plan"].min() >= 0.0 and update["plan"].max() <= 1.0
