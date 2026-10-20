"""KnotSpline (learning/knots.py) matches the controller's cubic noise
(base_mppi.cpp): knot steps, the not-a-knot spline and its 2- and 3-knot cases,
reading a task's knots, and plans to knots as values at the knot steps."""

import numpy as np
import pytest
import torch
import yaml

from learning.config import load_config, repo_path
from learning.knots import KnotSpline, knot_steps
from learning.observation import NUM_MUSCLES

HORIZON = 40


def plan_of(spline: KnotSpline, knot_values) -> np.ndarray:
    knots = torch.tensor(knot_values, dtype=torch.float32)[None, :, None]
    return spline.to_plan(knots)[0, :, 0].numpy()


def test_knot_steps_match_base_mppi():
    assert knot_steps(40, 4) == [0, 13, 26, 39]
    # std::round rounds halves up: 39 · k / 6 = 6.5 -> 7, 19.5 -> 20, 32.5 -> 33.
    assert knot_steps(40, 7) == [0, 7, 13, 20, 26, 33, 39]


def test_knot_count_is_clamped_as_base_mppi():
    assert KnotSpline(40, 1).n_knots == 2
    assert KnotSpline(5, 9).n_knots == 5


@pytest.mark.parametrize("n_knots", [2, 3, 4, 6])
def test_constant_knots_give_a_constant_plan(n_knots):
    torch.testing.assert_close(KnotSpline(HORIZON, n_knots).basis.sum(dim=1), torch.ones(HORIZON))


@pytest.mark.parametrize("n_knots", [2, 3, 4, 6])
def test_plan_passes_through_its_knots(n_knots):
    spline = KnotSpline(HORIZON, n_knots)
    knots = torch.rand(3, spline.n_knots, NUM_MUSCLES)
    torch.testing.assert_close(spline.from_plan(spline.to_plan(knots)), knots)


def test_two_knots_are_a_straight_line():
    plan = plan_of(KnotSpline(HORIZON, 2), [0.2, 0.8])
    np.testing.assert_allclose(plan, 0.2 + 0.6 * np.arange(HORIZON) / 39, atol=1e-6)


def test_four_knots_are_the_cubic_through_them():
    # With 4 knots both not-a-knot conditions make the spline one cubic.
    spline = KnotSpline(HORIZON, 4)
    y = np.array([0.1, 0.9, 0.3, 0.6])
    cubic = np.polyval(np.polyfit(spline.steps, y, 3), np.arange(HORIZON))
    np.testing.assert_allclose(plan_of(spline, y), cubic, atol=1e-5)


def test_three_knots_are_a_natural_spline():
    # base_mppi.cpp falls back to a natural spline (S'' = 0 at both ends) for 3 knots.
    # Compared against the textbook second-derivative ("moment") form of it.
    spline = KnotSpline(HORIZON, 3)
    x, y = np.array(spline.steps, dtype=float), np.array([0.1, 0.9, 0.3])
    h = np.diff(x)
    m = np.array([0.0, 3 * ((y[2] - y[1]) / h[1] - (y[1] - y[0]) / h[0]) / (h[0] + h[1]), 0.0])

    t = np.arange(HORIZON, dtype=float)
    i = np.where(t <= x[1], 0, 1)
    a, b = x[i + 1] - t, t - x[i]
    natural = (m[i] * a**3 + m[i + 1] * b**3) / (6 * h[i]) \
        + (y[i] / h[i] - m[i] * h[i] / 6) * a + (y[i + 1] / h[i] - m[i + 1] * h[i] / 6) * b
    np.testing.assert_allclose(plan_of(spline, y), natural, atol=1e-5)


def test_plans_knots_are_its_values_at_the_knot_steps():
    spline = KnotSpline(HORIZON, 4)
    plans = torch.rand(2, HORIZON, NUM_MUSCLES)
    torch.testing.assert_close(spline.from_plan(plans), plans[:, [0, 13, 26, 39], :])


def test_plans_of_another_horizon_are_rejected():
    with pytest.raises(ValueError, match="horizon"):
        KnotSpline(HORIZON, 4).from_plan(torch.rand(2, 30, NUM_MUSCLES))


def test_configured_task_reads_from_tasks_yaml():
    cfg = load_config()
    tasks_yaml = repo_path(cfg.mppi_task.tasks_yaml)
    task = yaml.safe_load(tasks_yaml.read_text())[cfg.mppi_task.name]

    spline = KnotSpline.from_task(cfg.mppi_task.name, tasks_yaml)
    assert spline.horizon == task["horizon"]
    assert spline.steps == knot_steps(task["horizon"], task["n_knots"])


def test_task_without_cubic_sampling_is_rejected(tmp_path):
    path = tmp_path / "tasks.yaml"
    path.write_text(yaml.safe_dump({"plain": {"horizon": 40, "sample_type": "normal"}}))
    with pytest.raises(ValueError, match="cubic"):
        KnotSpline.from_task("plain", path)
