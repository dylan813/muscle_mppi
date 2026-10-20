"""Flow matching (learning/flow_matching.py): network units, the noised path,
the loss's target, and the sampler, using velocity fields with known answers
in place of a trained network."""

import pytest
import torch

from learning.flow_matching import (SIGMA_MIN, from_network, loss, noised_plan, sample,
                                    to_network)
from learning.observation import NUM_MUSCLES, OBS_DIM

SHAPE = (2, 40, NUM_MUSCLES)   # batch, horizon, muscles


class Field(torch.nn.Module):
    """A fixed velocity field v(u, t) standing in for the network. Records the
    flow time of each call."""

    def __init__(self, v):
        super().__init__()
        self.v = v
        self.times = []

    def forward(self, u, y, t):
        self.times.append(t[0, 0].item())
        return self.v(u, t)


def run(field, prev_plan=None, warm_start=0.0):
    prev_plan = torch.zeros(SHAPE) if prev_plan is None else prev_plan
    return sample(field.eval(), torch.zeros(SHAPE[0], OBS_DIM), prev_plan, warm_start)


def test_activations_map_to_network_units():
    act = torch.tensor([0.0, 0.5, 1.0, 0.76])
    torch.testing.assert_close(to_network(act), torch.tensor([-1.0, 0.0, 1.0, 0.52]))
    torch.testing.assert_close(from_network(to_network(act)), act)


def test_noised_plan_runs_from_noise_to_plan():
    plan, noise = torch.randn(SHAPE), torch.randn(SHAPE)
    torch.testing.assert_close(noised_plan(plan, noise, torch.zeros(2, 1)), noise)
    torch.testing.assert_close(noised_plan(plan, noise, torch.ones(2, 1)), plan + SIGMA_MIN * noise)


def test_loss_scores_the_paths_velocity():
    plan, noise, t, y = torch.randn(SHAPE), torch.randn(SHAPE), torch.rand(2, 1), torch.zeros(2, OBS_DIM)
    target = plan - (1 - SIGMA_MIN) * noise

    exact = Field(lambda u, t_: target)
    assert loss(exact, y, plan, noise=noise, t=t).item() == 0.0

    off_by_2 = Field(lambda u, t_: target + 2.0)
    assert loss(off_by_2, y, plan, noise=noise, t=t).item() == pytest.approx(4.0)


def test_full_warm_start_with_zero_field_returns_previous_plan():
    # Decision D: the previous plan is mapped to network units before blending.
    # gpc blends it unmapped, which would return (prev + 1) / 2 here.
    prev = torch.rand(SHAPE)
    out = run(Field(lambda u, t: torch.zeros_like(u)), prev, warm_start=1.0)
    torch.testing.assert_close(out, prev)


def test_exact_field_reaches_its_target():
    # v = (c - u) / (1 - t) carries any start to c in exactly the last Euler step.
    c = to_network(torch.full(SHAPE, 0.3))
    out = run(Field(lambda u, t: (c - u) / (1 - t[..., None])))
    torch.testing.assert_close(out, torch.full(SHAPE, 0.3), atol=1e-5, rtol=0)


def test_ten_steps_at_gpc_flow_times():
    field = Field(lambda u, t: torch.zeros_like(u))
    run(field)
    assert field.times == pytest.approx([k * 0.1 for k in range(10)])


def test_steps_are_clipped_to_activation_range():
    assert torch.all(run(Field(lambda u, t: torch.full_like(u, 100.0))) == 1.0)
    assert torch.all(run(Field(lambda u, t: torch.full_like(u, -100.0))) == 0.0)


def test_sampling_needs_eval_mode():
    field = Field(lambda u, t: torch.zeros_like(u)).train()
    with pytest.raises(ValueError):
        sample(field, torch.zeros(SHAPE[0], OBS_DIM), torch.zeros(SHAPE))
