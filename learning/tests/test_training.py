"""Training as gpc does it (learning/training.py, learning/policy.py): the
optimizer, how many steps and how batches are drawn, the checkpoint rule, the
policy file, and that fitting reproduces simple datasets, including one with
two equally good plans for the same observation."""

import matplotlib.pyplot as plt
import pytest
import torch

from learning import flow_matching
from learning.config import FitConfig, ModelConfig
from learning.knots import KnotSpline
from learning.observation import NUM_MUSCLES, OBS_DIM
from learning.policy import Policy
from learning.training import checkpoint_due, fit_round, make_optimizer

HORIZON = 40
SPLINE = KnotSpline(HORIZON, 4)   # the walk task's knots
SMALL = ModelConfig(feature_dims=[8], timestep_embedding_dim=4, kernel_size=3)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def random_round(rows: int):
    return torch.randn(rows, OBS_DIM), torch.rand(rows, HORIZON, NUM_MUSCLES)


def test_optimizer_is_adamw_with_optax_defaults():
    opt = make_optimizer(Policy(SMALL), 1e-3)
    assert isinstance(opt, torch.optim.AdamW)
    group = opt.param_groups[0]
    assert group["lr"] == 1e-3 and group["betas"] == (0.9, 0.999)
    assert group["eps"] == 1e-8 and group["weight_decay"] == 1e-4


@pytest.mark.parametrize("rows, expected_steps", [(300, 2 * 3), (50, 1 * 3)])
def test_steps_and_batches(rows, expected_steps):
    # (rows // 128) batches per epoch, at least one; each a full batch of 128,
    # which with 50 rows is only possible drawing with replacement.
    torch.manual_seed(0)
    policy = Policy(SMALL)
    batches = []
    policy.model.register_forward_hook(lambda m, args, out: batches.append(args[0].shape[0]))

    losses = fit_round(policy, make_optimizer(policy, 1e-3), *random_round(rows), FitConfig(1e-3, 128, 3))
    assert len(losses) == len(batches) == expected_steps
    assert all(b == 128 for b in batches)


def test_checkpoint_rule_matches_gpc():
    assert [r for r in range(6) if checkpoint_due(r, 1)] == [1, 2, 3, 4, 5]
    assert [r for r in range(31) if checkpoint_due(r, 10)] == [10, 20, 30]


def test_policy_file_round_trip(tmp_path):
    torch.manual_seed(0)
    policy = Policy(SMALL)
    fit_round(policy, make_optimizer(policy, 1e-3), *random_round(200), FitConfig(1e-3, 128, 1))
    policy.save(tmp_path / "policy.pt")
    loaded = Policy.load(tmp_path / "policy.pt")

    for (name, a), (_, b) in zip(policy.state_dict().items(), loaded.state_dict().items()):
        assert torch.equal(a, b), name

    y, prev = torch.randn(3, OBS_DIM), torch.rand(3, HORIZON, NUM_MUSCLES)
    policy.eval(), loaded.eval()
    torch.manual_seed(1)
    out = policy.sample(y, prev)
    torch.manual_seed(1)
    torch.testing.assert_close(loaded.sample(y, prev), out)


def test_policy_samples_with_running_statistics():
    torch.manual_seed(0)
    policy = Policy(SMALL)
    fit_round(policy, make_optimizer(policy, 1e-3), *random_round(200), FitConfig(1e-3, 128, 1))
    policy.eval()

    y, prev = torch.randn(3, OBS_DIM), torch.rand(3, HORIZON, NUM_MUSCLES)
    torch.manual_seed(1)
    out = policy.sample(y, prev)
    torch.manual_seed(1)
    torch.testing.assert_close(flow_matching.sample(policy.model, policy.normalizer(y), prev), out)


def two_plan_knots():
    """512 knot plans [N, 4, A]: the first half every activation 0.2, the rest 0.8."""
    knots = torch.empty(512, SPLINE.n_knots, NUM_MUSCLES)
    knots[:256], knots[256:] = 0.2, 0.8
    return knots


def test_fit_reproduces_two_observations_plans():
    # Observation feature 0 = +1 -> every activation 0.2; = -1 -> every activation 0.8.
    torch.manual_seed(0)
    obs = torch.zeros(512, OBS_DIM)
    obs[:256, 0], obs[256:, 0] = 1.0, -1.0

    policy = Policy(ModelConfig([32, 32], 16, 3)).to(DEVICE)
    fit_round(policy, make_optimizer(policy, 1e-3), obs, two_plan_knots(), FitConfig(1e-3, 128, 1000))   # 4,000 steps

    policy.eval()
    y = torch.zeros(2, OBS_DIM, device=DEVICE)
    y[0, 0], y[1, 0] = 1.0, -1.0
    knots = policy.sample(y, torch.zeros(2, SPLINE.n_knots, NUM_MUSCLES, device=DEVICE)).cpu()
    plans = SPLINE.to_plan(knots)
    for plan, target in [(plans[0], 0.2), (plans[1], 0.8)]:
        err = (plan - target).abs()
        assert err.mean() < 0.03 and err.max() < 0.15, (target, err.mean().item(), err.max().item())


def test_two_modes_fit():
    # gpc's test_fit (gpc/tests/test_training.py:62-117): fit data with two answers
    # for the same observation (here every activation 0.2, or every activation
    # 0.8), and check only the final loss, with gpc's bound. Whether samples give
    # whole plans of either answer is checked by eye: run this file directly.
    torch.manual_seed(0)
    policy = Policy(ModelConfig([128, 128, 128], 64, 3)).to(DEVICE)   # config.yaml's network
    losses = fit_round(policy, make_optimizer(policy, 1e-3), torch.zeros(512, OBS_DIM),
                       two_plan_knots(), FitConfig(1e-3, 128, 1000))   # 4,000 steps
    assert losses[-1] < 1.0

    if __name__ == "__main__":
        policy.eval()
        knots = policy.sample(torch.zeros(100, OBS_DIM, device=DEVICE),
                              torch.zeros(100, SPLINE.n_knots, NUM_MUSCLES, device=DEVICE)).cpu()
        plt.plot(SPLINE.to_plan(knots)[:, :, 0].T, alpha=0.3)
        plt.xlabel("horizon step")
        plt.ylabel("muscle 0 activation")
        plt.title("100 sampled plans; training plans are all 0.2 or all 0.8")
        plt.show()


if __name__ == "__main__":
    test_two_modes_fit()
