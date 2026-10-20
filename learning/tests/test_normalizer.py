"""ObsNormalizer (learning/normalizer.py) behaves as gpc's: each round is
standardized with its own statistics, the running statistics start at mean 0,
variance 1 and become 0.1 · old + 0.9 · round, and runtime uses them."""

import torch

from learning.normalizer import EPS, ObsNormalizer

SCALE  = torch.tensor([1.0, 0.01, 5.0])
OFFSET = torch.tensor([0.0, -1.0, 2.0])


def round_data(seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(1000, 3) * SCALE + OFFSET


def biased_var(y: torch.Tensor) -> torch.Tensor:
    return y.var(0, unbiased=False)


def test_round_is_standardized_with_its_own_statistics():
    y = round_data(0)
    out = ObsNormalizer(3).fit_round(y)
    torch.testing.assert_close(out, (y - y.mean(0)) / torch.sqrt(biased_var(y) + EPS))


def test_running_statistics_follow_gpc():
    norm = ObsNormalizer(3)
    y1, y2 = round_data(1), round_data(2)

    norm.fit_round(y1)
    torch.testing.assert_close(norm.mean, 0.1 * 0.0 + 0.9 * y1.mean(0))
    torch.testing.assert_close(norm.var,  0.1 * 1.0 + 0.9 * biased_var(y1))

    mean1, var1 = norm.mean.clone(), norm.var.clone()
    norm.fit_round(y2)
    torch.testing.assert_close(norm.mean, 0.1 * mean1 + 0.9 * y2.mean(0))
    torch.testing.assert_close(norm.var,  0.1 * var1 + 0.9 * biased_var(y2))


def test_runtime_uses_running_statistics():
    norm = ObsNormalizer(3)
    norm.fit_round(round_data(3))
    y = round_data(4)
    torch.testing.assert_close(norm(y), (y - norm.mean) / torch.sqrt(norm.var + EPS))


def test_constant_feature_stays_finite():
    out = ObsNormalizer(2).fit_round(torch.ones(10, 2))
    assert torch.all(out == 0)
