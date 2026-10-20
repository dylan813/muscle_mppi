"""gpc's observation normalizer, ported as is.

gpc uses a Flax BatchNorm with no learned scale or bias (gpc/training.py:324-331),
which standardizes each observation feature, (y - mean) / sqrt(var + eps):
  - training: each round's observations are standardized with that round's own
    mean and variance (training.py:420, use_running_average=False)
  - runtime: with running statistics (policy.py:80), which start at mean 0 and
    variance 1 and after each round become 0.1 · old + 0.9 · that round's
So in early rounds the runtime statistics still carry the starting values;
their share shrinks tenfold per round.
"""

import torch
from torch import nn

MOMENTUM = 0.1   # gpc: share of the old running statistics kept each round
EPS      = 1e-5  # Flax BatchNorm default


class ObsNormalizer(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("var", torch.ones(num_features))

    @torch.no_grad()
    def fit_round(self, y: torch.Tensor) -> torch.Tensor:
        """Standardize one round's observations [N, F] with their own statistics,
        and blend those into the running statistics."""
        mean = y.mean(0)
        var  = (y - mean).square().mean(0)   # biased, as Flax computes it
        self.mean.mul_(MOMENTUM).add_((1 - MOMENTUM) * mean)
        self.var.mul_(MOMENTUM).add_((1 - MOMENTUM) * var)
        return (y - mean) / torch.sqrt(var + EPS)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Standardize with the running statistics, as at runtime."""
        return (y - self.mean) / torch.sqrt(self.var + EPS)
