"""gpc's training step (gpc/training.py): fitting the policy to one round of
MPPI's plans, and when to save a checkpoint."""

import torch
from torch import nn

from learning.config import FitConfig
from learning.flow_matching import loss, to_network
from learning.policy import Policy


def make_optimizer(policy: Policy, learning_rate: float) -> torch.optim.AdamW:
    """gpc's optax.adamw(learning_rate), with optax's defaults. PyTorch's
    default weight decay is 0.01; optax's is 1e-4."""
    return torch.optim.AdamW(policy.model.parameters(), lr=learning_rate,
                             betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)


def fit_policy(model: nn.Module, optimizer: torch.optim.Optimizer, y: torch.Tensor,
               plans: torch.Tensor, batch_size: int, num_epochs: int) -> torch.Tensor:
    """gpc's fit_policy: (N // batch_size) × num_epochs steps (at least one batch
    per epoch), each on batch_size rows drawn at random with replacement.

    y: normalized observations [N, O]; plans [N, H, A] in network units.
    Returns every step's loss (gpc returns only the last).
    """
    n = y.shape[0]
    steps = max(1, n // batch_size) * num_epochs
    losses = torch.empty(steps, device=y.device)
    for i in range(steps):
        idx = torch.randint(0, n, (batch_size,), device=y.device)
        step_loss = loss(model, y[idx], plans[idx])
        optimizer.zero_grad(set_to_none=True)
        step_loss.backward()
        optimizer.step()
        losses[i] = step_loss.detach()
    return losses.cpu()


def fit_round(policy: Policy, optimizer: torch.optim.Optimizer, obs: torch.Tensor,
              plans: torch.Tensor, cfg: FitConfig) -> torch.Tensor:
    """gpc's jit_fit for one round: raw observations [N, OBS_DIM] and plans
    [N, H, A] in activations. Updates the normalizer with the round, then fits
    with the network in training mode (gpc: policy.model.train())."""
    device = next(policy.parameters()).device
    policy.model.train()
    y = policy.normalizer.fit_round(obs.to(device))
    return fit_policy(policy.model, optimizer, y, to_network(plans.to(device)),
                      cfg.batch_size, cfg.num_epochs)


def checkpoint_due(round_index: int, every: int) -> bool:
    """gpc's rule (training.py:470): a checkpoint every `every` rounds, never after round 0."""
    return round_index > 0 and round_index % every == 0
