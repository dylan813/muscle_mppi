"""gpc's flow matching, ported to PyTorch: the training loss
(gpc/training.py:164-190) and the sampler (gpc/policy.py:55-106).

The network works on plans in network units, [-1, 1]. Activations stay in
[ACT_MIN, ACT_MAX] = [0, 1] everywhere else, and are mapped at the network's
boundary as gpc maps any action range (training.py:412-416, policy.py:101-104).
"""

import torch
from torch import nn

ACT_MIN, ACT_MAX = 0.0, 1.0   # muscle activation range (hill_compute_torques clamps to it)
SIGMA_MIN  = 1e-2             # gpc's fit_policy default (Lipman et al. 2022, eq. 20-23)
FLOW_STEPS = 10               # gpc's Policy.dt = 0.1


def to_network(act: torch.Tensor) -> torch.Tensor:
    return (act - (ACT_MAX + ACT_MIN) / 2) / ((ACT_MAX - ACT_MIN) / 2)


def from_network(u: torch.Tensor) -> torch.Tensor:
    return u * ((ACT_MAX - ACT_MIN) / 2) + (ACT_MAX + ACT_MIN) / 2


def noised_plan(plan: torch.Tensor, noise: torch.Tensor, t: torch.Tensor,
                sigma_min: float = SIGMA_MIN) -> torch.Tensor:
    """The point at flow time t on the straight path from noise (t = 0) to
    plan (t = 1, up to sigma_min · noise). plan, noise [B, H, A]; t [B, 1]."""
    t = t[..., None]
    return t * plan + (1 - (1 - sigma_min) * t) * noise


def loss(model: nn.Module, y: torch.Tensor, plan: torch.Tensor,
         noise: torch.Tensor | None = None, t: torch.Tensor | None = None,
         sigma_min: float = SIGMA_MIN) -> torch.Tensor:
    """gpc's flow-matching loss, unweighted (gpc's weight is a constant in its
    code): mean squared error between the network's velocity at the noised plan
    and the path's velocity, plan - (1 - sigma_min) · noise.

    y: normalized observations [B, O]. plan: MPPI's plans [B, H, A] in network
    units. noise and t are drawn as in gpc unless given.
    """
    if noise is None:
        noise = torch.randn_like(plan)
    if t is None:
        t = torch.rand(plan.shape[0], 1, dtype=plan.dtype, device=plan.device)

    target = plan - (1 - sigma_min) * noise
    return torch.mean(torch.square(model(noised_plan(plan, noise, t, sigma_min), y, t) - target))


@torch.no_grad()
def sample(model: nn.Module, y: torch.Tensor, prev_plan: torch.Tensor,
           warm_start: float = 0.0, steps: int = FLOW_STEPS) -> torch.Tensor:
    """gpc's Policy.apply on a batch: flow a plan from noise, conditioned on y.

    y: normalized observations [B, O]. prev_plan: the previous plan [B, H, A],
    in activations. The flow starts at warm_start · prev_plan + (1 - warm_start)
    · noise, with prev_plan first mapped to network units (gpc blends it
    unmapped). Each of the Euler steps is clipped to [-1, 1]. Returns
    activations. The model must be in eval mode, as gpc samples (model.eval()).
    """
    if model.training:
        raise ValueError("sample() needs the model in eval mode (model.eval())")

    warm_start = min(max(warm_start, 0.0), 1.0)
    u = warm_start * to_network(prev_plan) + (1 - warm_start) * torch.randn_like(prev_plan)

    dt = 1.0 / steps
    for k in range(steps):
        t = torch.full((u.shape[0], 1), k * dt, dtype=u.dtype, device=u.device)
        u = torch.clamp(u + dt * model(u, y, t), -1.0, 1.0)
    return from_network(u)
