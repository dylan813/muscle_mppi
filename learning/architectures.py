"""gpc's DenoisingCNN (gpc/architectures.py:81-256), ported to PyTorch with the
Flax 0.10 defaults gpc ran with, where PyTorch's differ:
  - weights: lecun_normal (truncated normal, std 1/sqrt(fan_in)); biases zero.
    (PyTorch: Kaiming-uniform weights, random biases.)
  - BatchNorm: keeps 99% of its running statistics per batch, epsilon 1e-5.
    (PyTorch: keeps 90%.)
One small difference remains: PyTorch updates BatchNorm's running variance with
the unbiased batch variance, Flax with the biased one, a factor n / (n - 1) for
n = batch size × horizon.

Shapes as in gpc: plans u [B, horizon, NUM_MUSCLES] (time, then muscle),
observations y [B, OBS_DIM], flow time t [B, 1].
"""

import math
from collections.abc import Sequence

import torch
from torch import nn

BN_MOMENTUM = 0.01   # Flax's 0.99 kept, in PyTorch's convention (share of the new batch)
BN_EPS      = 1e-5

# Spread of a unit normal truncated to [-2, 2]. Flax's lecun_normal divides by
# it so the truncated draw still has std 1/sqrt(fan_in).
TRUNCATED_STD = 0.87962566103423978


def lecun_normal_(weight: torch.Tensor) -> None:
    """Flax's default weight init: truncated normal, std 1/sqrt(fan_in)."""
    fan_in = weight[0].numel()   # Linear: in_features; Conv1d: in_channels × kernel_size
    std = 1.0 / math.sqrt(fan_in) / TRUNCATED_STD
    nn.init.trunc_normal_(weight, std=std, a=-2 * std, b=2 * std)


class PositionalEmbedding(nn.Module):
    """Flow time t as [sin(k·π·t), cos(k·π·t)] for k = 1..dim/2."""

    def __init__(self, dim: int):
        super().__init__()
        freqs = torch.arange(1, dim // 2 + 1, dtype=torch.float32) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:   # [B, 1] -> [B, dim]
        emb = t * self.freqs
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class Conv1DBlock(nn.Module):
    """Conv over time (zero padding keeping the length) -> BatchNorm -> SiLU (gpc's swish)."""

    def __init__(self, in_features: int, out_features: int, kernel_size: int):
        super().__init__()
        self.c  = nn.Conv1d(in_features, out_features, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_features, momentum=BN_MOMENTUM, eps=BN_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, C, H]
        return nn.functional.silu(self.bn(self.c(x)))


class ConditionalResidualBlock(nn.Module):
    """Encoder conv block, plus a bias computed from the conditioning (the same
    at every time step), dropout, decoder conv block, plus a 1x1-conv residual."""

    def __init__(self, in_features: int, out_features: int, cond_features: int, kernel_size: int):
        super().__init__()
        self.encoder  = Conv1DBlock(in_features, out_features, kernel_size)
        self.decoder  = Conv1DBlock(out_features, out_features, kernel_size)
        self.linear   = nn.Linear(cond_features, out_features)
        self.dropout  = nn.Dropout(0.1)
        self.residual = nn.Conv1d(in_features, out_features, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:   # x [B, C, H], cond [B, F]
        z = self.encoder(x) + self.linear(cond)[..., None]
        z = self.decoder(self.dropout(z))
        return z + self.residual(x)


class DenoisingCNN(nn.Module):
    """Flow velocity for a plan, conditioned on the observation and flow time.

    feature_dims sets the blocks' widths: len(feature_dims) + 1 blocks, from
    action_size channels through each width and back to action_size.
    """

    def __init__(self, action_size: int, observation_size: int, feature_dims: Sequence[int],
                 kernel_size: int = 3, timestep_embedding_dim: int = 32):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd, so zero padding keeps the horizon length")

        self.positional_embedding = PositionalEmbedding(timestep_embedding_dim)
        sizes = [action_size, *feature_dims, action_size]
        cond_features = observation_size + timestep_embedding_dim
        self.blocks = nn.ModuleList(
            ConditionalResidualBlock(i, o, cond_features, kernel_size)
            for i, o in zip(sizes[:-1], sizes[1:]))

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                lecun_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, u: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        cond = torch.cat([y, self.positional_embedding(t)], dim=-1)
        x = u.transpose(1, 2)   # PyTorch convolutions take [B, channels, time]
        for block in self.blocks:
            x = block(x, cond)
        return x.transpose(1, 2) + u
