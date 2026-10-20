"""DenoisingCNN (learning/model.py): shapes, gpc's time embedding, the Flax
settings it was ported with, its structure, and dropout."""

import math

import pytest
import torch

from learning.model import TRUNCATED_STD, DenoisingCNN, PositionalEmbedding
from learning.observation import NUM_MUSCLES, OBS_DIM

HORIZON = 40


def small_net() -> DenoisingCNN:
    torch.manual_seed(0)
    return DenoisingCNN(NUM_MUSCLES, OBS_DIM, feature_dims=(16, 16), timestep_embedding_dim=8)


def inputs(batch: int):
    return torch.randn(batch, HORIZON, NUM_MUSCLES), torch.randn(batch, OBS_DIM), torch.rand(batch, 1)


@pytest.mark.parametrize("batch", [1, 5])
def test_output_shape(batch):
    assert small_net().eval()(*inputs(batch)).shape == (batch, HORIZON, NUM_MUSCLES)


def test_time_embedding_formula():
    emb = PositionalEmbedding(4)(torch.tensor([[0.25]]))
    angles = torch.tensor([1.0, 2.0]) * math.pi * 0.25
    torch.testing.assert_close(emb[0], torch.cat([torch.sin(angles), torch.cos(angles)]))


def test_batchnorm_uses_flax_settings():
    norms = [m for m in small_net().modules() if isinstance(m, torch.nn.BatchNorm1d)]
    assert norms
    for m in norms:
        assert m.momentum == 0.01 and m.eps == 1e-5


def test_weights_use_flax_init():
    torch.manual_seed(0)
    net = DenoisingCNN(NUM_MUSCLES, OBS_DIM, feature_dims=(128, 128, 128), timestep_embedding_dim=64)
    for m in net.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d)):
            assert torch.all(m.bias == 0)

    # A 128 -> 128 conv with kernel 3: fan_in 384, 49,152 weights.
    weight = net.blocks[1].encoder.c.weight
    std = 1 / math.sqrt(384)
    assert abs(weight.std().item() - std) < 0.03 * std
    assert weight.abs().max().item() <= 2 * std / TRUNCATED_STD + 1e-6


def test_output_adds_input_plan():
    # With every parameter zero, each block outputs zero, leaving gpc's final "+ u".
    net = small_net().eval()
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
    u, y, t = inputs(3)
    torch.testing.assert_close(net(u, y, t), u)


def test_dropout_only_in_training():
    net = small_net()
    u, y, t = inputs(4)
    net.eval()
    torch.testing.assert_close(net(u, y, t), net(u, y, t))
    net.train()
    assert not torch.equal(net(u, y, t), net(u, y, t))
