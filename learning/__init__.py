"""Flow-matching policy for muscle MPPI, ported from gpc (Generative Predictive
Control, Kurtz & Burdick 2025, github.com/vincekurtz/gpc) to PyTorch.

    observation.py     layout of the policy observation y (mirrors observation.h)
    normalizer.py      gpc's observation normalizer
    model.py           gpc's DenoisingCNN
    flow_matching.py   gpc's flow-matching loss and sampler
    knots.py           plans as cubic knots, the policy's output (the controller's spline)
    policy.py          gpc's Policy: network + normalizer, saved as one file
    training.py        gpc's training step and checkpoint rule
    data.py            loading mppi_sim --dataset episodes
    config.py          settings from config.yaml
    log.py             wandb run logging
    train.py           entry point: python -m learning.train
"""
