"""Flow-matching policy for muscle MPPI, ported from gpc (Generative Predictive
Control, Kurtz & Burdick 2025, github.com/vincekurtz/gpc) to PyTorch.

    architectures.py   gpc's architectures.py: the DenoisingCNN
    policy.py          gpc's policy.py: Policy, observation normalizer, flow sampler
    training.py        gpc's training.py: loss, fitting, settings, logging, the
                       command (python -m learning.training)
    mppi.py            the C++ controller's side: observation layout, knots, dataset files
    config.yaml        training settings
"""
