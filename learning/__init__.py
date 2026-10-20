"""Flow-matching policy for muscle MPPI, ported from gpc (Generative Predictive
Control, Kurtz & Burdick 2025, github.com/vincekurtz/gpc) to PyTorch.

    observation.py     layout of the policy observation y (mirrors observation.h)
    normalizer.py      gpc's observation normalizer
    model.py           gpc's DenoisingCNN
    flow_matching.py   gpc's flow-matching loss and sampler
"""
