"""gpc's Policy (gpc/policy.py): the network with its observation normalizer.
It samples plans from raw observations, and saves and loads as one file."""

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from learning import flow_matching
from learning.config import ModelConfig
from learning.model import DenoisingCNN
from learning.normalizer import ObsNormalizer
from learning.observation import NUM_MUSCLES, OBS_DIM


class Policy(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.normalizer = ObsNormalizer(OBS_DIM)
        self.model = DenoisingCNN(NUM_MUSCLES, OBS_DIM, cfg.feature_dims,
                                  cfg.kernel_size, cfg.timestep_embedding_dim)

    def sample(self, y: torch.Tensor, prev_plan: torch.Tensor, warm_start: float = 0.0) -> torch.Tensor:
        """Plans [B, H, A] in activations for raw observations y [B, OBS_DIM],
        normalized with the running statistics as gpc's Policy.apply does."""
        return flow_matching.sample(self.model, self.normalizer(y), prev_plan, warm_start)

    def save(self, path: Path) -> None:
        torch.save({"model_config": asdict(self.cfg), "state": self.state_dict()}, path)

    @staticmethod
    def load(path: Path, map_location: str = "cpu") -> "Policy":
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        policy = Policy(ModelConfig(**ckpt["model_config"]))
        policy.load_state_dict(ckpt["state"])
        return policy
