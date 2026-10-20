"""Loading the training data mppi_sim --dataset writes (controllers/muscle/mppi_sim.cpp):
one <prefix>_obs.npy / <prefix>_plan.npy pair per episode."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from learning.observation import NUM_MUSCLES, OBS_DIM

OBS_SUFFIX  = "_obs.npy"    # mppi_sim.cpp's Dataset::kObsSuffix
PLAN_SUFFIX = "_plan.npy"   # mppi_sim.cpp's Dataset::kPlanSuffix


def load_round(prefixes: Sequence[str | Path]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Join the episodes at prefixes into one round.

    Returns observations [N, OBS_DIM], plans [N, horizon, NUM_MUSCLES] in
    activations (both float32, episodes in order), and the prefixes skipped
    because they have neither file: mppi_sim writes none for an episode that
    falls before logging starts.
    """
    obs, plans, skipped = [], [], []
    for prefix in prefixes:
        obs_path, plan_path = Path(f"{prefix}{OBS_SUFFIX}"), Path(f"{prefix}{PLAN_SUFFIX}")
        if not obs_path.exists() and not plan_path.exists():
            skipped.append(str(prefix))
            continue
        if obs_path.exists() != plan_path.exists():
            raise FileNotFoundError(f"{prefix}: has only one of {OBS_SUFFIX}, {PLAN_SUFFIX}")

        y, u = np.load(obs_path), np.load(plan_path)
        if y.ndim != 2 or y.shape[1] != OBS_DIM:
            raise ValueError(f"{obs_path}: shape {y.shape}, expected (N, {OBS_DIM})")
        if u.ndim != 3 or u.shape[2] != NUM_MUSCLES:
            raise ValueError(f"{plan_path}: shape {u.shape}, expected (N, horizon, {NUM_MUSCLES})")
        if len(y) != len(u):
            raise ValueError(f"{prefix}: {len(y)} observations but {len(u)} plans")
        if plans and u.shape[1] != plans[0].shape[1]:
            raise ValueError(f"{plan_path}: horizon {u.shape[1]}, earlier episodes {plans[0].shape[1]}")

        obs.append(y)
        plans.append(u)

    if not obs:
        raise ValueError(f"no episode data at any of {[str(p) for p in prefixes]}")
    return torch.from_numpy(np.concatenate(obs)), torch.from_numpy(np.concatenate(plans)), skipped
