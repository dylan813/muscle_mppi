"""gpc's training.py: fitting the policy to MPPI's plans, round by round.

  settings      config.yaml, read into Config
  loss          gpc's flow-matching loss (inside gpc's fit_policy)
  fitting       fit_policy, fit_round (gpc's jit_fit), the optimizer
  checkpoints   gpc's rule
  logging       to wandb, as analysis/optimize/cmaes_walk.py does
  command       gpc's train(); for now one round, fitted to existing episodes:

    python -m learning.training --data analysis/data/mppi_sim/mppi_sim --out analysis/log/gpc/<run>

Settings come from --config (default learning/config.yaml). MPPI's plans are
converted to the task's knots (mppi.KnotSpline). The policy is saved to
<out>/policy.pt, and wandb keeps its local files in <out>.
"""

import argparse
import time
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path

import torch
import wandb
import yaml
from torch import nn

from learning.mppi import KnotSpline, load_round, repo_path
from learning.policy import ModelConfig, Policy, to_network

# ── Settings (config.yaml; see it for each value's source) ───────────────────

DEFAULT_PATH = Path(__file__).with_name("config.yaml")


@dataclass
class MPPITaskConfig:
    name: str
    tasks_yaml: str


@dataclass
class FitConfig:
    learning_rate: float
    batch_size: int
    num_epochs: int


@dataclass
class WandbConfig:
    project: str
    run_name: str | None
    mode: str


@dataclass
class Config:
    seed: int
    mppi_task: MPPITaskConfig
    model: ModelConfig
    fit: FitConfig
    checkpoint_every: int
    wandb: WandbConfig


def load_config(path: Path = DEFAULT_PATH) -> Config:
    return _build(Config, yaml.safe_load(Path(path).read_text()), str(path))


def _build(cls, node, where: str):
    """cls from a YAML mapping whose keys are exactly cls's fields."""
    if not isinstance(node, dict):
        raise ValueError(f"{where}: expected a mapping, got {node!r}")

    names = {f.name for f in fields(cls)}
    if unknown := node.keys() - names:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}")
    if missing := names - node.keys():
        raise ValueError(f"{where}: missing key(s) {sorted(missing)}")

    values = {}
    for f in fields(cls):
        value = node[f.name]
        if is_dataclass(f.type):
            value = _build(f.type, value, f"{where}: {f.name}")
        elif f.type in (int, float):
            value = f.type(value)
        values[f.name] = value
    return cls(**values)


# ── Loss (gpc/training.py:164-190) ───────────────────────────────────────────

SIGMA_MIN = 1e-2   # gpc's fit_policy default (Lipman et al. 2022, eq. 20-23)


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


# ── Fitting (gpc's fit_policy and jit_fit) ───────────────────────────────────

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


# ── Logging ──────────────────────────────────────────────────────────────────

class RunLogger:
    """wandb.init with the run's settings as config, then one wandb.log per
    round (step = round), as cmaes_walk.py does, next to a printed summary
    line as gpc prints one per round."""

    def __init__(self, cfg: WandbConfig, config: dict, out_dir: Path):
        """config: what the run was set up with, shown in wandb. wandb keeps its
        local files in out_dir."""
        out_dir.mkdir(parents=True, exist_ok=True)
        self.run = wandb.init(project=cfg.project, name=cfg.run_name, config=config,
                              dir=str(out_dir), mode=cfg.mode)

    def log_round(self, round_index: int, metrics: dict[str, float]) -> None:
        self.run.log(metrics, step=round_index)
        print(f"round {round_index} | " + " | ".join(f"{k} {v:.4g}" for k, v in metrics.items()))

    def finish(self) -> None:
        self.run.finish()


# ── Command (gpc's train()) ──────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the flow-matching policy (learning/).")
    parser.add_argument("--config", type=Path, default=DEFAULT_PATH, help="settings YAML")
    parser.add_argument("--data", nargs="+", required=True,
                        help="episode prefixes, as given to mppi_sim --dataset")
    parser.add_argument("--out", type=Path, required=True,
                        help="output folder for the policy and wandb's files")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spline = KnotSpline.from_task(cfg.mppi_task.name, repo_path(cfg.mppi_task.tasks_yaml))

    policy = Policy(cfg.model).to(device)
    optimizer = make_optimizer(policy, cfg.fit.learning_rate)
    logger = RunLogger(cfg.wandb, {**asdict(cfg), "data": args.data}, args.out)

    obs, plans, skipped = load_round(args.data)
    if skipped:
        print(f"no data for (fell before logging): {skipped}")
    print(f"round 0: {len(obs)} rows from {len(args.data) - len(skipped)} episode(s), "
          f"{spline.horizon}-step plans as {spline.n_knots} knots, on {device}")

    start = time.time()
    losses = fit_round(policy, optimizer, obs, spline.from_plan(plans), cfg.fit)
    logger.log_round(0, {"fit/loss": losses[-1].item(), "fit/time": time.time() - start})

    policy.save(args.out / "policy.pt")
    print(f"saved {args.out / 'policy.pt'}")
    logger.finish()


if __name__ == "__main__":
    main()
