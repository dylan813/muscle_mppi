"""Train the flow-matching policy (gpc's train(), gpc/training.py:236-500).

For now, one round fitted to existing mppi_sim --dataset episodes, whose
plans are converted to the task's knots (learning/knots.py):

    python -m learning.train --data analysis/data/mppi_sim/mppi_sim --out analysis/log/gpc/<run>

Settings come from --config (default learning/config.yaml). The policy is
saved to <out>/policy.pt, and wandb keeps its local files in <out>.
"""

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch

from learning.config import DEFAULT_PATH, load_config, repo_path
from learning.data import load_round
from learning.knots import KnotSpline
from learning.log import RunLogger
from learning.policy import Policy
from learning.training import fit_round, make_optimizer


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
