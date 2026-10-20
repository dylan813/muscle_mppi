"""Run logging to Weights & Biases, as analysis/optimize/cmaes_walk.py does it:
wandb.init with the run's settings as config, then one wandb.log per round
(step = round), next to a printed summary line as gpc prints one per round."""

from pathlib import Path

import wandb

from learning.config import WandbConfig


class RunLogger:
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
