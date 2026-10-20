"""RunLogger (learning/training.py) with wandb disabled: logs a round and prints its
summary line, without contacting wandb."""

from learning.training import RunLogger, WandbConfig


def test_logs_a_round_offline(tmp_path, capsys):
    logger = RunLogger(WandbConfig(project="test", run_name="test", mode="disabled"),
                       {"seed": 0}, tmp_path / "run")
    logger.log_round(0, {"fit/loss": 0.5, "fit/time": 2.0})
    logger.finish()

    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line == "round 0 | fit/loss 0.5 | fit/time 2"
