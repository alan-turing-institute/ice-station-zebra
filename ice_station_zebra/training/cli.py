import logging

import typer
from omegaconf import DictConfig

from ice_station_zebra.cli import hydra_adaptor

from .trainer import ZebraTrainer

# Create the typer app
training_cli = typer.Typer(help="Train models")

log = logging.getLogger(__name__)


@training_cli.command()
@hydra_adaptor
def train(config: DictConfig) -> None:
    """Train a model."""
    trainer = ZebraTrainer(config)
    trainer.train()

@training_cli.command()
@hydra_adaptor
def tune(
    config: DictConfig,
    n_trials = 2
) -> None:
    """Tune hyperparameters using Optuna"""
    import optuna

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-6, 1e-2)
        weight_decay = trial.suggest_float("weight_decay", 0.05, 0.1)

        config["train"]["optimizer"]["lr"] = lr
        config["train"]["optimizer"]["weight_decay"] = weight_decay

        trainer = ZebraTrainer(config)
        trainer.train()

        val_loss = trainer.trainer.callback_metrics.get("validation_loss")
        return val_loss.item() if val_loss else float('inf')

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=int(n_trials))
    log.info(f"best param: {study.best_params}")

if __name__ == "__main__":
    training_cli()
