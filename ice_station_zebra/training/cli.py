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
    n_trials = 15
) -> None:
    """Tune hyperparameters using Optuna"""
    import optuna
    import torch
    import gc
    torch.set_float32_matmul_precision('medium')
    
    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-5, 5e-3)
        weight_decay = trial.suggest_float("weight_decay", 0.01, 0.15)

        config["train"]["optimizer"]["lr"] = lr
        config["train"]["optimizer"]["weight_decay"] = weight_decay

        trainer = ZebraTrainer(config)
        trainer.train()

        val_loss = trainer.trainer.callback_metrics.get("validation_loss")
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
        return val_loss.item() if val_loss else float('inf')

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=int(n_trials))
    log.info(f"best param: {study.best_params}")

if __name__ == "__main__":
    training_cli()
