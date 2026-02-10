import logging

import typer
from omegaconf import DictConfig

from icenetmp.cli import hydra_adaptor

from .trainer import ModelTrainer

# Create the typer app
training_cli = typer.Typer(help="Train models")

log = logging.getLogger(__name__)


@training_cli.command()
@hydra_adaptor
def train(config: DictConfig) -> None:
    """Train a model."""
    trainer = ModelTrainer(config)
    trainer.train()


if __name__ == "__main__":
    training_cli()
