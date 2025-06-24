import logging

import typer
from omegaconf import DictConfig

from ice_station_zebra.cli import hydra_adaptor

# Create the typer app
training_cli = typer.Typer(help="Train models")

log = logging.getLogger(__name__)


@training_cli.command("train")
@hydra_adaptor
def train(
    config: DictConfig,
) -> None:
    """Train a model"""
    pass


if __name__ == "__main__":
    training_cli()
