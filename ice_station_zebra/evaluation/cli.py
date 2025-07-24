import logging

import typer
from omegaconf import DictConfig

from ice_station_zebra.cli import hydra_adaptor

# Create the typer app
evaluation_cli = typer.Typer(help="Evaluate models")

log = logging.getLogger(__name__)


@evaluation_cli.command(help="Evaluate a model")
@hydra_adaptor
def evaluate(config: DictConfig) -> None:
    """Evaluate a model"""
    pass


if __name__ == "__main__":
    evaluation_cli()
