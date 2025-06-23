import logging

import typer
from omegaconf import DictConfig

from ice_station_zebra.cli import hydra_adaptor

from .anemoi_dataset import AnemoiDatasetManager

# Create the typer app
datasets_cli = typer.Typer(help="Manage datasets")

log = logging.getLogger(__name__)


@datasets_cli.command("create")
@hydra_adaptor
def create(
    config: DictConfig,
) -> None:
    """Create all datasets"""
    manager = AnemoiDatasetManager(config)
    for dataset in manager.datasets:
        log.info(f"Working on {dataset.name}")
        dataset.download()


if __name__ == "__main__":
    datasets_cli()
