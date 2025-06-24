import logging

import typer
from omegaconf import DictConfig

from ice_station_zebra.cli import hydra_adaptor

from .anemoi_dataset_factory import AnemoiDatasetFactory

# Create the typer app
datasets_cli = typer.Typer(help="Manage datasets")

log = logging.getLogger(__name__)


@datasets_cli.command("create")
@hydra_adaptor
def create(
    config: DictConfig,
) -> None:
    """Create all datasets"""
    factory = AnemoiDatasetFactory(config)
    for dataset in factory.datasets:
        log.info(f"Working on {dataset.name}")
        dataset.download()


@datasets_cli.command("inspect")
@hydra_adaptor
def inspect(
    config: DictConfig,
) -> None:
    """Inspect all datasets"""
    factory = AnemoiDatasetFactory(config)
    for dataset in factory.datasets:
        log.info(f"Working on {dataset.name}")
        dataset.inspect()


if __name__ == "__main__":
    datasets_cli()
