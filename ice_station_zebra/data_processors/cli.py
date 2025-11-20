import logging

import typer
from typing import Annotated
from omegaconf import DictConfig

from ice_station_zebra.cli import hydra_adaptor

from .filters import register_filters
from .zebra_data_processor_factory import ZebraDataProcessorFactory

# Create the typer app
datasets_cli = typer.Typer(help="Manage datasets")

logger = logging.getLogger(__name__)


@datasets_cli.command("create")
@hydra_adaptor
def create(config: DictConfig,
            overwrite: Annotated[
                bool, typer.Option(help="Specify whether to overwrite existing datasets")
           ] = False,
           ) -> None:
    """Create all datasets."""
    register_filters()
    factory = ZebraDataProcessorFactory(config)
    for dataset in factory.datasets:
        logger.info("Working on %s.", dataset.name)
        dataset.create(overwrite=overwrite)


@datasets_cli.command("inspect")
@hydra_adaptor
def inspect(config: DictConfig) -> None:
    """Inspect all datasets."""
    factory = ZebraDataProcessorFactory(config)
    for dataset in factory.datasets:
        logger.info("Working on %s.", dataset.name)
        dataset.inspect()


if __name__ == "__main__":
    datasets_cli()
