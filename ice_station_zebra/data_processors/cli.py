import logging
from typing import Annotated

import typer
from omegaconf import DictConfig

from ice_station_zebra.cli import hydra_adaptor

from .filters import register_filters
from .zebra_data_processor_factory import ZebraDataProcessorFactory

# Create the typer app
datasets_cli = typer.Typer(help="Manage datasets")

logger = logging.getLogger(__name__)


@datasets_cli.command("create")
@hydra_adaptor
def create(
    config: DictConfig,
    *,
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


@datasets_cli.command("init")
@hydra_adaptor
def init(
    config: DictConfig,
    *,
    overwrite: Annotated[
        bool, typer.Option(help="Specify whether to overwrite existing datasets")
    ] = False,
) -> None:
    """Create all datasets."""
    register_filters()
    factory = ZebraDataProcessorFactory(config)
    for dataset in factory.datasets:
        logger.info("Working on %s.", dataset.name)
        dataset.init(overwrite=overwrite)


@datasets_cli.command("load")
@hydra_adaptor
def load(
    config: DictConfig,
    parts: Annotated[str, typer.Option(help="The part to process, specified as 'i/n'")],
) -> None:
    """Load dataset in parts."""
    register_filters()
    factory = ZebraDataProcessorFactory(config)
    for dataset in factory.datasets:
        logger.info("Working on %s.", dataset.name)
        dataset.load(parts=parts)


@datasets_cli.command("finalise")
@hydra_adaptor
def finalise(
    config: DictConfig,
) -> None:
    """Finalise loaded dataset."""
    register_filters()
    factory = ZebraDataProcessorFactory(config)
    for dataset in factory.datasets:
        logger.info("Working on %s.", dataset.name)
        dataset.finalise()


if __name__ == "__main__":
    datasets_cli()
