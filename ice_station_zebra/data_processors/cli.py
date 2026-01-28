import logging
from typing import Annotated

import typer
from omegaconf import DictConfig

from ice_station_zebra.cli import hydra_adaptor

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
    factory = ZebraDataProcessorFactory(config)
    for dataset in factory.datasets:
        logger.info("Working on %s.", dataset.name)
        dataset.load(parts=parts)


@datasets_cli.command("load_in_parts")
@hydra_adaptor
def load_in_parts(
    config: DictConfig,
    *,
    continue_on_error: Annotated[
        bool, typer.Option(help="Continue to next part on error")
    ] = True,
    force_reset: Annotated[
        bool,
        typer.Option(
            help="Clear existing progress part_tracker file and start from part 1"
        ),
    ] = False,
    dataset: Annotated[
        str | None, typer.Option(help="Run only a single dataset by name")
    ] = None,
    total_parts: Annotated[
        int, typer.Option(help="Override default total parts (10)")
    ] = 10,
    overwrite: Annotated[
        bool,
        typer.Option(help="Delete the dataset directory before loading"),
    ] = False,
) -> None:
    """Load all parts for all datasets in parts, recording progress so runs can be resumed."""
    factory = ZebraDataProcessorFactory(config)
    for ds in factory.datasets:
        if dataset and ds.name != dataset:
            logger.info("Not loading %s.", ds.name)
            continue
        logger.info("Working on %s.", ds.name)
        ds.load_in_parts(
            continue_on_error=continue_on_error,
            force_reset=force_reset,
            total_parts=total_parts,
            overwrite=overwrite,
        )


@datasets_cli.command("finalise")
@hydra_adaptor
def finalise(
    config: DictConfig,
) -> None:
    """Finalise loaded dataset."""
    factory = ZebraDataProcessorFactory(config)
    for dataset in factory.datasets:
        logger.info("Working on %s.", dataset.name)
        dataset.finalise()


if __name__ == "__main__":
    datasets_cli()
