import logging
from pathlib import Path
from typing import Annotated

import typer
from omegaconf import DictConfig

from ice_station_zebra.cli import hydra_adaptor

from .evaluator import ZebraEvaluator

# Create the typer app
evaluation_cli = typer.Typer(help="Evaluate models")

log = logging.getLogger(__name__)


@evaluation_cli.command(help="Evaluate a model")
@hydra_adaptor
def evaluate(
    config: DictConfig,
    checkpoint: Annotated[
        str, typer.Option(help="Specify the path to a trained model checkpoint")
    ],
) -> None:
    """Evaluate a model"""
    evaluator = ZebraEvaluator(config, Path(checkpoint).resolve())
    evaluator.evaluate()


if __name__ == "__main__":
    evaluation_cli()
