import logging
from pathlib import Path
from typing import Annotated

import typer
from omegaconf import DictConfig

from icenet_mp.cli import hydra_adaptor

from .evaluator import ModelEvaluator

# Create the typer app
evaluation_cli = typer.Typer(help="Evaluate models")

log = logging.getLogger(__name__)


@evaluation_cli.command()
@hydra_adaptor
def evaluate(
    config: DictConfig,
    checkpoint: Annotated[
        str, typer.Option(help="Specify the path to a trained model checkpoint")
    ],
) -> None:
    """Evaluate a model."""
    evaluator = ModelEvaluator(config, Path(checkpoint).resolve())
    evaluator.evaluate()


if __name__ == "__main__":
    evaluation_cli()
