import logging
from pathlib import Path
from typing import Annotated

import typer

from .evaluator import ZebraEvaluator

# Create the typer app
evaluation_cli = typer.Typer(help="Evaluate models")

log = logging.getLogger(__name__)


@evaluation_cli.command(help="Evaluate a model")
def evaluate(
    checkpoint: Annotated[
        str, typer.Argument(help="Specify the path to a trained model checkpoint")
    ]
) -> None:
    """Evaluate a model"""
    evaluator = ZebraEvaluator(Path(checkpoint).resolve())
    evaluator.evaluate()


if __name__ == "__main__":
    evaluation_cli()
