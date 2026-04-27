import logging
from pathlib import Path
from typing import Annotated

import typer
from omegaconf import DictConfig

from icenet_mp.model_service import ModelService

from .hydra import hydra_adaptor

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
    save_activations: Annotated[  # noqa: FBT002
        bool,
        typer.Option(
            "--save-activations/--no-save-activations",
            help=(
                "Register forward hooks on selected model layers during the "
                "test loop and save captured activations to disk."
            ),
        ),
    ] = False,
    activation_layer: Annotated[
        list[str] | None,
        typer.Option(
            "--activation-layer",
            help=(
                "Dotted path of a model submodule to hook (e.g. 'processor.conv1'). "
                "Repeat the flag to hook multiple layers."
            ),
        ),
    ] = None,
) -> None:
    """Evaluate a pre-trained model."""
    # If activation saving is enabled, then add appropriate layers
    if save_activations:
        layers = list(activation_layer or [])
        if not layers:
            msg = (
                "At least one --activation-layer must be specified when "
                "--save-activations is set."
            )
            raise typer.BadParameter(msg)
        config.get("evaluate", {}).get("callbacks", {}).get("activation_saver", {})[
            "layer_paths"
        ] = layers

    model = ModelService.from_checkpoint(config, Path(checkpoint).resolve())
    model.evaluate()


if __name__ == "__main__":
    evaluation_cli()
