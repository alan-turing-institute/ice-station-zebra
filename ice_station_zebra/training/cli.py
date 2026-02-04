import logging

import typer
from omegaconf import DictConfig

from ice_station_zebra.cli import hydra_adaptor
from ice_station_zebra.xpu import xpu_available

from .trainer import ZebraTrainer

# Create the typer app
training_cli = typer.Typer(help="Train models")

log = logging.getLogger(__name__)


@training_cli.command()
@hydra_adaptor
def train(config: DictConfig) -> None:
    """Train a model."""
    if xpu_available():
        config["train"]["trainer"]["accelerator"] = "xpu"
        config["train"]["trainer"]["strategy"] = {
            "_target_": "lightning.pytorch.strategies.SingleDeviceStrategy",
            "device": "xpu:0",
        }
    trainer = ZebraTrainer(config)
    trainer.train()


if __name__ == "__main__":
    training_cli()
