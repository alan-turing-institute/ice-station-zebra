import typer
from omegaconf import DictConfig

from .hydra import hydra_override
from ice_station_zebra.commands import test, train

# Create the typer app
app = typer.Typer()


@app.command("test")
@hydra_override
def cmd_test(config: DictConfig) -> None:
    """Test command"""
    test(config)


@app.command("train")
@hydra_override
def cmd_train(config: DictConfig) -> None:
    """Train command"""
    train(config)


if __name__ == "__main__":
    app()
