import typer
from hydra.core.utils import simple_stdout_log_config
from omegaconf import DictConfig

from ice_station_zebra.commands import test, train
from ice_station_zebra.datasets import datasets_cli

from .hydra import hydra_adaptor

# Configure hydra logging
simple_stdout_log_config()

# Create the typer app
app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Entrypoint for zebra application commands",
    no_args_is_help=True,
)
app.add_typer(datasets_cli, name="datasets")


@app.command("test")
@hydra_adaptor
def cmd_test(config: DictConfig) -> None:
    """Test command"""
    test(config)


@app.command("train")
@hydra_adaptor
def cmd_train(config: DictConfig) -> None:
    """Train command"""
    train(config)


if __name__ == "__main__":
    app()
