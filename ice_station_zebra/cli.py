import typer
from hydra import compose, initialize
from omegaconf import DictConfig

from ice_station_zebra.commands import test, train

# Create the typer app
app = typer.Typer()


def hydra_override(function):
    def wrapper(
        overrides: list[str] | None = typer.Argument(None), config_name: str = "zebra"
    ):
        with initialize(config_path="config", version_base=None):
            config = compose(config_name=config_name, overrides=overrides)
        return function(config)

    return wrapper


@app.command(name="test")
@hydra_override
def cmd_test(config: DictConfig) -> None:
    """Test command"""
    test(config)


@app.command("train")
@hydra_override
def cmd_train(config: DictConfig) -> None:
    """Train command"""
    train(config)


def main() -> None:
    """Command-line entrypoint for zebra application"""
    app()


if __name__ == "__main__":
    main()
