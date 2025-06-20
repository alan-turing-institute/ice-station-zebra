import typer
from hydra import compose, initialize
from omegaconf import DictConfig

from ice_station_zebra.commands import test, train

# Create the typer app
app = typer.Typer()


def override_base_config(config_name: str, overrides: list[str] | None) -> DictConfig:
    """Apply command-line overrides to base config"""
    with initialize(config_path="config", version_base=None):
        return compose(config_name=config_name, overrides=overrides)


@app.command(name="test")
def cmd_test(overrides: list[str] | None = typer.Argument(None), config_name: str = "zebra") -> None:
    """Test command"""
    test(override_base_config(config_name, overrides))


@app.command("train")
def cmd_train(overrides: list[str] | None = typer.Argument(None), config_name: str = "zebra") -> None:
    """Train command"""
    train(override_base_config(config_name, overrides))


def main() -> None:
    """Command-line entrypoint for zebra application"""
    app()


if __name__ == "__main__":
    main()
