import typer
from omegaconf import DictConfig

from ice_station_zebra.cli import hydra_adaptor

# Create the typer app
datasets_cli = typer.Typer(help="Manage datasets")


@datasets_cli.command("create")
@hydra_adaptor
def create(
    config: DictConfig,
    dataset_name: str,
) -> None:
    """Create a dataset"""
    pass


if __name__ == "__main__":
    datasets_cli()
