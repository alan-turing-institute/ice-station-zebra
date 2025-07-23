import typer
from hydra.core.utils import simple_stdout_log_config

from ice_station_zebra.data.anemoi import datasets_cli
from ice_station_zebra.training import training_cli

# Configure hydra logging
simple_stdout_log_config()

# Create the typer app
app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Entrypoint for zebra application commands",
    no_args_is_help=True,
)
app.add_typer(datasets_cli, name="datasets")
app.add_typer(training_cli)


if __name__ == "__main__":
    app()
