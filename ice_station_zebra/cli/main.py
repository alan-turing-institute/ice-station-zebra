"""Main entrypoint for the CLI application."""

import logging

import typer
from hydra.core.utils import simple_stdout_log_config

from ice_station_zebra.data_processors.cli import datasets_cli
from ice_station_zebra.evaluation.cli import evaluation_cli
from ice_station_zebra.training.cli import training_cli

# Configure logging
simple_stdout_log_config()
logger = logging.getLogger(__name__)

# Create the typer app
app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Entrypoint for zebra application commands",
    no_args_is_help=True,
)
app.add_typer(datasets_cli, name="datasets")
app.add_typer(evaluation_cli)
app.add_typer(training_cli)


def main() -> None:
    """Initialise and run the CLI application."""
    # Run the app
    try:
        app()
    except NotImplementedError as exc:
        # Catch MPS-not-implemented errors
        if "not currently implemented for the MPS device" in str(exc):
            msg = (
                "WARNING: job failed due to running on MPS without CPU fallback enabled.\n"
                "Please rerun after setting the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1`. "
                "This *must* be set before starting the Python interpreter. "
                "It will be slower than running natively on MPS."
            )
            logger.error(msg)  # noqa: TRY400
            typer.Exit(1)


if __name__ == "__main__":
    main()
