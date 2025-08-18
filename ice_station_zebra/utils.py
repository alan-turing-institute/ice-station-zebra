from datetime import UTC, datetime

from lightning.pytorch.loggers import Logger, WandbLogger
from wandb.sdk.lib.runid import generate_id


def generate_run_name() -> str:
    """Generate a unique run name based on the current timestamp."""
    return f"run-{get_timestamp()}-{generate_id()}"


def get_timestamp() -> str:
    """Return the current time as a string."""
    return datetime.now(tz=UTC).strftime(r"%Y%m%d_%H%M%S")


def get_wandb_logger(lightning_loggers: list[Logger]) -> WandbLogger | None:
    """Get the WandbLogger from a list of Lightning loggers."""
    for logger in lightning_loggers:
        if isinstance(logger, WandbLogger):
            return logger
    return None
