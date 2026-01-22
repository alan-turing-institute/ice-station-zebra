from datetime import UTC, datetime

import numpy as np
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


def to_bool(v: object) -> bool:
    """Convert a text string to a boolean."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true"}:
            return True
        if s in {"false"}:
            return False
    msg = f"Cannot convert {v!r} to bool"
    raise ValueError(msg)


def parse_np_datetime(dt: np.datetime64) -> datetime:
    """Convert numpy-like datetime to aware datetime in UTC.

    Accepts objects such as ``np.datetime64``, plain ``datetime`` or their string
    representations. Supports values with and without microseconds.
    """
    if isinstance(dt, datetime):
        return dt.astimezone(UTC)

    date_str = str(dt)
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(date_str, fmt).astimezone(UTC)
        except ValueError:
            continue

    msg = f"Cannot parse datetime value {dt!r}"
    raise ValueError(msg)
