from datetime import UTC, datetime

import numpy as np
import torch
from lightning.pytorch.loggers import Logger, WandbLogger
from wandb.sdk.lib.runid import generate_id


def generate_run_name() -> str:
    """Generate a unique run name based on the current timestamp."""
    return f"run-{get_timestamp()}-{generate_id()}"


def get_device_name(accelerator_name: str) -> str:
    """Get the device name for the given accelerator."""
    if accelerator_name == "cuda":
        try:
            return torch.cuda.get_device_name()
        except AssertionError:
            return "Unknown CUDA device"
    if accelerator_name == "mps":
        return "Apple Silicon GPU"
    if accelerator_name == "xpu":
        try:
            return torch.xpu.get_device_name()
        except AssertionError:
            return "Unknown XPU device"
    return "CPU"


def get_device_threads() -> int:
    """Get the number of threads available for the current device."""
    return torch.get_num_threads()


def get_timestamp() -> str:
    """Return the current time as a string."""
    return datetime.now(tz=UTC).strftime(r"%Y%m%d_%H%M%S")


def get_wandb_logger(lightning_loggers: list[Logger]) -> WandbLogger | None:
    """Get the WandbLogger from a list of Lightning loggers."""
    for logger in lightning_loggers:
        if isinstance(logger, WandbLogger):
            return logger
    return None

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
