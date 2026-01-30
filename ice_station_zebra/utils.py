import logging
import os
from datetime import UTC, datetime

import torch
import typer
from lightning.pytorch.loggers import Logger, WandbLogger
from wandb.sdk.lib.runid import generate_id

logger = logging.getLogger(__name__)


def check_mps_fallback(device_type: str) -> None:
    """Check whether MPS fallback is needed and enabled."""
    if device_type != "mps":
        return
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", None):
        logger.info("CPU fallback enabled for Apple Silicon GPU device.")
        return
    msg = (
        "You are attempting to run on MPS without CPU fallback enabled. "
        "The operator 'aten::_upsample_bilinear2d_aa_backward.grad_input' "
        "needed by this code is not yet implemented for the MPS device. "
        "Please rerun after setting the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1`. "
        "Note that this must be set before starting the Python interpreter."
        "WARNING: this will be slower than running natively on MPS."
    )
    logger.error(msg)
    raise typer.Exit(0)


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
