import logging
from typing import Any

import torch
from lightning.fabric.accelerators.registry import _AcceleratorRegistry
from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning.fabric.utilities.types import _DEVICE
from lightning.pytorch.accelerators import Accelerator, AcceleratorRegistry
from torch import distributed as dist
from typing_extensions import override

logger = logging.getLogger(__name__)


class XPUAccelerator(Accelerator):
    """Accelerator for Intel XPU GPU devices."""

    @override
    def setup_device(self, device: torch.device) -> None:
        """Configure the current process to use a specified device.

        Raises:
            MisconfigurationException: If the selected device is not an xpu.
        """
        if device.type != "xpu":
            msg = f"Device should be xpu, got {device} instead."
            raise MisconfigurationException(msg)
        index = getattr(device, "index", None)
        if not isinstance(index, int):
            raise MisconfigurationException("Device index could not be determined.")
        torch.xpu.set_device(index - 1)

    @override
    def get_device_stats(self, device: _DEVICE) -> dict[str, Any]:
        """Get stats from torch xpu module."""
        return torch.xpu.memory_stats(device)

    @override
    def teardown(self) -> None:
        """Ensure that distributed processes close gracefully."""
        torch.xpu.empty_cache()
        if dist.is_initialized():
            dist.destroy_process_group()

    @staticmethod
    @override
    def parse_devices(devices: int | list[int]) -> list[int]:
        """Accelerator device parsing logic.

        Args:
            devices: Device(s) by number

        Returns:
            List of device numbers to use
        """
        if isinstance(devices, int):
            devices = [devices]
        return devices

    @staticmethod
    @override
    def get_parallel_devices(devices: list[int]) -> list[torch.device]:
        """Get parallel devices for the Accelerator.

        Args:
            devices: List of device numbers

        Returns:
            List of devices
        """
        return [torch.device("xpu", i - 1) for i in devices]

    @staticmethod
    @override
    def auto_device_count() -> int:
        """Get number of available devices from torch xpu module."""
        return torch.xpu.device_count()

    @staticmethod
    @override
    def is_available() -> bool:
        """Determines if an XPU is actually available.

        Returns:
            True if devices are detected, otherwise False
        """
        try:
            return torch.xpu.device_count() > 0
        except (AttributeError, NameError):
            return False

    @staticmethod
    @override
    def name() -> str:
        return "xpu"

    @classmethod
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register(
            "xpu",
            cls,
            description="Intel Data Center GPU Max - codename Ponte Vecchio",
        )


# Add this accelerator to the registry
def xpu_available() -> bool:
    """Register the XPU accelerator with Lightning."""
    if "xpu" not in AcceleratorRegistry.available_accelerators():
        XPUAccelerator.register_accelerators(AcceleratorRegistry)
    return XPUAccelerator.is_available()
