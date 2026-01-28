import logging

from anemoi.datasets.create.sources import source_registry

from .ftp import FTPSource

logger = logging.getLogger(__name__)


def register_sources() -> None:
    """Register all sources with anemoi-datasets."""
    if "ftp" not in source_registry.registered:
        source_registry.register("ftp", FTPSource)
        logger.debug("Registered FTPSource with anemoi-datasets.")


__all__ = [
    "FTPSource",
]
