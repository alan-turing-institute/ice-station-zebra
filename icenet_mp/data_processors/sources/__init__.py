import logging

from anemoi.datasets.create.sources import source_registry

from .argo import ArgoSource
from .ftp import FTPSource

logger = logging.getLogger(__name__)


def register_sources() -> None:
    """Register all sources with anemoi-datasets."""
    sources = {
        "ftp": FTPSource,
        "argo": ArgoSource,
    }
    for name, source in sources.items():
        if name not in source_registry.registered:
            source_registry.register(name, source)
            logger.debug(
                "Registered %s with anemoi-datasets.", source.__name__
            )


__all__ = [
    "ArgoSource",
    "FTPSource",
    "register_sources",
]
