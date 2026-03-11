import logging

from anemoi.datasets.create.sources import source_registry

from .ftp import FTPSource
from .argo import ArgoSource

logger = logging.getLogger(__name__)


def register_sources() -> None:
    """Register all sources with anemoi-datasets."""
    sources = {"ftp": FTPSource,
        "argo": ArgoSource,
    }
    for name, source in sources.items():
        if name not in source_registry.registered:
            source_registry.register(name, source)
            logger.debug(f"Registered {source.__class__.__name__} with anemoi-datasets.")


__all__ = [
    "FTPSource",
    "ArgoSource",
    "register_sources",
]
