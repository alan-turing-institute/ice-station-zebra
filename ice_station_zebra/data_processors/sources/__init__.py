# from anemoi.transform.filters import filter_registry
from anemoi.datasets.create.sources import source_registry

from .ftp import FTPSource


# Register all filers with anemoi-transform
def register_sources() -> None:
    if "ftp" not in source_registry._aliases:
        source_registry.register("ftp", FTPSource)


__all__ = [
    "FTPSource",
]
