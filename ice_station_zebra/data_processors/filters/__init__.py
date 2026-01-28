import logging

from anemoi.transform.filters import filter_registry

from .doubling_filter import DoublingFilter

logger = logging.getLogger(__name__)


def register_filters() -> None:
    """Register all filters with anemoi-transform."""
    if "doubling_filter" not in filter_registry.registered:
        filter_registry.register("doubling_filter", DoublingFilter)
        logger.debug("Registered DoublingFilter with anemoi-transform.")


__all__ = [
    "DoublingFilter",
]
