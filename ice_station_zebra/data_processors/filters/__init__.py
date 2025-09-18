from anemoi.transform.filters import filter_registry

from .doubling_filter import DoublingFilter


# Register all filers with anemoi-transform
def register_filters() -> None:
    filter_registry.register("doubling_filter", DoublingFilter)


__all__ = [
    "DoublingFilter",
]
