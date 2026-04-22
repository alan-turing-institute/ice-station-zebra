from collections.abc import Iterable
from datetime import datetime as Datetime
from typing import Any

from earthkit.data.core.geography import Geography
from earthkit.data.core.metadata import Metadata
from typing_extensions import override as override_

from .geographic_grid import GeographicGrid


class GeographicMetadata(Metadata):
    def __init__(self, metadata: Metadata, geography: GeographicGrid) -> None:
        self.metadata_ = metadata
        self.geography_ = geography

    @property
    @override_
    def geography(self) -> Geography:
        return self.geography_

    @override_
    def __contains__(self, key) -> bool:
        return key in self.metadata_

    @override_
    def __iter__(self) -> Iterable:
        """Return an iterator over the metadata keys."""
        return iter(self.metadata_)

    @override_
    def __len__(self) -> int:
        return len(self.metadata_)

    @override_
    def __repr__(self):
        return f"{self.__class__.__name__}({self.metadata_!r},{self.geography_!r})"

    @override_
    def _hide_internal_keys(self) -> Metadata:
        return self.metadata_._hide_internal_keys()

    @override_
    def as_namespace(self, namespace=None) -> dict[str, Any]:
        return self.metadata_.as_namespace(namespace)

    @override_
    def base_datetime(self) -> Datetime:
        return self.metadata_.base_datetime()

    @override_
    def data_format(self) -> str:
        return self.metadata_.data_format()

    @override_
    def datetime(self) -> dict[str, Datetime]:
        return self.metadata_.datetime()

    @override_
    def describe_keys(self) -> list[str]:
        return self.metadata_.describe_keys()

    @override_
    def dump(self, **kwargs):
        return self.metadata_.dump(**kwargs)

    @override_
    def get(
        self,
        key,
        default=None,
        *,
        astype: type | None = None,
        raise_on_missing: bool = False,
    ):
        kwargs = {}
        if not raise_on_missing:
            kwargs["default"] = default
        result = self.metadata_.get(key, **kwargs)
        if astype is None:
            return result
        try:
            return astype(result)
        except Exception as exc:
            msg = f"Failed to convert metadata key '{key}' to type {astype}: {exc}"
            raise ValueError(msg) from exc

    @override_
    def index_keys(self) -> list[str]:
        return self.metadata_.index_keys()

    @override_
    def items(self):
        return self.metadata_.items()

    @override_
    def keys(self):
        return self.metadata_.keys()

    @override_
    def ls_keys(self) -> list[str]:
        return self.metadata_.ls_keys()

    @override_
    def namespaces(self) -> list[str]:
        return self.metadata_.namespaces()

    @override_
    def override(self, *args, **kwargs):
        return self.metadata_.override(*args, **kwargs)

    @override_
    def valid_datetime(self) -> Datetime:
        return self.metadata_.valid_datetime()
