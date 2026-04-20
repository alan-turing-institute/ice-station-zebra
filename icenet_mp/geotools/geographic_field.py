from typing import Any, cast

import numpy as np
from earthkit.data import Field
from earthkit.data.core.metadata import Metadata

from .geographic_grid import GeographicGrid
from .geographic_metadata import GeographicMetadata


class GeographicField(Field):
    def __init__(self, field: Field, geography: GeographicGrid) -> None:
        """Initialise a GeographicField from an EarthKit Field and a GeographicGrid."""
        self._field = field
        self.geo_metadata = GeographicMetadata(
            cast("Metadata", field.metadata()), geography
        )

    def __repr__(self) -> str:
        """Return a string representation of the GeographicField, including the underlying field."""
        return f"{self.__class__.__name__}({self._field!r})"

    @property
    def _metadata(self) -> Metadata:
        return self.geo_metadata

    def _values(self, dtype: Any | None = None) -> Any:  # noqa: ANN401
        return self._field._values(dtype)

    def clone(
        self,
        *,
        values: Any | None = None,  # noqa: ANN401
        metadata: Metadata | None = None,
        **kwargs: Any,
    ) -> "GeographicField":
        return GeographicField(
            self._field.clone(values=values, metadata=metadata, **kwargs),
            self.geo_metadata.geography,
        )

    def grid_points(self) -> tuple[np.ndarray, np.ndarray]:
        return self._field.grid_points()

    def message(self) -> bytes:
        return self._field.message()

    def to_latlon(self, flatten: bool = True) -> dict[str, Any]:  # noqa: FBT001, FBT002
        return self._field.to_latlon(flatten=flatten)

    def to_numpy(
        self,
        flatten: bool = False,  # noqa: FBT001, FBT002
        dtype: type | None = None,
        index: int | None = None,
    ) -> np.ndarray[Any]:
        return self._field.to_numpy(flatten=flatten, dtype=dtype, index=index)
