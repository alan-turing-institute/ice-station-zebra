import logging
import re
import time
from collections.abc import Sequence
from itertools import product
from typing import cast

import earthkit.data as ekd
import numpy as np
import pandas as pd
from anemoi.transform.fields import (
    WrappedField,
    new_field_from_numpy,
    new_fieldlist_from_list,
)
from anemoi.transform.filter import Filter
from earthkit.data import Field
from haversine import haversine_vector

from icenet_mp.data_processors.geographic import GeographicField, GeographicGrid

logger = logging.getLogger(__name__)


class Reproject(Filter):
    """A filter to reproject fields to a new grid."""

    output_geography: GeographicGrid

    def __init__(self, *, crs: str, resolution: str, shape: tuple[int, int]) -> None:
        # Generate output grid
        if crs == "EPSG:6932":
            self.output_geography = self.epsg_6932_grid(resolution, *shape)
        else:
            raise ValueError(f"Unsupported output CRS: {crs}")
        self.mapped_indices_h: np.ndarray[tuple[int, int]] | None = None
        self.mapped_indices_w: np.ndarray[tuple[int, int]] | None = None

    def epsg_6932_grid(
        self, resolution: str, h_size: int, w_size: int
    ) -> GeographicGrid:
        # Normalise the resolution
        if (match := re.match(r"^([0-9p]+)([^0-9]+)$", resolution)) is None:
            msg = f"Invalid resolution format: {resolution}"
            raise ValueError(msg)
        scale = float(match.group(1).replace("p", "."))
        unit = match.group(2)
        scale_m = scale * (1000 if unit in ("k", "km") else 1)
        normalised_resolution = str(scale_m / 1000).replace(".", "p") + "km"
        # Get grid positions in EPSG:6932
        h_lim = scale_m * ((h_size - 1) / 2 if h_size % 2 == 0 else h_size // 2)
        w_lim = scale_m * ((w_size - 1) / 2 if w_size % 2 == 0 else w_size // 2)
        h_points = np.linspace(-h_lim, h_lim, h_size)
        w_points = np.linspace(w_lim, -w_lim, h_size)
        return GeographicGrid("EPSG:6932", normalised_resolution, h_points, w_points)

    def build_projection(
        self, data: ekd.FieldList
    ) -> tuple[np.ndarray[tuple[int, int]], np.ndarray[tuple[int, int]]]:
        """Set the input grid based on the input data."""
        # Get the input grid from the data
        if field := next(field for field in data if isinstance(field, Field)):
            lats, lons = field.grid_points()
            input_latlons = np.stack(
                (
                    np.clip(lats, -90, 90).reshape(field.shape),
                    np.clip(lons, -180, 180).reshape(field.shape),
                ),
                axis=-1,
            )
        else:
            raise ValueError("No latitudes/longitudes were found in the input data.")

        # We record the time taken in reprojection as this can be slow
        start = time.perf_counter()
        input_h, input_w = int(input_latlons.shape[0]), int(input_latlons.shape[1])
        output_h, output_w = (
            int(self.output_geography.shape()[0]),
            int(self.output_geography.shape()[1]),
        )
        logger.warning(
            "Calculating reprojection from input grid (%d x %d) to output grid (%d x %d)...",
            input_h,
            input_w,
            output_h,
            output_w,
        )

        # We want to find the closest input grid point for each output grid point. If we
        # try to fully vectorise the call, generating a single array of shape
        # [n_output_latlons, n_input_latlons], then we will run out of memory.
        # Instead we loop over the output points, using argmin to reduce to the closest
        # source point for each output point. We then look up the source grid indices
        # for that source point and store each of the height and width indices in an
        # array of [output_h, output_w]. This allows easy application of the index
        # lookup during the forward pass.
        input_indices = list(product(range(input_h), range(input_w)))
        input_latlons_flat = input_latlons.reshape(-1, 2)
        output_latlons_flat = np.stack(
            (self.output_geography.latitudes(), self.output_geography.longitudes()),
            axis=-1,
        ).reshape(-1, 2)
        closest_src_point_indices = np.array(
            [
                input_indices[
                    np.argmin(
                        haversine_vector(input_latlons_flat, output_latlon, comb=True)
                    )
                ]
                for output_latlon in output_latlons_flat
            ]
        )  # [output_h * output_w, 2]

        # Construct grids of shape [output_h, output_w] that give the indices of the
        # closest input point for each output point
        mapped_indices_h = closest_src_point_indices[:, 0].reshape(output_h, output_w)
        mapped_indices_w = closest_src_point_indices[:, 1].reshape(output_h, output_w)

        logger.info(
            "Reprojection calculation took %.2f seconds", time.perf_counter() - start
        )
        return (mapped_indices_h, mapped_indices_w)

    def forward(self, data: ekd.FieldList | pd.DataFrame) -> ekd.FieldList:
        """Apply the forward regridding transformation.

        Parameters
        ----------
        data : ekd.FieldList
            The input data to be transformed.

        Returns
        -------
        ekd.FieldList
            The transformed data.
        """
        if not isinstance(data, ekd.FieldList):
            raise TypeError(f"Expected data to be a FieldList, but got {type(data)}.")

        if self.mapped_indices_h is None or self.mapped_indices_w is None:
            self.mapped_indices_h, self.mapped_indices_w = self.build_projection(data)

        return new_fieldlist_from_list(
            [
                new_field_from_numpy(
                    field.to_numpy()[self.mapped_indices_h, self.mapped_indices_w],
                    template=WrappedField(
                        GeographicField(field, self.output_geography)
                    ),
                )
                for field in cast(Sequence[Field], data)
            ]
        )
