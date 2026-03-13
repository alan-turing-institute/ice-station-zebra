import logging
from datetime import timedelta
from typing import Any

import earthkit.data as ekd
import numpy as np
import xarray as xr
from anemoi.datasets.create.sources import source_registry
from anemoi.datasets.create.sources.legacy import LegacySource
from anemoi.datasets.create.sources.xarray import load_one
from anemoi.datasets.dates.groups import GroupOfDates
from argopy import DataFetcher
from haversine import Unit, haversine_vector

logger = logging.getLogger(__name__)


@source_registry.register("argo")
class ArgoSource(LegacySource):
    @staticmethod
    def _execute(
        context: dict[str, Any],
        date_group: GroupOfDates,
        area: str,
        grid_resolution_degrees: float = 1,
    ) -> ekd.FieldList:
        """Download Argo float data within given parameters."""
        north, west, south, east = map(float, area.split("/"))
        if (north < south) or (east < west):
            msg = f"Invalid area: {area}. Expected format: 'N/W/S/E'"
            raise ValueError(msg)

        # Set constants
        distance_scale_km = 2000
        min_weight = 1e-10

        # Construct the grid that we want to project onto
        lats = np.arange(
            south + 0.5 * grid_resolution_degrees,
            north + 0.5 * grid_resolution_degrees,
            grid_resolution_degrees,
        )
        lons = np.arange(
            west + 0.5 * grid_resolution_degrees,
            east + 0.5 * grid_resolution_degrees,
            grid_resolution_degrees,
        )

        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
        grid_points = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
        logger.info(
            "Created grid with %d latitudes and %d longitudes, total %d grid points",
            len(lats),
            len(lons),
            len(lats) * len(lons),
        )

        requested_dates = sorted(date_group.dates)
        weighted_data: dict[str, np.ndarray] = {
            variable: np.full(
                (len(requested_dates), len(lats), len(lons)), np.nan, dtype=float
            )
            for variable in ("TEMP", "PSAL")
        }
        logger.info(
            "Fetching Argo data inside [N: %s, W: %s, S: %s, E: %s] between %s and %s.",
            north,
            west,
            south,
            east,
            min(requested_dates),
            max(requested_dates),
        )

        for t_idx, date in enumerate(requested_dates):
            logger.info("Processing data from %s", date.date())
            start_time = date - timedelta(hours=1)
            end_time = date + timedelta(hours=1)

            # Get all data up to the mixed layer
            fetcher = DataFetcher().region(
                [west, east, south, north, 0, 50, start_time, end_time]
            )
            df = fetcher.to_dataframe()

            # Get positions of observations
            obs_lat = df["LATITUDE"].to_numpy(dtype=float)
            obs_lon = df["LONGITUDE"].to_numpy(dtype=float)
            obs_points = np.column_stack((obs_lat, obs_lon))  # shape: (n_obs, 2)

            # Pairwise distances in km
            distance_km: np.ndarray = haversine_vector(
                obs_points,
                grid_points,
                unit=Unit.KILOMETERS,
                comb=True,
                check=False,  # faster if your lat/lon are already valid
            )  # shape: (n_lat*n_lon, n_obs)

            # Construct exponential weights, with a minimum for numerical stability
            weights = np.exp(-0.5 * (distance_km / distance_scale_km) ** 2).clip(
                min=min_weight
            )  # shape: (n_lat*n_lon, n_obs)
            sum_weights = np.sum(weights, axis=1)  # shape: (n_lat*n_lon,)

            # Apply weights to the data
            for variable, weighted_array in weighted_data.items():
                unweighted_data = df[variable].to_numpy(dtype=float)
                weighted_array[t_idx] = (
                    np.matmul(weights, unweighted_data) / sum_weights
                ).reshape(1, len(lats), len(lons))  # shape: (n_lat*n_lon,)

        # Construct an xarray dataset
        ds_out = xr.Dataset(
            data_vars={
                variable: (
                    ("time", "lat", "lon"),
                    data_array,
                    {"coordinates": "time lat lon"},
                )
                for variable, data_array in weighted_data.items()
            },
            coords={
                "time": (
                    "time",
                    np.asarray(requested_dates, dtype="datetime64[ns]"),
                    {
                        "standard_name": "time",
                        "units": "seconds since 1970-01-01 00:00:00",
                        "calendar": "standard",
                    },
                ),
                "lat": (
                    "lat",
                    lats,
                    {
                        "standard_name": "latitude",
                        "units": "degrees_north",
                    },
                ),
                "lon": (
                    "lon",
                    lons,
                    {
                        "standard_name": "longitude",
                        "units": "degrees_east",
                    },
                ),
            },
        )
        field_lists = load_one(
            "📂", context, [date.isoformat() for date in requested_dates], ds_out
        )

        if len(field_lists) != len(requested_dates):
            msg = f"Expected {len(requested_dates)} dates, got {len(field_lists)} dates"
            raise ValueError(msg)

        return field_lists
