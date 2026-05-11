import logging
import time
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from anemoi.datasets.create.input.context import Context
from anemoi.datasets.create.source import Source
from anemoi.datasets.create.sources import source_registry
from anemoi.datasets.create.sources.xarray import load_one
from anemoi.datasets.dates.groups import GroupOfDates
from argopy import DataFetcher
from argopy.errors import NoData
from earthkit.data import FieldList
from fsspec import FSTimeoutError
from haversine import Unit, haversine_vector
from pandas import DataFrame

logger = logging.getLogger(__name__)


@source_registry.register("argo")
class ArgoSource(Source):
    def __init__(  # noqa: PLR0913
        self,
        context: Context,
        *,
        area: str,
        param: list[str],
        grid_resolution_degrees: float = 1,
        distance_scale_km: float = 2000,
        min_weight: float = 1e-10,
        time_half_window_hrs: int = 2,
    ) -> None:
        """Set parameters for fetching and interpolating Argo float data."""
        self.context = context
        self.param = param
        self.grid_resolution_degrees = grid_resolution_degrees
        self.distance_scale_km = distance_scale_km
        self.min_weight = min_weight
        self.time_half_window_hrs = time_half_window_hrs
        self.north, self.west, self.south, self.east = map(float, area.split("/"))
        if (self.north < self.south) or (self.east < self.west):
            msg = f"Invalid area: {area}. Expected format: 'N/W/S/E'"
            raise ValueError(msg)

    def execute(self, dates: list[datetime] | GroupOfDates) -> FieldList:
        """Download Argo float data within given date range."""
        # Set constants
        # Construct the grid that we want to project onto
        lats = np.arange(
            self.south + 0.5 * self.grid_resolution_degrees,
            self.north + 0.5 * self.grid_resolution_degrees,
            self.grid_resolution_degrees,
        )
        lons = np.arange(
            self.west + 0.5 * self.grid_resolution_degrees,
            self.east + 0.5 * self.grid_resolution_degrees,
            self.grid_resolution_degrees,
        )

        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
        grid_points = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
        logger.info(
            "Created grid with %d latitudes and %d longitudes, total %d grid points",
            len(lats),
            len(lons),
            len(lats) * len(lons),
        )

        requested_dates: list[datetime] = sorted(date for date in dates)
        missing_dates: list[datetime] = []

        weighted_data: dict[str, np.ndarray] = {
            variable: np.full(
                (len(requested_dates), len(lats), len(lons)), np.nan, dtype=float
            )
            for variable in self.param
        }
        logger.info(
            "Fetching Argo data inside [N: %s, W: %s, S: %s, E: %s] between %s and %s.",
            self.north,
            self.west,
            self.south,
            self.east,
            min(requested_dates),
            max(requested_dates),
        )

        for t_idx, date in enumerate(requested_dates):
            logger.info("Processing data from %s", date.date())
            start_time = date - timedelta(hours=self.time_half_window_hrs)
            end_time = date + timedelta(hours=self.time_half_window_hrs)

            # Extract data for the mixed layer (top 50m), retry if 503 error from erddap
            region = [self.west, self.east, self.south, self.north, 0, 50]
            time_window = [
                start_time,
                end_time,
            ]
            df = _fetch_argo_dataframe_with_retry(region, time_window)

            if df.empty:
                logger.info("No Argo observations for %s; returning NaNs", date)
                missing_dates.append(date)
                continue

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
            weights = np.exp(-0.5 * (distance_km / self.distance_scale_km) ** 2).clip(
                min=self.min_weight
            )  # shape: (n_lat*n_lon, n_obs)
            sum_weights = np.sum(weights, axis=1)  # shape: (n_lat*n_lon,)

            # Apply weights to the data
            for variable, weighted_array in weighted_data.items():
                unweighted_data = df[variable].to_numpy(dtype=float)
                weighted_array[t_idx] = (
                    np.matmul(weights, unweighted_data) / sum_weights
                ).reshape(len(lats), len(lons))  # shape: (n_lat, n_lon)

        if missing_dates:
            logger.info(
                "%d of %d requested dates were missing:",
                len(missing_dates),
                len(requested_dates),
            )
            for missing_date in missing_dates:
                logger.warning(missing_date.isoformat())

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

        return load_one(
            "📂", self.context, [date.isoformat() for date in requested_dates], ds_out
        )


def _fetch_argo_dataframe_with_retry(
    region: list[float],
    time_window: list[datetime],
    max_retries: int = 5,
    initial_backoff_s: float = 0.5,
) -> DataFrame:
    """Fetch Argo data with exponential backoff.

    Args:
        region: List of [west, east, south, north, depth_min, depth_max]
        time_window: List of [start_time, end_time]
        max_retries: Number of retry attempts for ERDDAP 503 or 500 responses.
        initial_backoff_s: Initial backoff delay in seconds before retrying.

    Returns:
        DataFrame from Argo data

    """
    # Try erddap with exponential backoff
    for attempt in range(1, max_retries + 1):
        try:
            fetcher = DataFetcher().region(region + time_window)
            dataframe = fetcher.to_dataframe()
        except FSTimeoutError as exc:
            # Retry on timeout, with exponential backoff
            logger.info("Failed to fetch Argo data from ERDDAP: %s", type(exc))
            if attempt < max_retries:
                backoff = initial_backoff_s * (2 ** (attempt - 1))
                logger.warning(
                    "ERDDAP data server unavailable, retrying in %.1fs (attempt %d/%d)",
                    backoff,
                    attempt,
                    max_retries,
                )
                time.sleep(backoff)
                continue
            # Otherwise raise an exception
            msg = f"ERDDAP data server failed after {attempt} retries. Error: {exc!s}"
            raise RuntimeError(msg) from exc
        except (FileNotFoundError, NoData):
            logger.warning(
                "Failed to load data for %s between %s and %s.",
                region,
                time_window[0].isoformat(),
                time_window[1].isoformat(),
            )
        except Exception:
            logger.exception("Unexpected error while fetching Argo data")
            raise
        else:
            logger.debug("Successfully fetched data from ERDDAP on attempt %s", attempt)
            return dataframe

    # Return empty DataFrame if file not found (e.g., no data for that region/time)
    return DataFrame()
