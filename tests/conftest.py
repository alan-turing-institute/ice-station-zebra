import datetime
from pathlib import Path
from typing import Any

import pytest
import xarray as xr
from anemoi.datasets.commands.create import Create
from omegaconf import DictConfig

from ice_station_zebra.types import AnemoiCreateArgs


@pytest.fixture(scope="session")
def mock_data() -> dict[str, dict[str, Any]]:
    """Fixture to create a mock dataset for testing."""
    return {
        "coords": {
            "lat": {
                "dims": ("lat"),
                "attrs": {"units": "degrees_north", "standard_name": "latitude"},
                "data": [-89, -90],
            },
            "lon": {
                "dims": ("lon"),
                "attrs": {"units": "degrees_east", "standard_name": "longitude"},
                "data": [44, 45],
            },
            "time": {
                "dims": ("time",),
                "attrs": {"standard_name": "time"},
                "data": [
                    datetime.datetime(2020, 1, 1, 0, 0, 0),
                    datetime.datetime(2020, 1, 2, 0, 0, 0),
                    datetime.datetime(2020, 1, 3, 0, 0, 0),
                ],
            },
        },
        "attrs": {},
        "dims": {"lat": 2, "lon": 2, "time": 3},
        "data_vars": {
            "ice_conc": {
                "dims": ("time", "lat", "lon"),
                "attrs": {},
                "data": [
                    [[0.5, 1.0], [0.4, 0.0]],
                    [[0.4, 0.9], [0.3, 0.1]],
                    [[0.3, 0.8], [0.2, 0.2]],
                ],
            }
        },
    }


@pytest.fixture(scope="session")
def mock_dataset(
    tmpdir_factory: pytest.TempdirFactory, mock_data: dict[str, dict[str, Any]]
) -> Path:
    """Fixture to create a mock file for testing."""
    # Use the mock data to create a NetCDF file
    netcdf_path = tmpdir_factory.mktemp("data").join("dummy.nc")
    xr.Dataset.from_dict(mock_data).to_netcdf(netcdf_path)
    config = DictConfig(
        {
            "dates": {
                "start": "2020-01-01T00:00:00",
                "end": "2020-01-03T23:00:00",
                "frequency": "24h",
            },
            "input": {
                "netcdf": {
                    "path": str(netcdf_path),
                }
            },
        }
    )
    # Create an Anemoi dataset from the NetCDF file
    zarr_path = tmpdir_factory.mktemp("data").join("dummy.zarr")
    Create().run(
        AnemoiCreateArgs(
            path=str(zarr_path),
            config=config,
            overwrite=True,
        )
    )
    return Path(str(zarr_path))
