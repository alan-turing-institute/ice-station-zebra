from ftplib import FTP
import logging
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any

from argopy import DataFetcher
import earthkit.data as ekd
from anemoi.datasets.create.sources import source_registry
from anemoi.datasets.create.sources.legacy import LegacySource
from anemoi.datasets.create.sources.xarray import load_one
from anemoi.datasets.dates.groups import GroupOfDates
from earthkit.data.core.fieldlist import FieldList, MultiFieldList
from earthkit.data.utils.patterns import Pattern
from argopy import DataFetcher
import h5netcdf
import numpy as np
from datetime import datetime

from icenet_mp.utils import to_list


@source_registry.register("argo")
class ArgoSource(LegacySource):
    @staticmethod
    def _execute(
        context: dict[str, Any],
        date_group: GroupOfDates,
        area: str, # "90/-180/0/180" in N, W, S, E format
    ) -> ekd.FieldList:
        """Execute the data loading process from an FTP source."""
        north, west, south, east = map(float, area.split("/"))
        if (north < south) or (east < west):
            raise ValueError(f"Invalid area: {area}. Expected format: 'N/W/S/E'")
        
        # create gridded variable data
        lats = np.arange(south + 0.5, north + 0.5, 10)
        lons = np.arange(west + 0.5, east + 0.5, 10)
        n_lat = len(lats)
        n_lon = len(lons)
        
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
        grid_points = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
        logging.info(f"Created grid with {n_lat} latitudes and {n_lon} longitudes, total {n_lat * n_lon} grid points")
        logging.info(f"shape of lat_grid: {lat_grid.shape}, shape of lon_grid: {lon_grid.shape}, shape of grid_points: {grid_points.shape}")
        
        times = date_group.dates  # unique time steps in the data
        temp_data = np.full((len(times), len(lats), len(lons)), np.nan, dtype=float)
        
        for t_idx, date in enumerate(times):
            logging.info(f"Processing date: {date}")
            start_date = date.replace(hour=11, minute=0, second=0)
            end_date = date.replace(hour=12, minute=59, second=59)
            
            logging.info(f"Fetching Argo data for area {area} and dates from {date}")
            logging.info(f"North: {north}, West: {west}, South: {south}, East: {east}")
            logging.info(f"Start date: {start_date}, End date: {end_date}")
            logging.info(f"type of dates: {type(start_date)}, {type(end_date)}")
                
            fetcher = DataFetcher().region([west, east, south, north, 0, 50, start_date, end_date]) # check depth of Mixed layer
            df = fetcher.to_dataframe()
            
            # df['Date'] = df['TIME'].dt.date
            logging.info(f"Fetched {df} Argo profiles")
            sigma = 2000
        
            for lat_idx, lat in enumerate(lats):
                for lon_idx, lon in enumerate(lons):
                    # logging.info(f"Processing grid cell at lat: {lat}, lon: {lon}")
                    t_weighted, sum_weights = 0, 0
                    for row in df.itertuples():
                        
                        from haversine import Unit, haversine

                        obs_point = (row.LATITUDE, row.LONGITUDE)  # shape: (n_obs, 2)
                        grid_point = (lat, lon)  # shape: (n_lat*n_lon, 2)
                        
                        # Pairwise distances in km: shape (n_obs, n_lat*n_lon)
                        distance_km = haversine(
                            obs_point,
                            grid_point,
                            unit=Unit.KILOMETERS,
                            )
                        distance2 = distance_km ** 2
                        # calculate the weighted distcance
                        weight = np.exp(-0.5 * distance2 / sigma**2)
                        t_weighted += weight * row.TEMP
                        sum_weights += weight
                    temp_data[t_idx, lat_idx, lon_idx] = t_weighted / sum_weights
            
            
            # obs_lat = df["LATITUDE"].to_numpy(dtype=float)
            # obs_lon = df["LONGITUDE"].to_numpy(dtype=float)
            # obs_temp = df["TEMP"].to_numpy(dtype=float)

            # from haversine import Unit, haversine_vector

            # obs_points = np.column_stack((obs_lat, obs_lon))  # shape: (n_obs, 2)
            # logging.info(f"shape of obs_points: {obs_points.shape}")
            
            # # Pairwise distances in km: shape (n_obs, n_lat*n_lon)
            # distance_km = haversine_vector(
            #     obs_points,
            #     grid_points,
            #     unit=Unit.KILOMETERS,
            #     comb=True,
            #     check=False,   # faster if your lat/lon are already valid
            # )
            # logging.info(f"shape of distance_km: {distance_km.shape}")

            # distance_km = distance_km.reshape(len(obs_lat), n_lat, n_lon)
            # distance2 = distance_km ** 2
            # logging.info(f"shape of distance_km after reshape: {distance_km.shape}")
            # logging.info(f"shape of distance2: {distance2.shape}")
            
            # # Numerically stable Gaussian weights:
            # # subtract min distance^2 per grid cell to avoid underflow far from observations
            # min_d2 = np.min(distance2, axis=0, keepdims=True)
            # weights = np.exp(-0.5 * (distance2 - min_d2) / (sigma**2))

            # weighted_sum = np.sum(weights * obs_temp[:, None, None], axis=0)
            # sum_weights = np.sum(weights, axis=0)
            # logging.info(f"shape of weights: {weights.shape}, shape of weighted_sum: {weighted_sum.shape}, shape of sum_weights: {sum_weights.shape}")

            # temp_data[t_idx] = np.divide(
            #     weighted_sum,
            #     sum_weights,
            #     out=np.full((n_lat, n_lon), np.nan, dtype=float),
            #     where=sum_weights > 0,
            # )
            # logging.info(f"temp_data is {temp_data} for date {date}")
            

            print(f"lat_range: {lats}")
            print(f"lon_range: {lons}")
            print(f"time_range: {times}")
            
            import xarray as xr

            times64 = np.asarray(times, dtype="datetime64[ns]")
            
        ds = xr.Dataset(
            data_vars={
                "TEMP": (
                    ("time", "lat", "lon"),
                    temp_data,
                    {"coordinates": "time lat lon"},
                )
            },
            coords={
                "time": (
                    "time",
                    times64,
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

        nc = load_one("📂", context, list(times), ds)
        if len(nc) != len(times):
            raise ValueError(f"Expected {len(times)} dates, got {len(nc)} dates")
        
        return MultiFieldList([nc])
        
        
            # you don't need to create groups first
            # you also don't need to create dimensions first if you supply data
            # with the new variable
            # v = f.create_variable("/grouped/data", ("y",), data=np.arange(10))

            # access and modify attributes with a dict-like interface
            # v.attrs["foo"] = "bar"

            # you can access variables and groups directly using a hierarchical
            # keys like h5py
            # print(f["/grouped/data"])

            # add an unlimited dimension
            # f.dimensions["z"] = None
            # explicitly resize a dimension and all variables using it
            # f.resize_dimension("z", 3)

        
        # # # Parse the FTP URL
        # # server, path_pattern = url.replace("ftp://", "").split("/", 1)

        # # # Get list of remote file paths
        # # remote_paths = {
        # #     date.isoformat(): to_list(
        # #         Pattern(path_pattern).substitute(date=date, allow_extra=True)
        # #     )
        # #     for date in dates
        # # }

        # # # Connect to the FTP server
        # # downloaded_files: list[FieldList] = []
        # # with TemporaryDirectory() as tmpdir, FTP(server) as session:  # noqa: S321
        # #     base_path = Path(tmpdir)
        # #     session.login(user=user, passwd=passwd)

        # #     # Iterate over remote paths
        # #     for iso_date, remote_path_list in remote_paths.items():
        # #         for remote_path in remote_path_list:
        # #             directory, filename = remote_path.rsplit("/", 1)
        # #             session.cwd(("/" + directory).replace("//", "/"))
        # #             local_path = base_path / filename
        # #             # Download the remote file
        # #             with local_path.open("wb") as local_file:
        # #                 session.retrbinary(f"RETR {filename}", local_file.write)
        # #             downloaded_files.append(
        # #                 load_one("📂", context, [iso_date], str(local_path))
        # #             )

        # # Combine all downloaded files into a MultiFieldList
        # return MultiFieldList(downloaded_files)

# put these into a dataframe of lat * long * time and save as netcdf
        
        # make these into a temporary netcdf 
        
        # with h5netcdf.File("mydata.nc", "w") as f:
        #     # set dimensions with a dictionary
        #     # f.dimensions = {"lat": [-1, 0, 1], "lon": [2, 3, 4], "time": date_group.dates}  
        #     f.dimensions = {"time": len(times), "lat": len(lats), "lon": len(lons)}  
        #     # and update them with a dict-like interface
        #     # f.dimensions['x'] = 5
        #     # f.dimensions.update({'x': 5})

        #     times64 = np.asarray(times, dtype="datetime64[s]")
        #     v_time = f.create_variable("time", ("time",), "i8")
        #     v_time.attrs["standard_name"] = "time"
        #     v_time.attrs["units"] = "seconds since 1970-01-01 00:00:00"
        #     v_time.attrs["calendar"] = "standard"
        #     v_time[:] = times64.astype("int64")

        #     v_lat = f.create_variable("lat", ("lat",), float)
        #     v_lat.attrs["standard_name"] = "latitude"
        #     v_lat.attrs["long_name"] = "latitude"
        #     v_lat.attrs["units"] = "degrees_north"
        #     v_lat[:] = lats

        #     v_lon = f.create_variable("lon", ("lon",), float)
        #     v_lon.attrs["standard_name"] = "longitude"
        #     v_lon.attrs["long_name"] = "longitude"
        #     v_lon.attrs["units"] = "degrees_east"
        #     v_lon[:] = lons
                        
        #     temp = f.create_variable("TEMP", ("time", "lat", "lon"), float)
        #     temp.attrs["coordinates"] = "time lat lon"
        #     temp[:] = np.ones((len(times), len(lats), len(lons)))