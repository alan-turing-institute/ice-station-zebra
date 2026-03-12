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
        times = date_group.dates  # unique time steps in the data
        temp_data = np.full((len(times), len(lats), len(lons)), np.nan, dtype=float)
        
        for t_idx, date in enumerate(times):
            logging.info(f"Processing date: {date}")
            start_date = date.replace(hour=11, minute=55, second=0)
            end_date = date.replace(hour=12, minute=5, second=59)
            
            logging.info(f"Fetching Argo data for area {area} and dates from {date_group}")
            logging.info(f"date_group.dates: {date_group.dates}")
            logging.info(f"North: {north}, West: {west}, South: {south}, East: {east}")
            logging.info(f"Start date: {start_date}, End date: {end_date}")
            logging.info(f"type of dates: {type(start_date)}, {type(end_date)}")
                
            fetcher = DataFetcher().region([west, east, south, north, 0, 100, start_date, end_date]) # check depth of Mixed layer
            df = fetcher.to_dataframe()
            
            # df['Date'] = df['TIME'].dt.date
            logging.info(f"Fetched {df} Argo profiles")
            sigma = 100
        
            for lat_idx, lat in enumerate(lats):
                for lon_idx, lon in enumerate(lons):
                    logging.info(f"Processing grid cell at lat: {lat}, lon: {lon}")
                    t_weighted, sum_weights = 0, 0
                    for row in df.itertuples():
                        distance2 = (row.LATITUDE - lat) ** 2 + (row.LONGITUDE - lon) ** 2
                        # calculate the weighted distcance
                        weight = np.exp(-0.5 * distance2 / sigma**2)
                        t_weighted += weight * row.TEMP
                        sum_weights += weight
                    logging.info(f"Sum of weights for lat: {lat}, lon: {lon} is {sum_weights}")
                    logging.info(f"Weighted sum of TEMP for lat: {lat}, lon: {lon} is {t_weighted}")
                    t_weighted /= sum_weights
                    logging.info(f"Gridded value at lat: {lat}, lon: {lon} is {t_weighted}")
                    temp_data[t_idx, lat_idx, lon_idx] = t_weighted / sum_weights

        print(f"lat_range: {lats}")
        print(f"lon_range: {lons}")
        print(f"time_range: {times}")
        
        import xarray as xr

        times64 = np.asarray(times, dtype="datetime64[ns]")
        # temp_data = np.ones((len(times), len(lats), len(lons)), dtype=float)

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