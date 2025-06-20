from pathlib import Path

import xarray as xr
from torch.utils.data import Dataset


class OSISAFDataset(Dataset):

    def __init__(self, file_path: Path) -> None:
        # N.B. 'drop_conflicts' is not currently working for combine_attrs
        # One workaround would be to strip attrs from each xarray beforehand
        self.data = xr.concat(
            [xr.open_dataset(path, engine="netcdf4") for path in file_path.glob("*")],
            dim="time",
            coords="minimal",
            compat="identical",
            combine_attrs="drop_conflicts",
            data_vars="all",
        )
        print(self.data)
