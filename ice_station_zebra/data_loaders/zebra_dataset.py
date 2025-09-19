from collections.abc import Sequence
from functools import cached_property
from pathlib import Path

import numpy as np
from anemoi.datasets.data import open_dataset
from anemoi.datasets.data.dataset import Dataset as AnemoiDataset
from cachetools import LRUCache, cachedmethod
from torch.utils.data import Dataset

from ice_station_zebra.types import ArrayCHW, ArrayTCHW, DataSpace


class ZebraDataset(Dataset):
    def __init__(
        self,
        name: str,
        input_files: list[Path],
        *,
        date_ranges: Sequence[dict[str, str | None]] = [{"start": None, "end": None}],
    ) -> None:
        """A dataset for use by Zebra.

        The underlying Anemoi dataset has shape [T; C; ensembles; position].
        We reshape this to CHW before returning.
        """
        super().__init__()
        self._cache: LRUCache = LRUCache(maxsize=128)
        self._datasets: list[AnemoiDataset] = []
        self._date_ranges = sorted(
            date_ranges, key=lambda dr: "" if dr["start"] is None else dr["start"]
        )
        self._input_files = input_files
        self._name = name

    @property
    def datasets(self) -> list[AnemoiDataset]:
        """Load one or more underlying Anemoi datasets.

        Each date range results in a separate dataset.
        """
        if not self._datasets:
            for date_range in self._date_ranges:
                _dataset = open_dataset(
                    self._input_files, start=date_range["start"], end=date_range["end"]
                )
                _dataset._name = self._name
                self._datasets.append(_dataset)
        return self._datasets

    @cached_property
    def dates(self) -> list[np.datetime64]:
        """Return all dates in the dataset."""
        return sorted({date for ds in self.datasets for date in ds.dates})

    @cached_property
    def end_date(self) -> np.datetime64:
        """Return the end date of the dataset."""
        return self.dates[-1]

    @property
    def frequency(self) -> np.timedelta64:
        """Return the frequency of the dataset."""
        return np.timedelta64(self.datasets[0].frequency)

    @property
    def name(self) -> str:
        """Return the name of the dataset."""
        return self._name

    @cached_property
    def space(self) -> DataSpace:
        """Return the data space for this dataset."""
        # Check all datasets have the same number of channels
        per_ds_channels = sorted({ds.shape[1] for ds in self.datasets})
        if len(per_ds_channels) != 1:
            msg = f"All datasets must have the same number of channels, found {len(per_ds_channels)} different values"
            raise ValueError(msg)
        # Check all datasets have the same shape
        per_ds_shape = sorted({ds.field_shape for ds in self.datasets})
        if len(per_ds_shape) != 1:
            msg = f"All datasets must have the same shape, found {len(per_ds_shape)} different values"
            raise ValueError(msg)
        # Return the data space
        return DataSpace(
            channels=per_ds_channels[0],
            name=self.name,
            shape=per_ds_shape[0],
        )

    @property
    def start_date(self) -> np.datetime64:
        """Return the start date of the dataset."""
        return self.dates[0]

    def __len__(self) -> int:
        """Return the total length of the dataset."""
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx: int) -> ArrayCHW:
        """Return the data for a single timestep in [C, H, W] format."""
        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset[idx].reshape(self.space.chw)
            idx -= len(dataset)
        msg = f"Index {idx} out of range for dataset of length {len(self)}"
        raise IndexError(msg)

    def get_tchw(self, dates: Sequence[np.datetime64]) -> ArrayTCHW:
        """Return the data for a series of timesteps in [T, C, H, W] format."""
        return np.stack(
            [self[self.index_from_date(target_date)] for target_date in dates], axis=0
        )

    @cachedmethod(lambda self: self._cache)
    def index_from_date(self, date: np.datetime64) -> int:
        """Return the index of a given date in the dataset."""
        idx = 0
        for dataset in self.datasets:
            if date not in dataset.dates:
                idx += len(dataset)
            else:
                ds_idx, _, _ = dataset.to_index(date, 0)
                idx += ds_idx
                break
        return idx
