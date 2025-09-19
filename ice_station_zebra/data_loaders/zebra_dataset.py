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
        return sorted({ds.dates for ds in self.datasets})

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

    @property
    def space(self) -> DataSpace:
        """Return the data space for this dataset."""
        return DataSpace(
            channels=self.datasets[0].shape[1],
            name=self.name,
            shape=self.datasets[0].field_shape,
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
        return self.datasets[0][idx].reshape(self.space.chw)

    def get_tchw(self, dates: Sequence[np.datetime64]) -> ArrayTCHW:
        """Return the data for a series of timesteps in [T, C, H, W] format."""
        return np.stack(
            [self[self.index_from_date(target_date)] for target_date in dates], axis=0
        )

    @cachedmethod(lambda self: self._cache)
    def index_from_date(self, date: np.datetime64) -> int:
        """Return the index of a given date in the dataset."""
        idx, _, _ = self.datasets[0].to_index(date, 0)
        return idx
