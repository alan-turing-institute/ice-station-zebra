from collections.abc import Sequence
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
        date_ranges: Sequence[tuple[str | None, str | None]] = [(None, None)],
    ) -> None:
        """A dataset for use by Zebra.

        The underlying Anemoi dataset has shape [T; C; ensembles; position].
        We reshape this to CHW before returning.
        """
        super().__init__()
        self._cache: LRUCache = LRUCache(maxsize=128)
        self._datasets: list[AnemoiDataset] = []
        self._end = date_ranges[0][1]
        self._input_files = input_files
        self._name = name
        self._start = date_ranges[0][0]

    @property
    def datasets(self) -> list[AnemoiDataset]:
        """Load the underlying Anemoi datasets."""
        if not self._datasets:
            _dataset = open_dataset(self._input_files, start=self._start, end=self._end)
            _dataset._name = self._name
            self._datasets.append(_dataset)
        return self._datasets

    @property
    def dates(self) -> list[np.datetime64]:
        """Return all dates in the dataset."""
        return list(self.datasets[0].dates)

    @property
    def end_date(self) -> np.datetime64:
        """Return the end date of the dataset."""
        return self.datasets[0].end_date

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
        return self.datasets[0].start_date

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
