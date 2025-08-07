from pathlib import Path
from collections.abc import Sequence

import numpy as np
from anemoi.datasets.data import open_dataset
from cachetools import LRUCache, cachedmethod
from torch.utils.data import Dataset

from ice_station_zebra.types import ArrayCHW, ArrayTCHW, DataSpace


class ZebraDataset(Dataset):
    def __init__(
        self,
        name: str,
        input_files: list[Path],
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> None:
        """A dataset for use by Zebra

        Dataset shape is: time; variables; ensembles; position
        We reshape each time point to: variables; pos_x; pos_y
        """
        super().__init__()
        self.dataset = open_dataset(input_files, start=start, end=end)
        self.dataset._name = name
        self.chw = (self.space.channels, *self.space.shape)
        self._cache: LRUCache = LRUCache(maxsize=128)

    @property
    def end_date(self) -> np.datetime64:
        """Return the end date of the dataset."""
        return self.dataset.end_date

    @property
    def name(self) -> str:
        """Return the name of the dataset."""
        return self.dataset.name or "unnamed"

    @property
    def space(self) -> DataSpace:
        """Return the data space for this dataset."""
        return DataSpace(
            channels=self.dataset.shape[1],
            name=self.name,
            shape=self.dataset.field_shape,
        )

    @property
    def start_date(self) -> np.datetime64:
        """Return the start date of the dataset."""
        return self.dataset.start_date

    def __len__(self) -> int:
        """Return the total length of the dataset"""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> ArrayCHW:
        """Return the data for a single timestep in [C, H, W] format"""
        return self.dataset[idx].reshape(self.chw)

    @cachedmethod(lambda self: self._cache)
    def index_from_date(self, date: np.datetime64) -> int:
        """Return the index of a given date in the dataset."""
        idx, _, _ = self.dataset.to_index(date, 0)
        return idx

    def get_tchw(self, dates: Sequence[np.datetime64]) -> ArrayTCHW:
        """Return the data for a series of timesteps in [T, C, H, W] format"""
        return np.stack(
            [self[self.index_from_date(target_date)] for target_date in dates], axis=0
        )
