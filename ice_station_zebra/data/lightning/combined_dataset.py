from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset

from .zebra_dataset import ZebraDataset


class CombinedDataset(Dataset):
    def __init__(self, datasets: Sequence[ZebraDataset], *, target: str) -> None:
        """Constructor"""
        super().__init__()
        self.target = next(ds for ds in datasets if ds.name == target)
        self.datasets = [ds for ds in datasets if ds != self.target]

    def __len__(self) -> int:
        """Return the total length of the dataset"""
        return min([len(ds) for ds in self.datasets] + [len(self.target)])

    def __getitem__(self, idx: int) -> NDArray[np.float32]:
        """Return a single timestep"""
        return tuple([ds[idx] for ds in self.datasets] + [self.target[idx]])

    def date_from_index(self, idx: int) -> np.datetime64:
        """Return the date of the timestep"""
        return self.target.dataset.dates[idx]

    @property
    def end_date(self) -> np.datetime64:
        """Return the end date of the dataset."""
        end_date = set(dataset.end_date for dataset in self.datasets)
        if len(end_date) != 1:
            msg = f"Datasets have {len(end_date)} different end dates"
            raise ValueError(msg)
        return end_date.pop()

    @property
    def start_date(self) -> np.datetime64:
        """Return the start date of the dataset."""
        start_date = set(dataset.start_date for dataset in self.datasets)
        if len(start_date) != 1:
            msg = f"Datasets have {len(start_date)} different start dates"
            raise ValueError(msg)
        return start_date.pop()
