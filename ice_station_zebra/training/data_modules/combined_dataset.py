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
