from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset

from .zebra_dataset import ZebraDataset


class CombinedDataset(Dataset):
    def __init__(self, datasets: Sequence[ZebraDataset], *, target: str) -> None:
        """Constructor"""
        super().__init__()
        self.target = next(d for d in datasets if d.name == target)
        self.datasets = [d for d in datasets if d != self.target]

    def __len__(self) -> int:
        """Return the total length of the dataset"""
        return min([len(d) for d in self.datasets] + [len(self.target)])

    def __getitem__(self, idx: int) -> NDArray[np.float32]:
        """Return a single timestep"""
        return tuple([d[idx] for d in self.datasets] + [self.target[idx]])
