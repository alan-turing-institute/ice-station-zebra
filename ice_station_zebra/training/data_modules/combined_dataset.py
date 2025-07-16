from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset


class CombinedDataset(Dataset):
    def __init__(self, datasets: Sequence[Dataset]) -> None:
        """Constructor"""
        super().__init__()
        self.datasets = [d for d in datasets]

    def __len__(self) -> int:
        """Return the total length of the dataset"""
        return min(len(d) for d in self.datasets)

    def __getitem__(self, idx: int) -> NDArray[np.float32]:
        """Return a single timestep"""
        return tuple(d[idx] for d in self.datasets)
