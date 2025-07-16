from pathlib import Path

from anemoi.datasets.data import open_dataset
from numpy.typing import NDArray
import numpy as np

class ZebraDataset:
    def __init__(self, input_files: list[Path], *, start: str | None, end: str | None) -> None:
        """Constructor"""
        self.dataset = open_dataset(input_files, start=start, end=end)
        # Dataset shape is: time; variables; ensembles; position
        # We want to reshape a single time point to: variables; ensembles; lat; lon
        self.shape = self.dataset.shape[1:-1] + self.dataset.field_shape

    def __len__(self) -> int:
        """Return the total length of the dataset"""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> NDArray[np.float32]:
        """Return a single timestep after reshaping"""
        return self.dataset[idx].reshape(self.shape)
