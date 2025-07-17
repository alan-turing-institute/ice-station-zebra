from pathlib import Path
from typing import Any

import numpy as np
from anemoi.datasets.data import open_dataset
from numpy.typing import NDArray
from torch.utils.data import Dataset


class ZebraDataset(Dataset):
    def __init__(
        self,
        name: str,
        input_files: list[Path],
        *,
        start: str | None = None,
        end: str | None = None,
        **kwargs: Any,
    ) -> None:
        """A dataset for use by Zebra

        Dataset shape is: time; variables; ensembles; position
        We reshape each time point to: variables; pos_x; pos_y
        """
        super().__init__(**kwargs)
        self.dataset = open_dataset(input_files, start=start, end=end)
        self.dataset._name = name
        self.shape = (self.dataset.shape[1], *self.dataset.field_shape)

    @property
    def name(self) -> str | None:
        return self.dataset.name

    def __len__(self) -> int:
        """Return the total length of the dataset"""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> NDArray[np.float32]:
        """Return a single timestep after reshaping"""
        return self.dataset[idx].reshape(self.shape)
