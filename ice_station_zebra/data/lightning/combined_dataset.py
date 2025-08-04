from collections.abc import Sequence
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset

from .zebra_dataset import ZebraDataset


class CombinedDataset(Dataset):
    def __init__(
        self,
        datasets: Sequence[ZebraDataset],
        target: str,
        *,
        n_forecast_steps: int = 1,
        n_history_steps: int = 1,
    ) -> None:
        """Constructor"""
        super().__init__()

        # Store the number of forecast and history steps
        self.n_forecast_steps = n_forecast_steps
        self.n_history_steps = n_history_steps

        # Define target and input datasets
        self.target = next(ds for ds in datasets if ds.name == target)
        self.inputs = [ds for ds in datasets]

        # Require that all datasets have the same frequency
        frequencies = sorted(set(ds.dataset.frequency for ds in datasets))
        if len(frequencies) != 1:
            msg = f"Cannot combine datasets with different frequencies: {frequencies}."
            raise ValueError(msg)
        self.frequency = np.timedelta64(frequencies[0])

    def __len__(self) -> int:
        """Return the total length of the dataset"""
        return min([len(ds) for ds in self.inputs] + [len(self.target)])

    def __getitem__(self, idx: int) -> dict[str, NDArray[np.float32]]:
        """Return the data for a single timestep as a dictionary

        Returns:
            A dictionary with dataset names as keys and a numpy array as the value.
            The shape of each array is:
            - input datasets: [n_history_steps, C_input_k, H_input_k, W_input_k]
            - target dataset: [n_forecast_steps, C_target, H_target, W_target]
        """
        return {ds.name: ds[idx] for ds in self.inputs} | {"target": self.target[idx]}

    def date_from_index(self, idx: int) -> datetime:
        """Return the date of the timestep"""
        np_datetime = self.target.dataset.dates[idx]
        return datetime.strptime(str(np_datetime), r"%Y-%m-%dT%H:%M:%S")

    def get_forecast_steps(self, start_date: np.datetime64) -> list[np.datetime64]:
        """Return list of consecutive forecast dates for a given start date."""
        return [
            start_date + (idx + self.n_history_steps) * self.frequency
            for idx in range(self.n_forecast_steps)
        ]

    def get_history_steps(self, start_date: np.datetime64) -> list[np.datetime64]:
        """Return list of consecutive history dates for a given start date."""
        return [
            start_date + idx * self.frequency for idx in range(self.n_history_steps)
        ]

    @property
    def end_date(self) -> np.datetime64:
        """Return the end date of the dataset."""
        end_date = set(dataset.end_date for dataset in self.inputs)
        if len(end_date) != 1:
            msg = f"Datasets have {len(end_date)} different end dates"
            raise ValueError(msg)
        return end_date.pop()

    @property
    def start_date(self) -> np.datetime64:
        """Return the start date of the dataset."""
        start_date = set(dataset.start_date for dataset in self.inputs)
        if len(start_date) != 1:
            msg = f"Datasets have {len(start_date)} different start dates"
            raise ValueError(msg)
        return start_date.pop()
