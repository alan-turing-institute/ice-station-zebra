from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
from torch.utils.data import Dataset

from ice_station_zebra.types import ArrayTCHW

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
        """Initialise a combined dataset from a sequence of ZebraDatasets.

        One of the datasets must be the target and all must have the same frequency. The
        number of forecast and history steps can be set, which will determine the shape
        of the NTCHW tensors returned by __getitem__.
        """
        super().__init__()

        # Store the number of forecast and history steps
        self.n_forecast_steps = n_forecast_steps
        self.n_history_steps = n_history_steps

        # Define target and input datasets
        self.target = next(ds for ds in datasets if ds.name == target)
        self.inputs = list(datasets)

        # Require that all datasets have the same frequency
        frequencies = sorted({ds.frequency for ds in datasets})  # type: ignore[type-var]
        if len(frequencies) != 1:
            msg = f"Cannot combine datasets with different frequencies: {frequencies}."
            raise ValueError(msg)
        self.frequency = frequencies[0]

        # Get list of dates that are available in all datasets
        self.available_dates = sorted(
            [
                available_date
                # Iterate over all dates in any dataset
                for available_date in sorted(
                    {date for ds in datasets for date in ds.dates}
                )  # type: ignore[type-var]
                # Check that all inputs have n_history_steps starting on start_date
                if all(
                    date in ds.dates
                    for date in self.get_history_steps(available_date)
                    for ds in self.inputs
                )
                # Check that the target has n_forecast_steps starting after the history dates
                and all(
                    date in self.target.dates
                    for date in self.get_forecast_steps(available_date)
                )
            ]
        )

    @property
    def end_date(self) -> np.datetime64:
        """Return the end date of the dataset."""
        return self.available_dates[-1]

    @property
    def start_date(self) -> np.datetime64:
        """Return the start date of the dataset."""
        return self.available_dates[0]

    def __len__(self) -> int:
        """Return the total length of the dataset."""
        return len(self.available_dates)

    def __getitem__(self, idx: int) -> dict[str, ArrayTCHW]:
        """Return the data for a single timestep as a dictionary.

        Returns:
            A dictionary with dataset names as keys and a numpy array as the value.
            The shape of each array is:
            - input datasets: [n_history_steps, C_input_k, H_input_k, W_input_k]
            - target dataset: [n_forecast_steps, C_target, H_target, W_target]

        """
        return {
            ds.name: ds.get_tchw(self.get_history_steps(self.available_dates[idx]))
            for ds in self.inputs
        } | {
            "target": self.target.get_tchw(
                self.get_forecast_steps(self.available_dates[idx])
            )
        }

    def date_from_index(self, idx: int) -> datetime:
        """Return the date of the timestep."""
        np_datetime = self.available_dates[idx]
        return datetime.strptime(str(np_datetime), r"%Y-%m-%dT%H:%M:%S").astimezone(UTC)

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
