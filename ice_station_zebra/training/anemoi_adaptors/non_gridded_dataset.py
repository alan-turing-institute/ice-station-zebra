from collections.abc import Callable

from anemoi.training.data.dataset import NativeGridDataset
from anemoi.training.data.grid_indices import BaseGridIndices


class NonGriddedDataset(NativeGridDataset):
    """Iterable dataset for AnemoI data on the arbitrary grids."""

    def __init__(
        self,
        data_reader: Callable,
        relative_date_indices: list,
        timestep: str | None = None,
        shuffle: bool | None = None,
        label: str | None = None,
    ) -> None:
        """Non-gridded version of the Anemoi dataloader"""
        kwargs = {}
        if timestep is not None:
            kwargs["timestep"] = timestep
        if shuffle is not None:
            kwargs["shuffle"] = shuffle
        if label is not None:
            kwargs["label"] = label
        super().__init__(
            data_reader,
            grid_indices=type[BaseGridIndices],
            relative_date_indices=relative_date_indices,
            **kwargs,
        )
