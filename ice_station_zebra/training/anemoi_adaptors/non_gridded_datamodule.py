from collections.abc import Callable

from anemoi.training.data.datamodule import AnemoiDatasetsDataModule

from .non_gridded_dataset import NonGriddedDataset


class NonGriddedDataModule(AnemoiDatasetsDataModule):
    def _get_dataset(
        self,
        data_reader: Callable,
        shuffle: bool | None = None,
        val_rollout: int = 1,
        label: str | None = None,
    ) -> NonGriddedDataset:
        """A non-gridded version of Anemoi's Datasets DataModule"""
        data_reader = self.add_trajectory_ids(data_reader)

        kwargs = {}
        if shuffle is not None:
            kwargs["shuffle"] = shuffle
        if label is not None:
            kwargs["label"] = label

        return NonGriddedDataset(
            data_reader=data_reader,
            relative_date_indices=self.relative_date_indices(val_rollout),
            timestep=self.config.data.timestep,
            **kwargs,
        )
