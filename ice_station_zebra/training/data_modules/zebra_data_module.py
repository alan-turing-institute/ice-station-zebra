from pathlib import Path

from lightning import LightningDataModule
from numpy.typing import NDArray
from torch.utils.data import DataLoader
import numpy as np

from .dataloader_args import DataloaderArgs
from .zebra_dataset import ZebraDataset

class ZebraDataModule(LightningDataModule):
    def __init__(self, input_paths: list[Path], batch_size: int = 32) -> None:
        super().__init__()
        self.input_paths = [input_path.resolve() for input_path in input_paths]
        self.batch_size = batch_size
        self.train_period = [None, "2019-12-31"]
        self.val_period = ["2020-01-01", "2020-01-15"]
        self.test_period = ["2020-01-16", None]

        # Set common arguments for the dataloader
        self._common_dataloader_kwargs = DataloaderArgs(
            batch_size=batch_size,
            sampler=None,
            batch_sampler=None,
            drop_last=False,
            worker_init_fn=None,
        )

    def train_dataloader(self) -> DataLoader[tuple[NDArray[np.float32], NDArray[np.float32]]]:
        """Construct train dataloader"""
        dataset = ZebraDataset(self.input_paths, start=self.train_period[0], end=self.train_period[1])
        return DataLoader(dataset, shuffle=True, **self._common_dataloader_kwargs)

    def val_dataloader(self) -> DataLoader[tuple[NDArray[np.float32], NDArray[np.float32]]]:
        """Construct validation dataloader"""
        dataset = ZebraDataset(self.input_paths, self.val_period[0], self.val_period[1], preshuffle=True)
        return DataLoader(dataset, shuffle=False, **self._common_dataloader_kwargs)

    def test_dataloader(self) -> DataLoader[tuple[NDArray[np.float32], NDArray[np.float32]]]:
        """Construct test dataloader"""
        dataset = ZebraDataset(self.input_paths, self.test_period[0], self.test_period[1])
        return DataLoader(dataset, shuffle=False, **self._common_dataloader_kwargs)
