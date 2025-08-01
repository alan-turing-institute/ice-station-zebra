import logging
from collections import defaultdict
from functools import cached_property
from pathlib import Path

import numpy as np
from lightning import LightningDataModule
from numpy.typing import NDArray
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ice_station_zebra.types import DataloaderArgs, DataSpace

from .combined_dataset import CombinedDataset
from .zebra_dataset import ZebraDataset

logger = logging.getLogger(__name__)


class ZebraDataModule(LightningDataModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        # Load paths
        self.base_path = Path(config["base_path"])

        # Construct dataset groups
        self.dataset_groups = defaultdict(list)
        for dataset in config["datasets"].values():
            self.dataset_groups[dataset["group_as"]].append(
                (
                    self.base_path / "data" / "anemoi" / f"{dataset['name']}.zarr"
                ).resolve()
            )
        logger.info(f"Found {len(self.dataset_groups)} dataset_groups")
        for dataset_group in self.dataset_groups.keys():
            logger.debug(f"... {dataset_group}")

        # Check prediction target
        self.predict_target = config["train"]["predict"]["dataset_group"]
        if self.predict_target not in self.dataset_groups:
            raise ValueError(f"Could not find prediction target {self.predict_target}")

        # Set periods for train, validation, and test
        self.batch_size = int(config["split"]["batch_size"])
        self.train_period = {
            k: None if v == "None" else v for k, v in config["split"]["train"].items()
        }
        self.val_period = {
            k: None if v == "None" else v
            for k, v in config["split"]["validate"].items()
        }
        self.test_period = {
            k: None if v == "None" else v for k, v in config["split"]["test"].items()
        }

        # Set history and forecast steps
        self.n_forecast_steps = config["train"]["predict"].get("n_forecast_steps", 1)
        self.n_history_steps = config["train"]["predict"].get("n_history_steps", 1)

        # Set common arguments for the dataloader
        self._common_dataloader_kwargs = DataloaderArgs(
            batch_size=self.batch_size,
            sampler=None,
            batch_sampler=None,
            drop_last=False,
            worker_init_fn=None,
        )

    @cached_property
    def input_spaces(self) -> list[DataSpace]:
        """Return the data space for each input"""
        return [
            ZebraDataset(name, paths).space
            for name, paths in self.dataset_groups.items()
            if name != self.predict_target
        ]

    @cached_property
    def output_space(self) -> DataSpace:
        """Return the data space of the desired output"""
        return next(
            ZebraDataset(name, paths).space
            for name, paths in self.dataset_groups.items()
            if name == self.predict_target
        )

    def train_dataloader(
        self,
    ) -> DataLoader[tuple[NDArray[np.float32], NDArray[np.float32]]]:
        """Construct train dataloader"""
        dataset = CombinedDataset(
            [
                ZebraDataset(
                    name,
                    paths,
                    start=self.train_period["start"],
                    end=self.train_period["end"],
                )
                for name, paths in self.dataset_groups.items()
            ],
            n_forecast_steps=self.n_forecast_steps,
            n_history_steps=self.n_history_steps,
            target=self.predict_target,
        )
        logger.info(
            "Loaded training dataset with %d samples between %s and %s",
            len(dataset),
            dataset.start_date,
            dataset.end_date,
        )
        return DataLoader(dataset, shuffle=True, **self._common_dataloader_kwargs)

    def val_dataloader(
        self,
    ) -> DataLoader[tuple[NDArray[np.float32], NDArray[np.float32]]]:
        """Construct validation dataloader"""
        dataset = CombinedDataset(
            [
                ZebraDataset(
                    name,
                    paths,
                    start=self.val_period["start"],
                    end=self.val_period["end"],
                )
                for name, paths in self.dataset_groups.items()
            ],
            n_forecast_steps=self.n_forecast_steps,
            n_history_steps=self.n_history_steps,
            target=self.predict_target,
        )
        logger.info(
            "Loaded validation dataset with %d samples between %s and %s",
            len(dataset),
            dataset.start_date,
            dataset.end_date,
        )
        return DataLoader(dataset, shuffle=False, **self._common_dataloader_kwargs)

    def test_dataloader(
        self,
    ) -> DataLoader[tuple[NDArray[np.float32], NDArray[np.float32]]]:
        """Construct test dataloader"""
        dataset = CombinedDataset(
            [
                ZebraDataset(
                    name,
                    paths,
                    start=self.test_period["start"],
                    end=self.test_period["end"],
                )
                for name, paths in self.dataset_groups.items()
            ],
            n_forecast_steps=self.n_forecast_steps,
            n_history_steps=self.n_history_steps,
            target=self.predict_target,
        )
        logger.info(
            "Loaded test dataset with %d samples between %s and %s",
            len(dataset),
            dataset.start_date,
            dataset.end_date,
        )
        return DataLoader(dataset, shuffle=False, **self._common_dataloader_kwargs)
