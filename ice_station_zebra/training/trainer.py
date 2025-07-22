import logging
from collections import defaultdict
from pathlib import Path

import hydra
from lightning import Trainer
from omegaconf import DictConfig, OmegaConf

from .data_modules import ZebraDataModule

logger = logging.getLogger(__name__)


class ZebraTrainer:
    """A wrapper for PyTorch training"""

    def __init__(self, config: DictConfig) -> None:
        """Initialize the Zebra trainer."""
        # Load the resolved config into Python format
        config = OmegaConf.to_container(config, resolve=True)

        # Load paths
        anemoi_data_path = Path(config["base_path"]) / "data" / "anemoi"

        # Construct dataset groups
        dataset_groups = defaultdict(list)
        for dataset in config["datasets"].values():
            dataset_groups[dataset["group_as"]].append(
                (anemoi_data_path / f"{dataset['name']}.zarr").resolve()
            )
        logger.info(f"Found {len(dataset_groups)} dataset_groups")
        for dataset_group in dataset_groups.keys():
            logger.debug(f"... {dataset_group}")

        # Create a data module combining all groups
        self.data_module = ZebraDataModule(
            dataset_groups,
            predict_target=config["train"]["predict_target"],
            split=config["split"],
        )

        # Construct the model
        self.model = hydra.utils.instantiate(
            dict(
                {
                    "input_spaces": self.data_module.input_spaces,
                    "output_space": self.data_module.output_space,
                    "optimizer": config["train"]["optimizer"],
                },
                **config["model"],
            ),
            _recursive_=False,
            _convert_="object",
        )

        # Construct lightning loggers
        lightning_loggers = [
            hydra.utils.instantiate(
                dict(
                    {
                        "job_type": "train",
                    },
                    **logger_config,
                )
            )
            for logger_config in config.get("loggers", {}).values()
        ]

        # Construct the trainer
        self.trainer: Trainer = hydra.utils.instantiate(
            config["train"]["trainer"], logger=lightning_loggers
        )

    def train(self) -> None:
        self.trainer.fit(
            model=self.model,
            datamodule=self.data_module,
        )
