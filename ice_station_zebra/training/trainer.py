import logging
from collections import defaultdict
from contextlib import suppress
from datetime import datetime
from pathlib import Path

import hydra
from lightning import Callback, Trainer
from omegaconf import DictConfig, OmegaConf
from wandb.sdk.lib.runid import generate_id

from ice_station_zebra.data.lightning import ZebraDataModule

logger = logging.getLogger(__name__)


class ZebraTrainer:
    """A wrapper for PyTorch training"""

    def __init__(self, config: DictConfig) -> None:
        """Initialize the Zebra trainer."""
        # Load the resolved config into Python format
        config = OmegaConf.to_container(config, resolve=True)

        # Load paths
        base_path = Path(config["base_path"])

        # Construct dataset groups
        dataset_groups = defaultdict(list)
        for dataset in config["datasets"].values():
            dataset_groups[dataset["group_as"]].append(
                (base_path / "data" / "anemoi" / f"{dataset['name']}.zarr").resolve()
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

        # Get run directory from wandb logger if available
        run_directory = None
        for lightning_logger in lightning_loggers:
            with suppress(AttributeError):
                run_directory = Path(lightning_logger.experiment._settings.sync_dir)
        if not run_directory:
            name_ = f"run-{datetime.now().strftime(r'%Y%m%d_%H%M%S')}-{generate_id()}"
            run_directory = base_path / "training" / "local" / name_

        # Add callbacks
        callbacks: list[Callback] = []
        for callback_cfg in config["train"].get("callbacks", {}).values():
            callbacks.append(hydra.utils.instantiate(callback_cfg))

        # Construct the trainer
        self.trainer: Trainer = hydra.utils.instantiate(
            dict(
                {
                    "callbacks": callbacks,
                    "logger": lightning_loggers,
                },
                **config["train"]["trainer"],
            )
        )
        self.trainer.checkpoint_callback.dirpath = run_directory / "checkpoints"

        # Save config to the output directory
        OmegaConf.save(config, run_directory / "model_config.yaml")

    def train(self) -> None:
        self.trainer.fit(
            model=self.model,
            datamodule=self.data_module,
        )
