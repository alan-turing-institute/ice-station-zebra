import logging
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from lightning.fabric.utilities import suggested_max_num_workers
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from ice_station_zebra.callbacks import UnconditionalCheckpoint
from ice_station_zebra.data_loaders import ZebraDataModule
from ice_station_zebra.utils import generate_run_name, get_wandb_logger

if TYPE_CHECKING:
    from lightning import Callback, Trainer

    from ice_station_zebra.models import ZebraModel


logger = logging.getLogger(__name__)


class ZebraTrainer:
    """A wrapper for PyTorch training."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize the Zebra trainer."""
        # Load inputs into a data module
        self.data_module = ZebraDataModule(config)

        # Construct the model
        self.model: ZebraModel = hydra.utils.instantiate(
            dict(
                {
                    "input_spaces": [
                        s.to_dict() for s in self.data_module.input_spaces
                    ],
                    "n_forecast_steps": self.data_module.n_forecast_steps,
                    "n_history_steps": self.data_module.n_history_steps,
                    "output_space": self.data_module.output_space.to_dict(),
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
                        "project": self.model.name,
                    },
                    **logger_config,
                )
            )
            for logger_config in config.get("loggers", {}).values()
        ]

        # Get run directory from wandb logger or generate a new one
        if wandb_logger := get_wandb_logger(lightning_loggers):
            run_directory = Path(wandb_logger.experiment._settings.sync_dir)
        else:
            run_directory = (
                self.data_module.base_path / "training" / "local" / generate_run_name()
            )

        # Add callbacks
        callbacks: list[Callback] = [
            hydra.utils.instantiate(cfg)
            for cfg in config["train"].get("callbacks", {}).values()
        ]
        for callback in callbacks:
            logger.debug("Adding training callback %s.", callback.__class__.__name__)

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

        # Set properties for checkpoint callbacks
        for callback in self.trainer.callbacks:
            if isinstance(callback, (ModelCheckpoint, UnconditionalCheckpoint)):
                callback.dirpath = run_directory / "checkpoints"

        # Assign workers for data loading
        self.data_module.assign_workers(
            suggested_max_num_workers(self.trainer.num_devices)
        )

        # Save config to the output directory
        OmegaConf.save(config, run_directory / "model_config.yaml")

    def train(self) -> None:
        logger.info("Starting training for %d epochs.", self.trainer.max_epochs)
        self.trainer.fit(
            model=self.model,
            datamodule=self.data_module,
        )
