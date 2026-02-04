import logging
from pathlib import Path, PosixPath
from typing import TYPE_CHECKING

# Set matplotlib backend BEFORE any imports that might use it
import matplotlib as mpl

mpl.use("Agg")

import hydra
import torch
from lightning.fabric.utilities import suggested_max_num_workers
from omegaconf import DictConfig, OmegaConf

from ice_station_zebra.data_loaders import ZebraDataModule
from ice_station_zebra.utils import get_device_name, get_device_threads, get_timestamp

if TYPE_CHECKING:
    from lightning import Callback, Trainer

    from ice_station_zebra.models import ZebraModel

logger = logging.getLogger(__name__)


class ZebraEvaluator:
    """A wrapper for PyTorch evaluation."""

    def __init__(self, config: DictConfig, checkpoint_path: Path) -> None:
        """Initialize the Zebra evaluator from a config and checkpoint."""
        # Verify the checkpoint path
        if checkpoint_path.exists():
            logger.debug("Loaded checkpoint from %s.", checkpoint_path)
        else:
            msg = f"Checkpoint file {checkpoint_path} does not exist."
            raise FileNotFoundError(msg)

        # Load the model configuration
        config_path = checkpoint_path.parent.parent / "model_config.yaml"
        try:
            ckpt_config = OmegaConf.load(config_path)
            logger.debug("Loaded checkpoint config from %s.", ckpt_config)
            config["model"]["_target_"] = ckpt_config["model"]["_target_"]  # type: ignore[index]
        except (NotADirectoryError, FileNotFoundError):
            logger.debug("Could not find model configuration file at %s.", config_path)

        # Load the model from checkpoint
        model_cls: type[ZebraModel] = hydra.utils.get_class(config["model"]["_target_"])
        with torch.serialization.safe_globals([PosixPath]):
            self.model = model_cls.load_from_checkpoint(checkpoint_path=checkpoint_path)

        # Load inputs into a data module
        self.data_module = ZebraDataModule(config)

        # Add callbacks
        callbacks: list[Callback] = []
        callbacks_dict = config["evaluate"].get("callbacks") or {}
        for cfg in callbacks_dict.values():
            # Skip configs without _target_ (they can't be instantiated)
            # Convert DictConfig to a plain dict for easier checking
            cfg_dict = (
                OmegaConf.to_container(cfg, resolve=True)
                if isinstance(cfg, DictConfig)
                else cfg
            )
            if not isinstance(cfg_dict, dict) or "_target_" not in cfg_dict:
                logger.warning(
                    "Skipping callback config without _target_: %s",
                    cfg_dict,
                )
                continue
            callback = hydra.utils.instantiate(cfg_dict)
            # Pass config to PlottingCallback for land mask detection
            if callback.__class__.__name__ == "PlottingCallback":
                callback.config = OmegaConf.to_container(config, resolve=True)
            callbacks.append(callback)
        for callback in callbacks:
            logger.debug("Adding evaluation callback %s.", callback.__class__.__name__)

        # Construct lightning loggers
        lightning_loggers = [
            hydra.utils.instantiate(
                dict(**logger_config)
                | {
                    "job_type": "evaluate",
                    "name": f"{self.model.name}-{get_timestamp()}",
                    "project": "leaderboard",
                },
            )
            for logger_config in config.get("loggers", {}).values()
        ]

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

        # Assign workers for data loading
        self.data_module.assign_workers(
            suggested_max_num_workers(self.trainer.num_devices)
        )

    def evaluate(self) -> None:
        logger.info(
            "Starting evaluation using %d threads across %d %s device(s).",
            get_device_threads(),
            self.trainer.num_devices,
            get_device_name(self.trainer.accelerator.name()),
        )

        self.trainer.test(
            model=self.model,
            datamodule=self.data_module,
        )
