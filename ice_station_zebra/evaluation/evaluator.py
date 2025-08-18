import logging
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from omegaconf import DictConfig, OmegaConf

from ice_station_zebra.data_loaders import ZebraDataModule
from ice_station_zebra.utils import get_timestamp

if TYPE_CHECKING:
    from lightning import Callback, Trainer

    from ice_station_zebra.models import ZebraModel

logger = logging.getLogger(__name__)


class ZebraEvaluator:
    """A wrapper for PyTorch evaluation"""

    def __init__(self, config: DictConfig, checkpoint_path: Path) -> None:
        """Initialize the Zebra evaluator."""
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
        self.model = model_cls.load_from_checkpoint(checkpoint_path=checkpoint_path)

        # Load inputs into a data module
        self.data_module = ZebraDataModule(config)

        # Add callbacks
        callbacks: list[Callback] = [
            hydra.utils.instantiate(cfg)
            for cfg in config["evaluate"].get("callbacks", {}).values()
        ]
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

    def evaluate(self) -> None:
        self.trainer.test(
            model=self.model,
            datamodule=self.data_module,
        )
