import logging
from pathlib import Path

import hydra
from lightning import Callback, Trainer
from omegaconf import OmegaConf

from ice_station_zebra.data.lightning import ZebraDataModule
from ice_station_zebra.models import ZebraModel

logger = logging.getLogger(__name__)


class ZebraEvaluator:
    """A wrapper for PyTorch evaluation"""

    def __init__(self, checkpoint_path: Path) -> None:
        """Initialize the Zebra evaluator."""
        # Verify the checkpoint path
        if checkpoint_path.exists():
            logger.debug(f"Loaded checkpoint from {checkpoint_path}.")
        else:
            msg = f"Checkpoint file {checkpoint_path} does not exist."
            raise FileNotFoundError(msg)

        # Load the model configuration
        config_path = checkpoint_path.parent.parent / "model_config.yaml"
        try:
            config = OmegaConf.load(config_path)
            logger.debug(f"Loaded model config from {config_path}.")
        except (NotADirectoryError, FileNotFoundError):
            msg = f"Could not find model configuration file at {config_path}."
            raise FileNotFoundError(msg)

        # Load the model from checkpoint
        model_cls: type[ZebraModel] = hydra.utils.get_class(config["model"]["_target_"])
        self.model = model_cls.load_from_checkpoint(checkpoint_path=checkpoint_path)

        # Load inputs into a data module
        self.data_module = ZebraDataModule(config)

        # Add callbacks
        callbacks: list[Callback] = []
        for callback_cfg in config["evaluate"].get("callbacks", {}).values():
            callbacks.append(hydra.utils.instantiate(callback_cfg))
            logger.debug(
                f"Adding evaluation callback {callbacks[-1].__class__.__name__}."
            )

        # Construct lightning loggers
        lightning_loggers = [
            hydra.utils.instantiate(
                dict(
                    {
                        "job_type": "evaluate",
                        "project": "leaderboard",
                    },
                    **logger_config,
                )
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
