import logging
from collections import defaultdict
from pathlib import Path

from lightning import Trainer
from omegaconf import DictConfig, OmegaConf

from ice_station_zebra.models import ZebraModelEncProcDec
from ice_station_zebra.types import ZebraDataSpace

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
        logger.info(f"Found {len(dataset_groups)} dataset_groups {dataset_groups}")

        # Create a data module combining all groups
        self.data_module = ZebraDataModule(
            dataset_groups, predict_target=config["train"]["predict_target"]
        )

    def train(self) -> None:
        model = ZebraModelEncProcDec(
            input_spaces=self.data_module.input_spaces,
            latent_space=ZebraDataSpace(shape=(32, 32), channels=20),
            output_space=self.data_module.output_space,
        )
        trainer = Trainer(fast_dev_run=100)
        trainer.fit(
            model=model,
            datamodule=self.data_module,
        )
