import logging
from collections import defaultdict
from pathlib import Path

from lightning import Trainer
from omegaconf import DictConfig, OmegaConf

from ice_station_zebra.models import ZebraModelEncProcDec

from .data_modules import ZebraDataModule

logger = logging.getLogger(__name__)


class ZebraTrainer:
    """An Anemoi trainer with no graph"""

    def __init__(self, config: DictConfig) -> None:
        """Initialize the Zebra trainer."""
        # Load the resolved config into Python format
        config = OmegaConf.to_container(config, resolve=True)

        # Load paths
        training_path = Path(config["base_path"]) / "training"
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
        model = ZebraModelEncProcDec(self.data_module.input_shapes)
        trainer = Trainer(fast_dev_run=100)
        trainer.fit(
            model=model,
            datamodule=self.data_module,
        )
