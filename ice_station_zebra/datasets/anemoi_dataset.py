import logging
from dataclasses import dataclass
from pathlib import Path

from anemoi.datasets.commands.create import Create
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@dataclass
class AnemoiCreateArgs:
    path: str
    config: DictConfig
    command: str = "unused"
    threads: int = 0
    processes: int = 0


class AnemoiDataset:
    def __init__(self, name: str, config: DictConfig) -> None:
        self.name = name
        # Convert the Anemoi config to a plain dict so references will be resolved at the file-level
        self.config = OmegaConf.to_container(config["datasets"][name], resolve=False)
        self.path = Path(config["data_path"]) / f"{self.name}.zarr"

    def download(self) -> None:
        """Download a single Anemoi dataset"""
        log.info(f"Creating dataset {self.name} at {self.path}")
        Create().run(
            AnemoiCreateArgs(
                path=str(self.path),
                config=self.config,
            )
        )


class AnemoiDatasetManager:
    def __init__(self, config: DictConfig) -> None:
        self.datasets = [
            AnemoiDataset(dataset_name, config) for dataset_name in config["datasets"]
        ]
