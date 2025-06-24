import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Type

from anemoi.datasets.commands.create import Create
from anemoi.datasets.commands.inspect import InspectZarr
from omegaconf import DictConfig, OmegaConf

from ice_station_zebra.datasets.preprocessors import IPreprocessor

log = logging.getLogger(__name__)


@dataclass
class AnemoiCreateArgs:
    path: str
    config: DictConfig
    command: str = "unused"
    threads: int = 0
    processes: int = 0


@dataclass
class AnemoiInspectArgs:
    path: str
    detailed: bool
    progress: bool
    statistics: bool
    size: bool


class AnemoiDataset:
    def __init__(
        self, name: str, config: DictConfig, cls_preprocessor: Type[IPreprocessor]
    ) -> None:
        self.name = name
        self.path = Path(config["data_path"]) / f"{self.name}.zarr"
        self.preprocessor = cls_preprocessor(
            config["datasets"][name], config["data_path"]
        )
        # Add preprocessor outputs to config then resolve references
        # Note that Anemoi 'forcings' need to be escaped with `\${}` to avoid being resolved here
        cfg_anemoi = OmegaConf.create(config["datasets"][name])
        if "preprocessor" in cfg_anemoi:
            cfg_anemoi["preprocessor"]["outputs"] = self.preprocessor.outputs()
        self.config = OmegaConf.to_container(cfg_anemoi, resolve=True)

    def download(self) -> None:
        """Download a single Anemoi dataset"""
        log.info(f"Creating dataset {self.name} at {self.path}")
        self.preprocessor.download()
        Create().run(
            AnemoiCreateArgs(
                path=str(self.path),
                config=self.config,
            )
        )

    def inspect(self) -> None:
        """Inspect a single Anemoi dataset"""
        log.info(f"Inspecting dataset {self.name} at {self.path}")
        InspectZarr().run(
            AnemoiInspectArgs(
                path=str(self.path),
                detailed=True,
                progress=True,
                statistics=True,
                size=True,
            )
        )
