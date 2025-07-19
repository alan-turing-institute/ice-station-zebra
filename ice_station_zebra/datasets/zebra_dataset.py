import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Type

from anemoi.datasets.commands.create import Create
from anemoi.datasets.commands.inspect import InspectZarr
from omegaconf import DictConfig, OmegaConf
from zarr.errors import PathNotFoundError

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


class ZebraDataset:
    def __init__(
        self, name: str, config: DictConfig, cls_preprocessor: Type[IPreprocessor]
    ) -> None:
        self.name = name
        _data_path = Path(config["base_path"]).resolve() / "data"
        self.path_dataset = _data_path / "anemoi" / f"{name}.zarr"
        self.path_preprocessor = _data_path / "preprocessing"
        # Note that Anemoi 'forcings' need to be escaped with `\${}` to avoid being resolved here
        self.config = OmegaConf.to_container(config, resolve=True)["datasets"][name]
        self.preprocessor = cls_preprocessor(self.config)

    def create(self) -> None:
        """Ensure that a single Anemoi dataset exists"""
        try:
            self.inspect()
            log.info(
                f"Dataset {self.name} already exists at {self.path_dataset}, no need to download"
            )
        except (AttributeError, FileNotFoundError, PathNotFoundError):
            log.info(f"Dataset {self.name} not found at {self.path_dataset}")
            shutil.rmtree(self.path_dataset, ignore_errors=True)
            self.download()

    def download(self) -> None:
        """Download a single Anemoi dataset"""
        self.preprocessor.download(self.path_preprocessor)
        log.info(f"Creating dataset {self.name} at {self.path_dataset}")
        Create().run(
            AnemoiCreateArgs(
                path=str(self.path_dataset),
                config=self.config,
            )
        )

    def inspect(self) -> None:
        """Inspect a single Anemoi dataset"""
        log.info(f"Inspecting dataset {self.name} at {self.path_dataset}")
        InspectZarr().run(
            AnemoiInspectArgs(
                path=str(self.path_dataset),
                detailed=True,
                progress=False,  # must be disabled until https://github.com/ecmwf/anemoi-datasets/pull/372 is merged
                statistics=False,
                size=True,
            )
        )
