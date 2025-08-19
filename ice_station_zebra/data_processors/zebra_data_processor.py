import logging
import shutil
from pathlib import Path

from anemoi.datasets.commands.create import Create
from anemoi.datasets.commands.inspect import InspectZarr
from omegaconf import DictConfig, OmegaConf
from zarr.errors import PathNotFoundError

from ice_station_zebra.types import AnemoiCreateArgs, AnemoiInspectArgs

from .preprocessors import IPreprocessor

logger = logging.getLogger(__name__)


class ZebraDataProcessor:
    def __init__(
        self, name: str, config: DictConfig, cls_preprocessor: type[IPreprocessor]
    ) -> None:
        """Initialise a ZebraDataProcessor from a config.

        Register a preprocessor if appropriate.
        """
        self.name = name
        _data_path = Path(config["base_path"]).resolve() / "data"
        self.path_dataset = _data_path / "anemoi" / f"{name}.zarr"
        self.path_preprocessor = _data_path / "preprocessing"
        # Note that Anemoi 'forcings' need to be escaped with `\${}` to avoid being resolved here
        self.config: DictConfig = OmegaConf.to_object(config["datasets"][name])  # type: ignore[assignment]
        self.preprocessor = cls_preprocessor(self.config)

    def create(self) -> None:
        """Ensure that a single Anemoi dataset exists."""
        try:
            self.inspect()
            logger.info(
                "Dataset %s already exists at %s, no need to download.",
                self.name,
                self.path_dataset,
            )
        except (AttributeError, FileNotFoundError, PathNotFoundError):
            logger.info("Dataset %s not found at %s.", self.name, self.path_dataset)
            shutil.rmtree(self.path_dataset, ignore_errors=True)
            self.download()

    def download(self) -> None:
        """Download a single Anemoi dataset."""
        self.preprocessor.download(self.path_preprocessor)
        logger.info("Creating dataset %s at %s.", self.name, self.path_dataset)
        Create().run(
            AnemoiCreateArgs(
                path=str(self.path_dataset),
                config=self.config,
            )
        )

    def inspect(self) -> None:
        """Inspect a single Anemoi dataset."""
        logger.info("Inspecting dataset %s at %s.", self.name, self.path_dataset)
        InspectZarr().run(
            AnemoiInspectArgs(
                path=str(self.path_dataset),
                detailed=True,
                progress=False,  # must be disabled until https://github.com/ecmwf/anemoi-datasets/pull/372 is merged
                statistics=False,
                size=True,
            )
        )
