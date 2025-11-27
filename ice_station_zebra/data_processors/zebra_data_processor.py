import logging
import shutil
from datetime import datetime
from pathlib import Path

from anemoi.datasets.commands.create import Create
from anemoi.datasets.commands.finalise import Finalise
from anemoi.datasets.commands.init import Init
from anemoi.datasets.commands.inspect import InspectZarr
from anemoi.datasets.commands.load import Load
from omegaconf import DictConfig, OmegaConf
from zarr.errors import PathNotFoundError

from ice_station_zebra.types import (
    AnemoiCreateArgs,
    AnemoiFinaliseArgs,
    AnemoiInitArgs,
    AnemoiInspectArgs,
    AnemoiLoadArgs,
)

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

    def create(self, *, overwrite: bool) -> None:
        """Ensure that a single Anemoi dataset exists."""
        if overwrite:
            logger.info(
                "Overwrite set to true, redownloading %s to %s",
                self.name,
                self.path_dataset,
            )
            shutil.rmtree(self.path_dataset, ignore_errors=True)
            self.download()
        else:
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

    def init(self, *, overwrite: bool) -> None:
        """Initialise a single Anemoi dataset."""
        logger.info("Initialising dataset %s at %s.", self.name, self.path_dataset)

        if overwrite:
            logger.info(
                "Overwrite set to true, reinitialising %s to %s",
                self.name,
                self.path_dataset,
            )
            shutil.rmtree(self.path_dataset, ignore_errors=True)
            Init().run(
                AnemoiInitArgs(
                    path=str(self.path_dataset),
                    config=self.config,
                )
            )
        else:
            try:
                self.inspect()
                logger.info(
                    "Dataset %s already exists at %s, no need to download.",
                    self.name,
                    self.path_dataset,
                )
            except (AttributeError, FileNotFoundError, PathNotFoundError):
                logger.info(
                    "Dataset %s not found at %s, initialising.",
                    self.name,
                    self.path_dataset,
                )
                shutil.rmtree(self.path_dataset, ignore_errors=True)
                Init().run(
                    AnemoiInitArgs(
                        path=str(self.path_dataset),
                        config=self.config,
                    )
                )

    def _compute_total_parts(self) -> int:
        """Infer total number of parts from config using start/end and group_by.

        Heuristic: if group_by contains 'month' or 'monthly' use month-count inclusive.
        Fallback to DEFAULT_TOTAL_PARTS parts.
        """
        default_total_parts = 1
        start = (
            self.config.get("start")
            or self.config.get("start_date")
            or self.config.get("begin")
        )
        end = (
            self.config.get("end")
            or self.config.get("end_date")
            or self.config.get("stop")
        )
        group_by = (self.config.get("group_by") or "").lower()

        if not start or not end:
            logger.debug(
                "Could not detect start/end in config; defaulting total_parts=%d",
                default_total_parts,
            )
            return default_total_parts

        try:
            start_ts = datetime.fromisoformat(start)
            end_ts = datetime.fromisoformat(end)
        except (ValueError, TypeError) as err:
            # Try to be permissive: strip timezone 'Z' etc.
            try:
                if start.endswith("Z"):
                    start_ts = datetime.fromisoformat(start[:-1])
                if end.endswith("Z"):
                    end_ts = datetime.fromisoformat(end[:-1])
            except (ValueError, TypeError) as tz_err:
                logger.debug(
                    "Failed to parse start/end dates (%s, %s): %s; defaulting total_parts=%d",
                    start,
                    end,
                    tz_err,
                    default_total_parts,
                )
                return default_total_parts
            else:
                logger.debug(
                    "Parsed start/end after stripping timezone info despite initial error: %s",
                    err,
                )

        if "month" in group_by or group_by in ("monthly", "month"):
            total = (
                (end_ts.year - start_ts.year) * 12 + (end_ts.month - start_ts.month) + 1
            )
            return max(default_total_parts, total)

        # treat daily/24h as monthly parts by default (keeps reasonable part counts)
        if "day" in group_by or group_by in ("daily", "1d", "24h"):
            total = (
                (end_ts.year - start_ts.year) * 12 + (end_ts.month - start_ts.month) + 1
            )
            return max(default_total_parts, total)

        return default_total_parts

    def load(self, parts: str) -> None:
        """Download a segment of an Anemoi dataset."""
        logger.info(
            "Downloading %s part of %s to %s.",
            parts,
            self.name,
            self.path_dataset,
        )
        Load().run(
            AnemoiLoadArgs(
                path=str(self.path_dataset),
                config=self.config,
                parts=parts,
            )
        )

    def finalise(self) -> None:
        """Finalise the segmented Anemoi dataset."""
        Finalise().run(
            AnemoiFinaliseArgs(
                path=str(self.path_dataset),
                config=self.config,
            )
        )
