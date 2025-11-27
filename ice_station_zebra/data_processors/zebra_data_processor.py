import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

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
        Fallback to DEFAULT_TOTAL_PARTS.
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

    def _part_tracker_path(self) -> Path:
        """Path for part_trackerdata file that tracks completed parts."""
        return (
            Path(self.path_dataset).with_suffix(".load_chunks.json")
            if not Path(self.path_dataset).is_dir()
            else Path(self.path_dataset) / ".load_chunks.json"
        )

    def _read_part_tracker(self) -> dict:
        """Read part_tracker data JSON (returns {'completed': {part_spec: timestamp}})."""
        part_tracker_file = self._part_tracker_path()
        if not part_tracker_file.exists():
            return {"completed": {}}
        try:
            with part_tracker_file.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            logger.warning(
                "Failed to read load chunks part_tracker data %s; starting fresh",
                part_tracker_file,
            )
            return {"completed": {}}

    def _write_part_tracker(self, data: dict) -> None:
        """Write part_tracker data JSON to disk.

        Ensure parent dir exists. We write to a temporary file in the same directory
        then use replace() to get an atomic move. This is a POSIX-only feature.
        This means that if the process is killed, the part_tracker data will not be lost.
        """
        part_tracker_file = self._part_tracker_path()
        # Ensure parent directory exists (NamedTemporaryFile requires it)
        try:
            part_tracker_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.debug(
                "Could not ensure parent dir for part_tracker data: %s",
                part_tracker_file.parent,
            )

        try:
            with NamedTemporaryFile(
                "w", delete=False, dir=str(part_tracker_file.parent), encoding="utf-8"
            ) as fh:
                json.dump(data, fh, indent=2, sort_keys=True)
                tmp_name = Path(fh.name)
            tmp_name.replace(part_tracker_file)  # atomic replace if on same filesystem
        except OSError:
            # fallback to naive write
            try:
                with part_tracker_file.open("w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2, sort_keys=True)
            except OSError:
                logger.exception(
                    "Failed to write part_tracker data file %s", part_tracker_file
                )

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
