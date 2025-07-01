import logging
import os
from pathlib import Path

import pandas as pd
from icenet.data.sic.mask import Masks
from icenet.data.sic.osisaf import SICDownloader
from omegaconf import DictConfig

from .base import IPreprocessor

log = logging.getLogger(__name__)


class IceNetSICPreprocessor(IPreprocessor):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.date_range = pd.date_range(
            config["dates"]["start"],
            config["dates"]["end"],
            freq=config["dates"]["frequency"],
        )
        self.is_north = config["preprocessor"]["hemisphere"].lower() == "north"

    @property
    def is_south(self) -> bool:
        return not self.is_north

    def download(self, preprocessor_path: Path) -> None:
        # Change to the output directory before downloading
        icenet_path = preprocessor_path / self.name
        icenet_path.mkdir(parents=True, exist_ok=True)
        current_directory = os.getcwd()
        os.chdir(icenet_path)

        # Generate polar masks: note that only 1979-2015 are available
        masks = Masks(north=self.is_north, south=self.is_south)
        mask_year = max(max(1979, min(date.year, 2015)) for date in self.date_range)
        log.info(f"Generating polar masks for {mask_year}...")
        masks.generate(year=mask_year)

        log.info("Downloading sea ice concentration data...")
        sic = SICDownloader(
            dates=[
                pd.to_datetime(date).date() for date in self.date_range
            ],  # Dates to download SIC data for
            delete_tempfiles=True,  # Delete temporary downloaded files after use
            north=self.is_north,  # Use mask for the Northern Hemisphere (set to True if needed)
            south=self.is_south,  # Use mask for the Southern Hemisphere
            parallel_opens=False,  # Concatenation is not working correctly with dask
        )
        sic.download()

        # Change back to the original directory
        os.chdir(current_directory)
