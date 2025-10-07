import logging
import os
from contextlib import suppress
from ftplib import FTP, error_perm
from pathlib import Path

import pandas as pd
from icenet.data.sic.mask import Masks
from icenet.data.sic.osisaf import SICDownloader
from omegaconf import DictConfig

from .ipreprocessor import IPreprocessor

logger = logging.getLogger(__name__)


class IceNetSICPreprocessor(IPreprocessor):
    def __init__(self, config: DictConfig) -> None:
        """Initialise the IceNetSIC preprocessor."""
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
        """Download data to the specified preprocessor path."""
        # Change to the output directory before downloading
        icenet_path = preprocessor_path / self.dataset_name / self.cls_name
        icenet_path.mkdir(parents=True, exist_ok=True)
        current_directory = Path.cwd()
        os.chdir(icenet_path)

        # Treat each year separately
        for year in sorted({date.year for date in self.date_range}):
            logger.info("Preparing to download data for %s.", year)
            # Generate polar masks: note that only 1979-2015 are available
            masks = Masks(north=self.is_north, south=self.is_south)
            mask_year = max(1979, min(year, 2015))
            logger.info("Generating polar masks for %s.", mask_year)
            self.pre_download_masks(mask_year, icenet_path)
            masks.generate(year=mask_year)

            logger.info("Downloading sea ice concentration data to %s.", icenet_path)
            sic = SICDownloader(
                dates=[
                    pd.to_datetime(date).date()
                    for date in self.date_range
                    if date.year == year
                ],  # Dates to download SIC data for
                delete_tempfiles=True,  # Delete temporary downloaded files after use
                north=self.is_north,  # Use mask for the Northern Hemisphere (set to True if needed)
                south=self.is_south,  # Use mask for the Southern Hemisphere
                parallel_opens=False,  # Concatenation is not working correctly with dask
            )
            sic.download()

        # Change back to the original directory
        os.chdir(current_directory)

    def pre_download_masks(self, mask_year: int, icenet_path: Path) -> None:
        """Pre-download the OSISAF masks for a given year.

        We need to do this ourselves, as IceNet tries to download the 2nd day of each
        month, which does not always exist.
        """
        mask_base_path = (
            icenet_path
            / "data"
            / "masks"
            / ("north" if self.is_north else "south")
            / "siconca"
        )
        for mask_month in range(1, 13):
            ftp = FTP("osisaf.met.no", "anonymous")  # noqa: S321
            ftp.cwd(f"reprocessed/ice/conc/v2p0/{mask_year:04d}/{mask_month:02d}")
            filename_stem = f"ice_conc_{'nh' if self.is_north else 'sh'}_ease2-250_cdr-v2p0_{mask_year:04d}{mask_month:02d}"
            local_filename = (
                mask_base_path
                / f"{mask_year:04d}"
                / f"{mask_month:02d}"
                / f"{filename_stem}021200.nc"
            )
            local_filename.parent.mkdir(parents=True, exist_ok=True)
            for mask_day in range(1, 29):
                remote_filename = f"{filename_stem}{mask_day:02d}1200.nc"
                with suppress(error_perm):
                    with local_filename.open("wb") as fp:
                        ftp.retrbinary(f"RETR {remote_filename}", fp.write)
                    if local_filename.stat().st_size:
                        break
