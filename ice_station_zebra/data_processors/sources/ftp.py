from ftplib import FTP
from tempfile import TemporaryDirectory
from typing import Any

import earthkit.data as ekd
from anemoi.datasets.create.sources import source_registry
from anemoi.datasets.create.sources.legacy import LegacySource
from anemoi.datasets.create.sources.xarray import load_one
from anemoi.datasets.dates.groups import GroupOfDates
from earthkit.data.core.fieldlist import MultiFieldList
from earthkit.data.utils.patterns import Pattern


@source_registry.register("ftp")
class FTPSource(LegacySource):
    @staticmethod
    def _execute(
        context: dict[str, Any],
        dates: GroupOfDates,
        url: str,
        *args: Any,
        passwd: str = "",
        user: str = "anonymous",
        **kwargs: Any,
    ) -> ekd.FieldList:
        """Execute the data loading process from an OpenDAP source.

        Parameters
        ----------
        context : dict
            The context in which the function is executed.
        dates : list
            List of dates for which data is to be loaded.
        passwd : str
            The FTP password (default "")
        url : str
            The URL of the FTP source.
        user : str
            The FTP user (default "anonymous")
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        xarray.Dataset
            The loaded dataset.

        """
        # Parse the FTP URL
        url = url.replace("ftp://", "")
        server = url.split("/")[0]
        directory_pattern = "/" + "/".join(url.split("/")[1:-1])
        filename_pattern = url.split("/")[-1]

        # Setup a temporary directory for downloads
        download_directory = TemporaryDirectory()
        downloaded_files = []

        # Connect to the FTP server
        with FTP(server) as session:
            session.login(user=user, passwd=passwd)

            # Iterate over dates, downloading one file per date
            for iso_date in [d.isoformat() for d in dates]:
                directory = Pattern(directory_pattern).substitute(date=iso_date)
                if not isinstance(directory, str):
                    raise ValueError(
                        "Expected a single directory string after substitution."
                    )
                filename = Pattern(filename_pattern).substitute(date=iso_date)
                if not isinstance(filename, str):
                    raise ValueError(
                        "Expected a single filename string after substitution."
                    )
                download_path = f"{download_directory.name}/{filename}"
                session.cwd(directory)
                with open(download_path, "wb") as input_file:
                    session.retrbinary(f"RETR {filename}", input_file.write)
                downloaded_files.append(
                    load_one("📂", context, [iso_date], download_path)
                )

        # Combine all downloaded files into a MultiFieldList
        return MultiFieldList(downloaded_files)
