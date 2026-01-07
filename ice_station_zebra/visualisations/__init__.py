# Set non-interactive matplotlib backend early to prevent hangs on macOS
# make sure the import is not reordered by the formatter
import matplotlib as mpl

mpl.use("Agg")

from ice_station_zebra.types import PlotSpec

from .plotting_core import detect_land_mask_path
from .plotting_maps import (
    DEFAULT_SIC_SPEC,
    plot_maps,
    video_maps,
)
from .range_check import compute_range_check_report

__all__ = [
    "DEFAULT_SIC_SPEC",
    "PlotSpec",
    "compute_range_check_report",
    "detect_land_mask_path",
    "plot_maps",
    "video_maps",
]
