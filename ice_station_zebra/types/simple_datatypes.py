from dataclasses import dataclass
from typing import Literal, NamedTuple, TypedDict

from matplotlib.colors import Normalize
from omegaconf import DictConfig

from .typedefs import DiffMode, DiffStrategy


@dataclass
class AnemoiCreateArgs:
    """Arguments for anemoi create."""

    config: DictConfig
    path: str
    command: str = "unused"
    overwrite: bool = False
    processes: int = 0
    threads: int = 0


@dataclass
class AnemoiInspectArgs:
    """Arguments for anemoi inspect."""

    detailed: bool
    path: str
    progress: bool
    size: bool
    statistics: bool


class DataloaderArgs(TypedDict):
    """Arguments for the data loader."""

    batch_size: int
    sampler: None
    batch_sampler: None
    drop_last: bool
    worker_init_fn: None


class DiffColourmapSpec(NamedTuple):
    """Colour scale specification for visualising a difference panel."""

    norm: Normalize | None
    vmin: float | None
    vmax: float | None
    cmap: str


@dataclass(frozen=True)
class PlotSpec:
    """Specification for a plotting strategy used by visualisations."""

    # What and how
    variable: str
    title_groundtruth: str = "Ground Truth"
    title_prediction: str = "Prediction"
    title_difference: str = "Difference"

    # Rendering
    n_contour_levels: int = 51
    colourmap: str = "viridis"

    # Difference pane
    include_difference: bool = True
    diff_mode: DiffMode = "signed"
    diff_strategy: DiffStrategy = "precompute"
    selected_timestep: int = 0

    # Ranges: Optional to allow inference upstream; defaults pin to [0,1]
    vmin: float | None = 0.0
    vmax: float | None = 1.0

    # Colourbar layout
    colourbar_location: Literal["vertical", "horizontal"] = "horizontal"
    colourbar_strategy: Literal["shared", "separate"] = "shared"

    # Sanity/warnings in badge
    outside_warn: float = 0.05
    severe_outside: float = 0.20
    include_shared_range_mismatch_check: bool = True
