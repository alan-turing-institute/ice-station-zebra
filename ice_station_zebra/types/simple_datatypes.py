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

    batch_sampler: None
    batch_size: int
    drop_last: bool
    num_workers: int
    persistent_workers: bool
    sampler: None
    worker_init_fn: None


class DiffColourmapSpec(NamedTuple):
    """Specify the colour scale used for a difference panel.

    Attributes:
        norm: Normalisation for mapping values to colours (e.g. TwoSlopeNorm for signed diffs).
        vmin: Lower bound if no norm is provided.
        vmax: Upper bound if no norm is provided.
        cmap: Matplotlib colormap name.

    """

    norm: Normalize | None
    vmin: float | None
    vmax: float | None
    cmap: str


@dataclass(frozen=True)
class PlotSpec:
    """Configure how sea-ice plots are rendered.

    Attributes:
        variable: Variable name shown in plots / used for routing.
        title_groundtruth: Title above the ground-truth panel.
        title_prediction: Title above the prediction panel.
        title_difference: Title above the difference panel.
        n_contour_levels: Number of contour levels per panel.
        colourmap: Colormap used for GT/prediction panels.
        include_difference: Whether to draw a difference panel.
        diff_mode: Difference definition (e.g. "signed", "absolute", "smape").
        diff_strategy: Strategy for animations (precompute, two-pass, per-frame).
        selected_timestep: Slice index when a single timestep is needed.
        vmin: Lower bound for GT/prediction colour scale (None = infer).
        vmax: Upper bound for GT/prediction colour scale (None = infer).
        colourbar_location: "vertical" or "horizontal".
        colourbar_strategy: "shared" or "separate" colourbars.
        outside_warn: Threshold for “values outside display range” warnings.
        severe_outside: Severe threshold for clipping warnings.
        include_shared_range_mismatch_check: If True, add magnitude mismatch nudges.

    """

    variable: str
    title_groundtruth: str = "Ground Truth"
    title_prediction: str = "Prediction"
    title_difference: str = "Difference"

    n_contour_levels: int = 51
    colourmap: str = "viridis"

    # Difference pane
    include_difference: bool = True
    diff_mode: DiffMode = "signed"
    diff_strategy: DiffStrategy = "precompute"
    selected_timestep: int = 0

    # Colourscale ranges: defaults to [0,1]
    vmin: float | None = 0.0
    vmax: float | None = 1.0

    # Colourbar layout
    colourbar_location: Literal["vertical", "horizontal"] = "horizontal"
    colourbar_strategy: Literal["shared", "separate"] = "shared"

    # Range Check/warnings in badge
    outside_warn: float = 0.05
    severe_outside: float = 0.20
    include_shared_range_mismatch_check: bool = True

    # Land mask overlay
    land_mask_path: str | None = None

    # Optional metadata for titling
    # hemisphere: "north" | "south" when known (used in titles)
    hemisphere: Literal["north", "south"] | None = None
    # metadata_subtitle: free-form text (e.g., "epochs=50; train=2010-2018")
    metadata_subtitle: str | None = None

    # Footer control
    include_footer_metadata: bool = True
