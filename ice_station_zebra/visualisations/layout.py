"""Plot Map Layout Module.

Handles the layout of the plot panels for maps and colourbars.
The module is primarily used through the plotting system to create comparison visualisations
with 2-3 panels (ground truth, prediction, optional difference).

Main Components:
    - Layout constants for consistent spacing and proportions
    - GridSpec-based layout system with configurable margins and gutters
    - Colourbar positioning (vertical/horizontal)
    - Figure sizing based on data dimensions and aspect ratio for the plot panels
    - Specialised tick formatting for colourbars

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure

    from .plotting_core import DiffColourmapSpec, PlotSpec

# Layout constants
PREDICTION_PANEL_INDEX = 1
DIFFERENCE_PANEL_INDEX = 2
MIN_PANEL_COUNT_FOR_PREDICTION = 2
MIN_PANEL_COUNT_FOR_DIFFERENCE = 3
MIN_TICKS_FOR_MIDDLE_ZERO = 3

#
# --- LAYOUT CONFIGURATION CONSTANTS ---
# These constants define the default proportional spacing for the GridSpec layout.
# All values are expressed as fractions of the total figure dimensions to ensure
# consistent scaling across different figure sizes.

# Spacing and margin constants (as fractions of figure dimensions)
DEFAULT_OUTER_MARGIN = 0.05  # Outer margin around entire figure (prevents clipping)
DEFAULT_GUTTER = 0.05  # Horizontal gap between panels (fraction of panel width)
DEFAULT_GUTTER_HORIZONTAL = 0.03  # Smaller gaps when colourbar is below
DEFAULT_GUTTER_VERTICAL = 0.03  # Default for side-by-side with vertical bars
DEFAULT_CBAR_WIDTH = (
    0.06  # Width allocated for colourbar slots (fraction of panel width)
)
DEFAULT_TITLE_SPACE = (
    0.08  # Vertical space reserved for figure title (prevents overlap)
)

# Horizontal colourbar sizing (fractions of figure height)
DEFAULT_CBAR_HEIGHT = (
    0.07  # Thickness of horizontal colourbar row (relative to plot height)
)
DEFAULT_CBAR_PAD = 0.03  # Vertical gap between plots and the colourbar row

# Horizontal colourbar width within its slot (inset fraction within parent slot)
HCBAR_WIDTH_FRAC = 0.75  # centred 75% width
HCBAR_LEFT_FRAC = (1.0 - HCBAR_WIDTH_FRAC) / 2.0

# Default figure sizes for different panel configurations
DEFAULT_FIGSIZE_TWO_PANELS = (12, 6)  # Ground truth + Prediction
DEFAULT_FIGSIZE_THREE_PANELS = (18, 6)  # Ground truth + Prediction + Difference


# --- Main Layout Functions ---


def _init_axes(  # noqa: PLR0913, C901, PLR0912, PLR0915
    *,
    plot_spec: PlotSpec,
    height: int | None = None,
    width: int | None = None,
    outer_margin: float = DEFAULT_OUTER_MARGIN,
    gutter: float | None = None,
    cbar_width: float = DEFAULT_CBAR_WIDTH,
    title_space: float = DEFAULT_TITLE_SPACE,
    cbar_height: float = DEFAULT_CBAR_HEIGHT,
    cbar_pad: float = DEFAULT_CBAR_PAD,
) -> tuple[Figure, list[Axes], dict[str, Axes | None]]:
    """Create a GridSpec layout for multi-panel plots.

    This function can accommodate 2-3 panels with associated colourbars (vertical or horizontal),
    proper spacing, and aspect-ratio-aware sizing.

    Layout Structure:
        2-panel: [Ground Truth] [Prediction] [Colourbar]
        3-panel: [Ground Truth] [Prediction] [Colourbar] [Difference] [Colourbar]

    All spacing parameters are expressed as fractions of figure dimensions to ensure
    consistent scaling across different figure sizes and data aspect ratios.

    Args:
        plot_spec: PlotSpec object containing whether to include a difference panel,
                   colourbar orientation ('vertical'/'horizontal'),
                   colourbar strategy ('shared'/'separate'), and other formatting preferences.
        height: Optional data height for aspect-ratio-aware figure sizing. If provided with width,
           the figure dimensions will be calculated to maintain proper data aspect ratios.
        width: Optional data width for aspect-ratio-aware figure sizing.
        outer_margin: Fraction of figure dimensions reserved for outer margins (prevents
                     plot elements from being clipped at figure edges).
        gutter: Fraction of panel width used as horizontal spacing between panel groups.
               Only applied between prediction+colourbar and difference panel.
        cbar_width: Fraction of panel width allocated for each colourbar slot.
        title_space: Fraction of figure height reserved at the top for figure title
                    (prevents title from overlapping with plot content).
        cbar_height: Height fraction for the horizontal colourbar row (row 2 when
            orientation is 'horizontal'). Controls the bar thickness.
        cbar_pad: Vertical gap fraction between the plot row and the colourbar row in
            horizontal layouts. Set to 0.0 for flush rows.

    Returns:
        tuple containing:
            - Figure: The matplotlib Figure object with GridSpec layout applied
            - list[Axes]: List of main plot axes [ground_truth, prediction, difference*]
                         (*difference only present if plot_spec.include_difference=True)
            - dict[str, Axes | None]: Dictionary with dedicated colourbar axes:
                {"prediction": Axes | None, "difference": Axes | None}

    Raises:
        InvalidArrayError: If the arrays are not 2D or have different shapes.

    """
    n_panels = 3 if plot_spec.include_difference else 2
    orientation = plot_spec.colourbar_location

    # Choose gutter default per orientation if not provided
    if gutter is None:
        gutter = (
            DEFAULT_GUTTER_HORIZONTAL
            if orientation == "horizontal"
            else DEFAULT_GUTTER_VERTICAL
        )

    # Calculate top boundary: ensure title space doesn't consume too much of figure
    # Minimum 60% of figure height must remain for plots
    top_val = max(0.6, 1.0 - (outer_margin + title_space))

    # Calculate figure size based on data aspect ratio or use defaults
    if height and width and height > 0:
        # Calculate panel width maintaining data aspect ratio
        base_h = 6.0  # Standard height in inches
        aspect = width / height

        if orientation == "vertical":
            panel_w = base_h * aspect
            # Account for colourbars: panels + gutters + colourbar slots
            fig_w = (
                n_panels * panel_w  # Main panels
                + (n_panels - 1) * gutter * panel_w  # Gutters between groups
                + (n_panels - 1) * cbar_width * panel_w  # Colourbar slots
            )
        else:
            # --- Horizontal colourbars ---
            # Fraction of usable (top..bottom) height that belongs to the plot row
            usable_h_frac = top_val - outer_margin
            plot_row_frac = 1.0 / (1.0 + cbar_pad + cbar_height)

            # True height of the plot row in inches
            effective_plot_h = base_h * usable_h_frac * plot_row_frac

            # Make each panel width match the square content in that row
            panel_w = effective_plot_h * aspect

            fig_w = n_panels * panel_w + (n_panels - 1) * gutter * panel_w
        fig_size = (fig_w, base_h)
    else:
        # Use predefined sizes when data dimensions are unknown
        fig_size = (
            DEFAULT_FIGSIZE_THREE_PANELS
            if plot_spec.include_difference
            else DEFAULT_FIGSIZE_TWO_PANELS
        )

    fig = plt.figure(figsize=fig_size, constrained_layout=False)

    if orientation == "vertical":
        # VERTICAL COLOURBAR LAYOUT
        # Panel arrangement: [GT][Pred][cbar][gutter][Diff][cbar]
        # This creates columns for each element with appropriate relative widths

        col_specs: list[float] = []
        separate_colourbars = plot_spec.colourbar_strategy == "separate"

        if separate_colourbars:
            # Each panel gets its own colourbar: [GT][cbar][gutter][Pred][cbar][gutter][Diff][cbar]
            for ii in range(n_panels):
                col_specs.append(1.0)  # Main panel
                col_specs.append(cbar_width)  # Its colourbar

                # Add gutter after each panel+colourbar group (except the last)
                if ii < n_panels - 1:
                    col_specs.append(gutter)
        else:
            # Shared colourbar logic
            for ii in range(n_panels):
                col_specs.append(1.0)  # Main panel: full relative width

                # Add colourbar columns after prediction and difference panels
                if ii == PREDICTION_PANEL_INDEX:  # After prediction panel
                    col_specs.append(cbar_width)
                if ii == DIFFERENCE_PANEL_INDEX:  # After difference panel (if present)
                    col_specs.append(cbar_width)

                # Add gutter spacing, but only between panel groups (not after ground truth)
                if (
                    ii != n_panels - 1 and ii > 0
                ):  # Not the last panel and not the first panel (ground truth)
                    col_specs.append(gutter)

        gs = GridSpec(
            nrows=1,
            ncols=len(col_specs),
            figure=fig,
            width_ratios=col_specs,
            left=outer_margin,
            right=1 - outer_margin,
            top=top_val,
            bottom=outer_margin,
            wspace=0.0,
        )

        # Create axes and colourbar axes
        axs: list[Axes] = []
        separate_colourbars = plot_spec.colourbar_strategy == "separate"

        if separate_colourbars:
            caxes: dict[str, Axes | None] = {
                "groundtruth": None,
                "prediction": None,
                "difference": None,
            }
        else:
            caxes = {"prediction": None, "difference": None}
        col_idx = 0

        for ii in range(n_panels):
            # Create main plot axis (ground truth, prediction, or difference)
            ax = fig.add_subplot(gs[0, col_idx])
            axs.append(ax)
            col_idx += 1

            # Create dedicated colourbar axes adjacent to certain panels
            if separate_colourbars:
                # Each panel gets its own colourbar
                panel_names = ["groundtruth", "prediction", "difference"]
                if ii < len(panel_names):
                    caxes[panel_names[ii]] = fig.add_subplot(gs[0, col_idx])
                col_idx += 1
                # Skip gutter after each panel+colourbar group (except the last)
                if ii < n_panels - 1:
                    col_idx += 1
            else:
                # Shared colourbar logic
                if (
                    ii == PREDICTION_PANEL_INDEX
                ):  # After prediction panel: shared GT/prediction colourbar
                    caxes["prediction"] = fig.add_subplot(gs[0, col_idx])
                    col_idx += 1
                if (
                    ii == DIFFERENCE_PANEL_INDEX
                ):  # After difference panel: difference colourbar
                    caxes["difference"] = fig.add_subplot(gs[0, col_idx])
                    col_idx += 1
                # Skip over gutter columns (spacing between panel groups)
                if (
                    ii == PREDICTION_PANEL_INDEX and ii != n_panels - 1
                ):  # After pred+cbar group, before diff
                    col_idx += 1

    else:
        # HORIZONTAL COLOURBAR LAYOUT
        # Grid structure: 3 rows ([plots][pad][colourbars])
        # Column pattern: [panel][gutter][panel][gutter][panel]

        width_ratios = [1.0, gutter] * n_panels  # Alternate panels and gutters
        width_ratios = width_ratios[:-1]  # Remove trailing gutter

        gs = GridSpec(
            nrows=3,  # 0: plots, 1: pad, 2: colourbars
            ncols=len(width_ratios),
            figure=fig,
            width_ratios=width_ratios,
            height_ratios=[1.0, cbar_pad, cbar_height],
            left=outer_margin,
            right=1 - outer_margin,
            top=top_val,
            bottom=outer_margin,
            wspace=0.0,  # Width spacing handled by explicit gutter columns
            hspace=0.0,  # Rows are explicitly allocated via height ratios
        )

        # Create main plot axes in top row
        axs = []
        separate_colourbars = plot_spec.colourbar_strategy == "separate"

        if separate_colourbars:
            caxes = {"groundtruth": None, "prediction": None, "difference": None}
        else:
            caxes = {"prediction": None, "difference": None}

        for i in range(n_panels):
            panel_col = 2 * i  # Skip gutter columns: 0, 2, 4, ...
            axs.append(fig.add_subplot(gs[0, panel_col]))

        # Anchor plots to the bottom of their GridSpec cells to remove intra-cell vertical whitespace
        for ax in axs:
            ax.set_anchor("S")

        # Create colourbar axes in bottom row (row index 2)
        if separate_colourbars:
            # Each panel gets its own colourbar directly beneath it (75% width, centred)
            if n_panels >= 1:
                _p_gt = fig.add_subplot(gs[2, 0])
                _p_gt.set_axis_off()
                caxes["groundtruth"] = _p_gt.inset_axes(
                    [HCBAR_LEFT_FRAC, 0.0, HCBAR_WIDTH_FRAC, 1.0]
                )
            else:
                caxes["groundtruth"] = None

            if n_panels >= MIN_PANEL_COUNT_FOR_PREDICTION:
                _p_pr = fig.add_subplot(gs[2, 2])
                _p_pr.set_axis_off()
                caxes["prediction"] = _p_pr.inset_axes(
                    [HCBAR_LEFT_FRAC, 0.0, HCBAR_WIDTH_FRAC, 1.0]
                )
            else:
                caxes["prediction"] = None

            if n_panels >= MIN_PANEL_COUNT_FOR_DIFFERENCE:
                _p_df = fig.add_subplot(gs[2, 4])
                _p_df.set_axis_off()
                caxes["difference"] = _p_df.inset_axes(
                    [HCBAR_LEFT_FRAC, 0.0, HCBAR_WIDTH_FRAC, 1.0]
                )
            else:
                caxes["difference"] = None
        else:
            # Shared colourbar logic
            # Under GT+gutter+Pred span columns 0..3 when two panels present
            if n_panels >= MIN_PANEL_COUNT_FOR_PREDICTION:
                # Create a full-width parent and then a centered inset at configured width
                _parent_cbar_ax = fig.add_subplot(gs[2, 0:3])
                _parent_cbar_ax.set_axis_off()
                inset_cbar_ax = _parent_cbar_ax.inset_axes(
                    [HCBAR_LEFT_FRAC, 0.0, HCBAR_WIDTH_FRAC, 1.0]
                )
                caxes["prediction"] = inset_cbar_ax
            else:
                # Fallback: only GT present; use full column
                caxes["prediction"] = fig.add_subplot(gs[2, 0:1])

            if n_panels >= MIN_PANEL_COUNT_FOR_DIFFERENCE:
                _parent_diff_ax = fig.add_subplot(gs[2, 4])
                _parent_diff_ax.set_axis_off()
                caxes["difference"] = _parent_diff_ax.inset_axes(
                    [HCBAR_LEFT_FRAC, 0.0, HCBAR_WIDTH_FRAC, 1.0]
                )
            else:
                caxes["difference"] = None

    _set_titles(axs, plot_spec)
    _style_axes(axs)

    # Store layout parameters on the figure for potential future use
    fig._zebra_layout = {  # type: ignore[attr-defined]
        "outer_margin": outer_margin,
        "gutter": gutter,
        "cbar_width": cbar_width,
        "title_space": title_space,
        "cbar_height": cbar_height,
        "cbar_pad": cbar_pad,
    }
    return fig, axs, caxes


# --- Helper Functions ---


def _set_titles(axs: list[Axes], plot_spec: PlotSpec) -> None:
    """Set titles for each plot panel based on the PlotSpec configuration.

    The titles are applied in order: ground truth, prediction, and optionally difference.
    For difference panels, the difference mode (e.g., "signed", "absolute") is appended
    to provide clear indication of the type of comparison being shown.

    Args:
        axs: List of matplotlib Axes objects to title (in order: GT, pred, diff)
        plot_spec: PlotSpec containing title strings, whether to include a difference panel,
                   and difference mode configuration.

    """
    # If difference panel is included, append the difference mode to the title
    title_difference = (
        plot_spec.title_difference + f" ({plot_spec.diff_mode})"
        if plot_spec.include_difference
        else None
    )

    titles = [plot_spec.title_groundtruth, plot_spec.title_prediction, title_difference]
    for ax, title in zip(axs, titles, strict=False):
        if title is not None:
            ax.set_title(title)


def _style_axes(axs: Sequence[Axes]) -> None:
    """Apply consistent styling to all plot axes.

    This function sets up the visual appearance:
    - Removes axis ticks and labels (common for image/map data)
    - Sets equal aspect ratio to prevent distortion of spatial data

    Note: This function only applies styling and does not affect layout.

    Args:
        axs: Sequence of matplotlib Axes objects to style

    """
    for ax in axs:
        ax.axis("off")  # Remove axis ticks, labels, and spines
        ax.set_aspect("equal")  # Maintain square pixels for spatial data


def _set_axes_limits(axs: list[Axes], *, width: int, height: int) -> None:
    """Set consistent axis limits across all plot panels.

    Ensures all axes display the same spatial extent.

    Args:
        axs: List of matplotlib Axes objects to configure
        width: Width of the data array (sets x-axis limits: 0 to width)
        height: Height of the data array (sets y-axis limits: 0 to height)

    """
    for ax in axs:
        ax.set_xlim(0, width)  # X-axis: left edge to right edge
        ax.set_ylim(0, height)  # Y-axis: bottom edge to top edge


# --- Colourbar Functions ---
def _get_cbar_limits_from_mappable(cbar: Colorbar) -> tuple[float, float]:
    """Return (vmin, vmax) for a colourbar's mappable with robust fallbacks."""
    vmin = vmax = None
    try:  # Works for many matplotlib mappables
        vmin, vmax = cbar.mappable.get_clim()  # type: ignore[attr-defined]
    except AttributeError:
        norm = getattr(cbar.mappable, "norm", None)
        vmin = getattr(norm, "vmin", None)
        vmax = getattr(norm, "vmax", None)
    if vmin is None or vmax is None:
        vmin, vmax = 0.0, 1.0
    return float(vmin), float(vmax)


def _add_colourbars(  # noqa: PLR0913
    axs: list,
    *,
    image_groundtruth: Any,  # noqa: ANN401  # QuadContourSet
    image_prediction: Any | None = None,  # noqa: ANN401  # QuadContourSet | None
    image_difference: Any | None = None,  # noqa: ANN401  # QuadContourSet | None
    plot_spec: PlotSpec,
    diff_colour_scale: DiffColourmapSpec | None = None,
    cbar_axes: dict[str, Axes | None] | None = None,
) -> None:
    """Create and position colourbars.

    This function creates two types of colourbars:
    1. Shared colourbar for ground truth and prediction (same data scale)
    2. Separate colourbar for difference data (different scale, often symmetric)

    The function works with the layout from _init_axes, using dedicated
    colourbar axes when available, or falling back to automatic matplotlib placement.

    Colourbar Design:
        - Ground truth/prediction: Uses data range from plot_spec
        - Difference: Often symmetric around zero with specialised tick formatting
        - Orientation: Matches plot_spec.colourbar_location ('vertical'/'horizontal')

    Args:
        axs: List of main plot axes in order [ground_truth, prediction, difference*]
             (*difference only present if include_difference=True)
        image_groundtruth: QuadContourSet from ground truth plot - provides colour mapping
                          and normalisation for shared ground truth/prediction colourbar
        image_prediction: Optional QuadContourSet from prediction plot.
        image_difference: Optional QuadContourSet from difference plot - provides
                         specialised colour mapping (often symmetric) for difference colourbar
        plot_spec: PlotSpec containing colourbar orientation, value ranges, and formatting
        diff_colour_scale: Optional difference colour scale specification.
        cbar_axes: Optional dict with pre-allocated colourbar axes from _init_axes:
                  {"prediction": Axes|None, "difference": Axes|None}
                  If None or keys are None, matplotlib automatically positions colourbars

    Layout:
        - cbar_axes["prediction"]: Used for shared ground truth/prediction colourbar
        - cbar_axes["difference"]: Used for difference panel colourbar

    """
    orientation = plot_spec.colourbar_location
    is_vertical = orientation == "vertical"
    separate_colourbars = plot_spec.colourbar_strategy == "separate"

    if separate_colourbars:
        # Create individual colourbars for each panel

        # Ground truth colourbar
        if cbar_axes and cbar_axes.get("groundtruth"):
            colourbar_groundtruth = plt.colorbar(
                image_groundtruth, cax=cbar_axes["groundtruth"], orientation=orientation
            )
            _format_truth_prediction_ticks(
                colourbar_groundtruth, plot_spec, is_vertical=is_vertical
            )

        # Prediction colourbar
        if image_prediction and cbar_axes and cbar_axes.get("prediction"):
            colourbar_prediction = plt.colorbar(
                image_prediction, cax=cbar_axes["prediction"], orientation=orientation
            )
            _format_prediction_ticks(colourbar_prediction, is_vertical=is_vertical)
    else:
        # Create shared colourbar for ground truth and prediction panels
        if cbar_axes is not None and cbar_axes.get("prediction") is not None:
            # Use dedicated colourbar axis (preferred - better layout control)
            colourbar_truth = plt.colorbar(
                image_groundtruth, cax=cbar_axes["prediction"], orientation=orientation
            )
        else:
            # Fallback: automatic positioning across both panels
            colourbar_truth = plt.colorbar(
                image_groundtruth, ax=[axs[0], axs[1]], orientation=orientation
            )

        # Tick formatting
        _format_truth_prediction_ticks(
            colourbar_truth, plot_spec, is_vertical=is_vertical
        )

    # Create separate colourbar for difference panel (if present)
    # Difference data often has different scale and symmetric range around zero
    if plot_spec.include_difference and image_difference is not None:
        # Build a fixed ScalarMappable so the colourbar never depends on per-frame QuadContourSet
        sm = None
        if diff_colour_scale is not None:
            if diff_colour_scale.norm is not None:
                sm = plt.cm.ScalarMappable(
                    norm=diff_colour_scale.norm, cmap=diff_colour_scale.cmap
                )
            else:
                sm = plt.cm.ScalarMappable(
                    norm=Normalize(
                        vmin=diff_colour_scale.vmin, vmax=diff_colour_scale.vmax
                    ),
                    cmap=diff_colour_scale.cmap,
                )
            sm.set_array([])

        if cbar_axes is not None and cbar_axes.get("difference") is not None:
            colourbar_diff = plt.colorbar(
                sm if sm is not None else image_difference,
                cax=cbar_axes["difference"],
                orientation=orientation,
            )
        else:
            colourbar_diff = plt.colorbar(
                sm if sm is not None else image_difference,
                ax=axs[2],
                orientation=orientation,
            )

        # Tick formatting, ensures zero tick is always present and often symmetric
        _format_difference_ticks(
            colourbar_diff, image_difference, is_vertical=is_vertical
        )


# --- Tick Formatting Functions ---


def _format_truth_prediction_ticks(
    colourbar: Any,  # noqa: ANN401
    plot_spec: PlotSpec,
    *,
    is_vertical: bool,
) -> None:
    """Tick formatting for ground truth and prediction colourbar: always 5 ticks.

    Prefers explicit vmin/vmax from the spec; falls back to the mappable limits.
    """
    axis = colourbar.ax.yaxis if is_vertical else colourbar.ax.xaxis

    if plot_spec.vmin is not None and plot_spec.vmax is not None:
        vmin, vmax = float(plot_spec.vmin), float(plot_spec.vmax)
    else:
        vmin, vmax = _get_cbar_limits_from_mappable(colourbar)

    ticks = np.linspace(vmin, vmax, 5)
    colourbar.set_ticks(ticks)
    axis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))

    if not is_vertical:
        colourbar.ax.xaxis.set_tick_params(pad=1)


def _format_prediction_ticks(
    colourbar: Any,  # noqa: ANN401
    *,
    is_vertical: bool,
) -> None:
    """Tick formatting for prediction colourbar when using separate strategy: 5 ticks."""
    axis = colourbar.ax.yaxis if is_vertical else colourbar.ax.xaxis
    vmin, vmax = _get_cbar_limits_from_mappable(colourbar)
    ticks = np.linspace(vmin, vmax, 5)
    colourbar.set_ticks(ticks)
    axis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
    if not is_vertical:
        colourbar.ax.xaxis.set_tick_params(pad=1)


def _format_difference_ticks(
    colourbar: Any,  # noqa: ANN401
    image_difference: Any,  # noqa: ANN401
    *,
    is_vertical: bool,
) -> None:
    """Tick formatting for difference/comparison colourbars: always 5 ticks.

    Signed differences use TwoSlopeNorm; force the middle tick to 0.0 and place
    midpoints halfway to each end. Absolute modes use 5 evenly spaced ticks.
    """
    axis = colourbar.ax.yaxis if is_vertical else colourbar.ax.xaxis

    if isinstance(image_difference.norm, TwoSlopeNorm):
        vmin = float(image_difference.norm.vmin or -1.0)
        vmax = float(image_difference.norm.vmax or 1.0)
        ticks = np.array(
            [vmin, 0.5 * (vmin + 0.0), 0.0, 0.5 * (0.0 + vmax), vmax], dtype=float
        )
        colourbar.set_ticks(ticks)
        axis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
    else:
        vmin, vmax = _get_cbar_limits_from_mappable(colourbar)
        ticks = np.linspace(vmin, vmax, 5)
        colourbar.set_ticks(ticks)
        axis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))

    if not is_vertical:
        colourbar.ax.xaxis.set_tick_params(pad=1)
