"""
Plot Map Layout Module

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

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, MaxNLocator

from .plotting_core import DiffColourmapSpec, PlotSpec

#
# --- LAYOUT CONFIGURATION CONSTANTS ---
# These constants define the default proportional spacing for the GridSpec layout.
# All values are expressed as fractions of the total figure dimensions to ensure
# consistent scaling across different figure sizes.

# Spacing and margin constants (as fractions of figure dimensions)
DEFAULT_OUTER_MARGIN = 0.05  # Outer margin around entire figure (prevents clipping)
DEFAULT_GUTTER = 0.05  # Horizontal gap between panels (fraction of panel width)
DEFAULT_CBAR_WIDTH = (
    0.06  # Width allocated for colourbar slots (fraction of panel width)
)
DEFAULT_TITLE_SPACE = (
    0.08  # Vertical space reserved for figure title (prevents overlap)
)

# Default figure sizes for different panel configurations
DEFAULT_FIGSIZE_TWO_PANELS = (12, 6)  # Ground truth + Prediction
DEFAULT_FIGSIZE_THREE_PANELS = (18, 6)  # Ground truth + Prediction + Difference


# --- Main Layout Functions ---


def _init_axes(
    *,
    plot_spec: PlotSpec,
    H: int | None = None,
    W: int | None = None,
    outer_margin: float = DEFAULT_OUTER_MARGIN,
    gutter: float = DEFAULT_GUTTER,
    cbar_width: float = DEFAULT_CBAR_WIDTH,
    title_space: float = DEFAULT_TITLE_SPACE,
) -> tuple[Figure, list[Axes], dict[str, Axes | None]]:
    """
    Create a GridSpec layout for multi-panel plots.

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
        H: Optional data height for aspect-ratio-aware figure sizing. If provided with W,
           the figure dimensions will be calculated to maintain proper data aspect ratios.
        W: Optional data width for aspect-ratio-aware figure sizing.
        outer_margin: Fraction of figure dimensions reserved for outer margins (prevents
                     plot elements from being clipped at figure edges).
        gutter: Fraction of panel width used as horizontal spacing between panel groups.
               Only applied between prediction+colourbar and difference panel.
        cbar_width: Fraction of panel width allocated for each colourbar slot.
        title_space: Fraction of figure height reserved at the top for figure title
                    (prevents title from overlapping with plot content).

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

    # Calculate figure size based on data aspect ratio or use defaults
    if H and W and H > 0:
        # Calculate panel width maintaining data aspect ratio
        base_h = 6.0  # Standard height in inches
        aspect = W / H
        panel_w = base_h * aspect

        if orientation == "vertical":
            # Account for colourbars: panels + gutters + colourbar slots
            fig_w = (
                n_panels * panel_w  # Main panels
                + (n_panels - 1) * gutter * panel_w  # Gutters between groups
                + (n_panels - 1) * cbar_width * panel_w  # Colourbar slots
            )
        else:
            # Horizontal colourbars don't affect width calculation
            fig_w = n_panels * panel_w + (n_panels - 1) * gutter * panel_w
        fig_size = (fig_w, base_h)
    else:
        # Use predefined sizes when data dimensions are unknown
        fig_size = (
            DEFAULT_FIGSIZE_THREE_PANELS
            if plot_spec.include_difference
            else DEFAULT_FIGSIZE_TWO_PANELS
        )

    # Calculate top boundary: ensure title space doesn't consume too much of figure
    # Minimum 60% of figure height must remain for plots
    top_val = max(0.6, 1.0 - (outer_margin + title_space))

    fig = plt.figure(figsize=fig_size, constrained_layout=False)

    if orientation == "vertical":
        # VERTICAL COLOURBAR LAYOUT
        # Panel arrangement: [GT][Pred][cbar][gutter][Diff][cbar]
        # This creates columns for each element with appropriate relative widths

        col_specs: list[float] = []
        separate_colorbars = plot_spec.colourbar_strategy == "separate"

        if separate_colorbars:
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
                if ii == 1:  # After prediction panel
                    col_specs.append(cbar_width)
                if ii == 2:  # After difference panel (if present)
                    col_specs.append(cbar_width)

                # Add gutter spacing, but only between panel groups (not after ground truth)
                if ii != n_panels - 1:  # Not the last panel
                    if ii > 0:  # Not the first panel (ground truth)
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
        separate_colorbars = plot_spec.colourbar_strategy == "separate"

        if separate_colorbars:
            caxes: dict[str, Axes | None] = {
                "groundtruth": None,
                "prediction": None,
                "difference": None,
            }
        else:
            caxes: dict[str, Axes | None] = {"prediction": None, "difference": None}
        col_idx = 0

        for ii in range(n_panels):
            # Create main plot axis (ground truth, prediction, or difference)
            ax = fig.add_subplot(gs[0, col_idx])
            axs.append(ax)
            col_idx += 1

            # Create dedicated colourbar axes adjacent to certain panels
            if separate_colorbars:
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
                if ii == 1:  # After prediction panel: shared GT/prediction colorbar
                    caxes["prediction"] = fig.add_subplot(gs[0, col_idx])
                    col_idx += 1
                if ii == 2:  # After difference panel: difference colorbar
                    caxes["difference"] = fig.add_subplot(gs[0, col_idx])
                    col_idx += 1
                # Skip over gutter columns (spacing between panel groups)
                if ii == 1 and ii != n_panels - 1:  # After pred+cbar group, before diff
                    col_idx += 1

    else:
        # HORIZONTAL COLOURBAR LAYOUT
        # Grid structure: 2 rows (plots on top, colourbars on bottom)
        # Column pattern: [panel][panel][gutter][panel]

        width_ratios = [1.0, gutter] * n_panels  # Alternate panels and gutters
        width_ratios = width_ratios[:-1]  # Remove trailing gutter

        gs = GridSpec(
            nrows=2,  # Top row: plots, Bottom row: colourbars
            ncols=len(width_ratios),
            figure=fig,
            width_ratios=width_ratios,
            height_ratios=[1.0, cbar_width],  # Plot area vs colourbar height
            left=outer_margin,
            right=1 - outer_margin,
            top=top_val,
            bottom=outer_margin,
            wspace=0.0,  # Width spacing handled by explicit gutter columns
            hspace=0.0,  # Height spacing handled by explicit height ratios
        )

        # Create main plot axes in top row
        axs = []
        separate_colorbars = plot_spec.colourbar_strategy == "separate"

        if separate_colorbars:
            caxes = {"groundtruth": None, "prediction": None, "difference": None}
        else:
            caxes = {"prediction": None, "difference": None}

        for i in range(n_panels):
            panel_col = 2 * i  # Skip gutter columns: 0, 2, 4, ...
            axs.append(fig.add_subplot(gs[0, panel_col]))

        # Create colorbar axes in bottom row at specific column positions
        if separate_colorbars:
            # Each panel gets its own colourbar
            caxes["groundtruth"] = (
                fig.add_subplot(gs[1, 0]) if n_panels >= 1 else None
            )  # Under ground truth
            caxes["prediction"] = (
                fig.add_subplot(gs[1, 2]) if n_panels >= 2 else None
            )  # Under prediction
            caxes["difference"] = (
                fig.add_subplot(gs[1, 4]) if n_panels >= 3 else None
            )  # Under difference
        else:
            # Original shared colourbar logic
            caxes["prediction"] = (
                fig.add_subplot(gs[1, 2]) if n_panels >= 2 else None
            )  # Under prediction
            caxes["difference"] = (
                fig.add_subplot(gs[1, 4]) if n_panels >= 3 else None
            )  # Under difference

    _set_titles(axs, plot_spec)
    _style_axes(axs)

    fig._zebra_layout = {
        "outer_margin": outer_margin,
        "gutter": gutter,
        "cbar_width": cbar_width,
        "title_space": title_space,
    }
    return fig, axs, caxes


# --- Helper Functions ---


def _set_titles(axs: list[Axes], plot_spec: PlotSpec) -> None:
    """
    Set titles for each plot panel based on the PlotSpec configuration.

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
    for ax, title in zip(axs, titles):
        ax.set_title(title)


def _style_axes(axs: Sequence[Axes]) -> None:
    """
    Apply consistent styling to all plot axes.

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
    """
    Set consistent axis limits across all plot panels, ensures all axes display
    the same spatial extent.
    Args:
        axs: List of matplotlib Axes objects to configure
        width: Width of the data array (sets x-axis limits: 0 to width)
        height: Height of the data array (sets y-axis limits: 0 to height)
    """
    for ax in axs:
        ax.set_xlim(0, width)  # X-axis: left edge to right edge
        ax.set_ylim(0, height)  # Y-axis: bottom edge to top edge


# --- Colourbar Functions ---


def _add_colourbars(
    axs: list,
    *,
    image_groundtruth,  # QuadContourSet
    image_prediction=None,  # QuadContourSet | None
    image_difference=None,  # QuadContourSet | None
    plot_spec: PlotSpec,
    diff_colour_scale: DiffColourmapSpec | None = None,
    cbar_axes: dict[str, Axes | None] | None = None,
) -> None:
    """
    Create and position colourbars.

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
        image_difference: Optional QuadContourSet from difference plot - provides
                         specialised colour mapping (often symmetric) for difference colourbar
        plot_spec: PlotSpec containing colourbar orientation, value ranges, and formatting
        include_difference: Whether difference panel and its colourbar should be created
        cbar_axes: Optional dict with pre-allocated colourbar axes from _init_axes:
                  {"prediction": Axes|None, "difference": Axes|None}
                  If None or keys are None, matplotlib automatically positions colourbars

    Layout:
        - cbar_axes["prediction"]: Used for shared ground truth/prediction colourbar
        - cbar_axes["difference"]: Used for difference panel colourbar
    """
    orientation = plot_spec.colourbar_location
    is_vertical = orientation == "vertical"
    separate_colorbars = plot_spec.colourbar_strategy == "separate"

    if separate_colorbars:
        # Create individual colorbars for each panel

        # Ground truth colourbar
        if cbar_axes and cbar_axes.get("groundtruth"):
            colourbar_gt = plt.colorbar(
                image_groundtruth, cax=cbar_axes["groundtruth"], orientation=orientation
            )
            _format_truth_prediction_ticks(colourbar_gt, plot_spec, is_vertical)

        # Prediction colourbar
        if image_prediction and cbar_axes and cbar_axes.get("prediction"):
            colourbar_pred = plt.colorbar(
                image_prediction, cax=cbar_axes["prediction"], orientation=orientation
            )
            _format_prediction_ticks(colourbar_pred, image_prediction, is_vertical)
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
        _format_truth_prediction_ticks(colourbar_truth, plot_spec, is_vertical)

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
        _format_difference_ticks(colourbar_diff, image_difference, is_vertical)


# --- Tick Formatting Functions ---


def _format_truth_prediction_ticks(
    colourbar, plot_spec: PlotSpec, is_vertical: bool
) -> None:
    """
    Tick formatting for ground truth and prediction colourbar.

    This function provides context-aware formatting optimized for scientific data display:

    Formatting Rules:
        - [0,1] ranges (common for probabilities/concentrations): 6 evenly spaced ticks (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        - Other ranges: 5 automatically positioned ticks using MaxNLocator
        - All values: 1 decimal place precision for readability

    The formatting adapts to colorbar orientation, applying tick formatting to the
    appropriate axis (y-axis for vertical, x-axis for horizontal colorbars).

    Args:
        colourbar: matplotlib Colorbar object
        plot_spec: PlotSpec containing vmin/vmax for range detection and formatting context
        is_vertical: Whether the colorbar is oriented vertically (affects which axis to format)

    """
    # Check if data is in the standard [0,1] range (e.g. for concentrations/probabilities)
    is_unit_range = (
        plot_spec.vmin is not None
        and plot_spec.vmax is not None
        and 0.0 <= plot_spec.vmin <= 1.0
        and 0.0 <= plot_spec.vmax <= 1.0
    )

    # Get appropriate axis based on colorbar orientation
    target_axis = colourbar.ax.yaxis if is_vertical else colourbar.ax.xaxis

    if is_unit_range:
        # For [0,1] data: use 6 evenly spaced ticks for clear percentage-like values
        colourbar.set_ticks(np.linspace(0.0, 1.0, 6))
        target_axis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1f}"))
    else:
        # For arbitrary ranges: let matplotlib choose ~5 good tick positions
        target_axis.set_major_locator(MaxNLocator(5))
        target_axis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1f}"))


def _format_prediction_ticks(colourbar, image_prediction, is_vertical: bool) -> None:
    """
    Tick formatting for prediction colourbar when using separate strategy.

    Uses automatic tick placement for data-driven ranges rather than
    the spec vmin/vmax range.

    Args:
        colourbar: matplotlib Colorbar object
        image_prediction: ContourSet containing the prediction data range
        is_vertical: Whether the colourbar is oriented vertically
    """
    target_axis = colourbar.ax.yaxis if is_vertical else colourbar.ax.xaxis

    # Use automatic tick placement for data-driven ranges
    target_axis.set_major_locator(MaxNLocator(5))
    target_axis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1f}"))


def _format_difference_ticks(colourbar, image_difference, is_vertical: bool) -> None:
    """
    Tick formatting for difference/comparison colourbars.

    Difference colourbars often use symmetric color scales (e.g., blue-white-red) to
    highlight positive and negative deviations from zero. This function ensures that
    zero is prominently displayed and that tick values provide meaningful context
    for the magnitude of differences.

    Formatting Features:
        - TwoSlopeNorm detection: Ensures zero is always included in tick marks
        - Symmetric tick placement: 5 evenly spaced ticks from vmin to vmax
        - Higher precision: 2 decimal places for difference values (more precise than raw data)
        - Zero-centered: Middle tick is always exactly 0.0 for symmetric scales

    Args:
        colourbar: matplotlib Colorbar object
        image_difference: ContourSet or QuadMesh containing the normalization object
                         used for the difference plot (checked for TwoSlopeNorm)
        is_vertical: Whether the colorbar is oriented vertically (affects axis selection)

    Example:
        For a difference range [-0.15, +0.15]:
        - Ticks at: -0.15, -0.08, 0.00, +0.08, +0.15,
        - Zero forced in the middle position
    """
    # Handle symmetric normalisation (common for difference plots)
    if isinstance(image_difference.norm, TwoSlopeNorm):
        # Extract actual data range from the normalization
        vmin_actual = float(image_difference.norm.vmin)
        vmax_actual = float(image_difference.norm.vmax)

        # Create 5 evenly spaced ticks across the range
        ticks = np.linspace(vmin_actual, vmax_actual, 5)

        # Force the middle tick to be exactly zero for symmetric display
        if ticks.size >= 3:
            ticks[ticks.size // 2] = 0.0

        colourbar.set_ticks(ticks)

    # 2 decimal places for difference values
    target_axis = colourbar.ax.yaxis if is_vertical else colourbar.ax.xaxis
    target_axis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2f}"))
