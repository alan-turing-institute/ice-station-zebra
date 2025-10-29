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

    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.contour import QuadContourSet
    from matplotlib.figure import Figure

    from .plotting_core import DiffColourmapSpec, PlotSpec

# Layout constants
PREDICTION_PANEL_INDEX = 1
DIFFERENCE_PANEL_INDEX = 2
MIN_PANEL_COUNT_FOR_PREDICTION = 2
MIN_PANEL_COUNT_FOR_DIFFERENCE = 3


#
# --- LAYOUT CONFIGURATION CONSTANTS ---
# These constants define the default proportional spacing for the GridSpec layout.
# All values are expressed as fractions of the total figure dimensions to ensure
# consistent scaling across different figure sizes.

# Spacing and margin constants (as fractions of figure dimensions)
DEFAULT_OUTER_MARGIN = 0.05  # Outer margin around entire figure (prevents clipping)
DEFAULT_GUTTER_HORIZONTAL = 0.03  # Smaller gaps when colourbar is below
DEFAULT_GUTTER_VERTICAL = 0.03  # Default for side-by-side with vertical bars
DEFAULT_CBAR_WIDTH = (
    0.06  # Width allocated for colourbar slots (fraction of panel width)
)
DEFAULT_TITLE_SPACE = 0.11  # Allow for the figure title, warning badge, and axes titles

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


def _build_layout(  # noqa: PLR0913
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
    # Decide how many main panels are required and which orientation the colourbars use
    n_panels = 3 if plot_spec.include_difference else 2
    orientation = plot_spec.colourbar_location

    # Choose gutter default per orientation if not provided
    if gutter is None:
        gutter = (
            DEFAULT_GUTTER_HORIZONTAL
            if orientation == "horizontal"
            else DEFAULT_GUTTER_VERTICAL
        )

    # Calculate top boundary: ensure title space does not consume too much of the figure.
    # At least 60% of the figure height is reserved for the plotting area.
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
                n_panels * panel_w
                + (n_panels - 1) * gutter * panel_w
                + (n_panels - 1) * cbar_width * panel_w
            )
        else:
            # --- Horizontal colourbars ---
            # Portion of height available to the plot row once title and margins are considered
            usable_h_frac = top_val - outer_margin
            plot_row_frac = 1.0 / (1.0 + cbar_pad + cbar_height)

            effective_plot_h = base_h * usable_h_frac * plot_row_frac
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

    fig = plt.figure(figsize=fig_size, constrained_layout=False, facecolor="none")

    if orientation == "vertical":
        # Delegate to the vertical builder which organises columns for panels and colourbars
        axs, caxes = _build_grid_vertical(
            fig,
            n_panels=n_panels,
            plot_spec=plot_spec,
            outer_margin=outer_margin,
            gutter=gutter,
            cbar_width=cbar_width,
            top_val=top_val,
        )
    else:
        # Delegate to the horizontal builder which organises rows for plots and colourbars
        axs, caxes = _build_grid_horizontal(
            fig,
            n_panels=n_panels,
            plot_spec=plot_spec,
            outer_margin=outer_margin,
            gutter=gutter,
            cbar_height=cbar_height,
            cbar_pad=cbar_pad,
            top_val=top_val,
        )

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


def _build_grid_vertical(  # noqa: PLR0913, C901, PLR0912
    fig: Figure,
    *,
    n_panels: int,
    plot_spec: PlotSpec,
    outer_margin: float,
    gutter: float,
    cbar_width: float,
    top_val: float,
) -> tuple[list[Axes], dict[str, Axes | None]]:
    """Construct a one-row GridSpec with vertical colourbars.

    Layout overview (left to right):
    - When using shared colourbars: [GroundTruth][Prediction][cbar][gutter][Difference][cbar]
    - When using separate colourbars: [GT][cbar][gutter][Pred][cbar][gutter][Diff][cbar]

    Args:
        fig: The target Matplotlib figure to attach the GridSpec to.
        n_panels: Number of main panels to create (2 or 3).
        plot_spec: Plot configuration containing colourbar strategy and titles.
        outer_margin: Fractional margin applied around the figure.
        gutter: Fractional spacing between panel groups.
        cbar_width: Fractional width allocated to colourbar slots.
        top_val: The top boundary of the usable plotting area (accounts for title space).

    Returns:
        A tuple of (axes, colourbar_axes) where axes are the main plot axes in order
        and colourbar_axes is a dict containing dedicated colourbar axes if present.

    """
    # Panel arrangement: [GT][Pred][cbar][gutter][Diff][cbar]
    col_specs: list[float] = []
    separate_colourbars = plot_spec.colourbar_strategy == "separate"

    if separate_colourbars:
        # Each panel gets its own colourbar: [GT][cbar][gutter][Pred][cbar][gutter][Diff][cbar]
        for ii in range(n_panels):
            col_specs.append(1.0)
            col_specs.append(cbar_width)
            if ii < n_panels - 1:
                col_specs.append(gutter)
    else:
        for ii in range(n_panels):
            col_specs.append(1.0)
            if ii == PREDICTION_PANEL_INDEX:
                col_specs.append(cbar_width)
            if ii == DIFFERENCE_PANEL_INDEX:
                col_specs.append(cbar_width)
            if ii != n_panels - 1 and ii > 0:
                col_specs.append(gutter)

    # One-row grid; width ratios encode panels, colourbars and gutters
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

    axs: list[Axes] = []
    caxes: dict[str, Axes | None] = {
        "groundtruth": None,
        "prediction": None,
        "difference": None,
    }

    col_idx = 0
    for ii in range(n_panels):
        # Create the main panel axis
        ax = fig.add_subplot(gs[0, col_idx])
        axs.append(ax)
        col_idx += 1

        if separate_colourbars:
            panel_names = ["groundtruth", "prediction", "difference"]
            if ii < len(panel_names):
                # Create a dedicated colourbar axis immediately to the right of the panel
                caxes[panel_names[ii]] = fig.add_subplot(gs[0, col_idx])
            col_idx += 1
            if ii < n_panels - 1:
                col_idx += 1
        else:
            if ii == PREDICTION_PANEL_INDEX:
                caxes["prediction"] = fig.add_subplot(gs[0, col_idx])
                col_idx += 1
            if ii == DIFFERENCE_PANEL_INDEX:
                caxes["difference"] = fig.add_subplot(gs[0, col_idx])
                col_idx += 1
            if ii == PREDICTION_PANEL_INDEX and ii != n_panels - 1:
                col_idx += 1

    return axs, caxes


def _build_grid_horizontal(  # noqa: PLR0913, PLR0912
    fig: Figure,
    *,
    n_panels: int,
    plot_spec: PlotSpec,
    outer_margin: float,
    gutter: float,
    cbar_height: float,
    cbar_pad: float,
    top_val: float,
) -> tuple[list[Axes], dict[str, Axes | None]]:
    """Construct a three-row GridSpec with horizontal colourbars.

    Grid structure (rows):
    - Row 0: Main plot panels
    - Row 1: Padding row to separate plots from colourbars
    - Row 2: Colourbar row (inset axes centred within parent slots)

    Column pattern alternates panels and gutters, with the trailing gutter removed.

    Args:
        fig: The target Matplotlib figure to attach the GridSpec to.
        n_panels: Number of main panels to create (2 or 3).
        plot_spec: Plot configuration containing colourbar strategy.
        outer_margin: Fractional margin applied around the figure.
        gutter: Fractional spacing between panel groups.
        cbar_height: Fractional height allocated to the colourbar row.
        cbar_pad: Fractional padding between the plot row and colourbar row.
        top_val: The top boundary of the usable plotting area (accounts for title space).

    Returns:
        A tuple of (axes, colourbar_axes) where axes are the main plot axes in order
        and colourbar_axes is a dict containing dedicated colourbar axes if present.

    """
    # Grid structure: 3 rows ([plots][pad][colourbars])
    width_ratios = [1.0, gutter] * n_panels
    width_ratios = width_ratios[:-1]

    # Three-row grid; height ratios encode plot, pad, and colourbar rows
    gs = GridSpec(
        nrows=3,
        ncols=len(width_ratios),
        figure=fig,
        width_ratios=width_ratios,
        height_ratios=[1.0, cbar_pad, cbar_height],
        left=outer_margin,
        right=1 - outer_margin,
        top=top_val,
        bottom=outer_margin,
        wspace=0.0,
        hspace=0.0,
    )

    axs: list[Axes] = []
    separate_colourbars = plot_spec.colourbar_strategy == "separate"
    caxes: dict[str, Axes | None] = {
        "groundtruth": None,
        "prediction": None,
        "difference": None,
    }

    for i in range(n_panels):
        # Columns are [panel, gutter, panel, gutter, ...]; select even columns for panels
        panel_col = 2 * i
        axs.append(fig.add_subplot(gs[0, panel_col]))

    for ax in axs:
        # Anchor to the bottom of the cell to avoid vertical slack above panels
        ax.set_anchor("S")

    def _inset_cbar(parent_ax: Axes) -> Axes:
        """Create a centred inset colourbar axis within parent, using configured fractions."""
        parent_ax.set_axis_off()
        return parent_ax.inset_axes((HCBAR_LEFT_FRAC, 0.0, HCBAR_WIDTH_FRAC, 1.0))

    if separate_colourbars:
        # ---- Separate colourbar logic ----
        # Ground truth bar under first panel (if present)
        if n_panels >= 1:
            caxes["groundtruth"] = _inset_cbar(fig.add_subplot(gs[2, 0]))
        else:
            caxes["groundtruth"] = None

        # Prediction bar under second panel (if present)
        if n_panels >= MIN_PANEL_COUNT_FOR_PREDICTION:
            caxes["prediction"] = _inset_cbar(fig.add_subplot(gs[2, 2]))
        else:
            caxes["prediction"] = None

        # Difference bar under third panel (if present)
        if n_panels >= MIN_PANEL_COUNT_FOR_DIFFERENCE:
            caxes["difference"] = _inset_cbar(fig.add_subplot(gs[2, 4]))
        else:
            caxes["difference"] = None
    else:
        # ---- Shared colourbar logic ----
        # Under GT+gutter+Pred span columns 0..3 when two panels present
        if n_panels >= MIN_PANEL_COUNT_FOR_PREDICTION:
            caxes["prediction"] = _inset_cbar(fig.add_subplot(gs[2, 0:3]))
        else:
            # Fallback: only GT present; use full column
            caxes["prediction"] = _inset_cbar(fig.add_subplot(gs[2, 0:1]))

        # Difference bar under third panel (if present)
        if n_panels >= MIN_PANEL_COUNT_FOR_DIFFERENCE:
            caxes["difference"] = _inset_cbar(fig.add_subplot(gs[2, 4]))
        else:
            caxes["difference"] = None

    return axs, caxes


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
            ax.set_title(
                title,
                fontfamily="monospace",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "pad": 2.0,
                    "alpha": 1.0,
                },
            )


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

    Ensures all axes display the same spatial extent following polar mapping conventions.

    Args:
        axs: List of matplotlib Axes objects to configure
        width: Width of the data array (sets x-axis limits: 0 to width)
        height: Height of the data array (sets y-axis limits: height to 0 for polar data)

    Note:
        The y-axis is inverted (height to 0) to follow environmental science conventions
        for polar data visualisation, where higher latitude values are
        positioned at the top of the display for both Arctic and Antarctic regions.

    """
    for ax in axs:
        ax.set_xlim(0, width)  # X-axis: left edge to right edge
        ax.set_ylim(
            height, 0
        )  # Y-axis: top edge to bottom edge (geographical convention)


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


def _add_colourbars(  # noqa: PLR0913, PLR0912
    axs: list[Axes],
    *,
    image_groundtruth: QuadContourSet,
    image_prediction: QuadContourSet | None = None,
    image_difference: QuadContourSet | None = None,
    plot_spec: PlotSpec,
    diff_colour_scale: DiffColourmapSpec | None = None,
    cbar_axes: dict[str, Axes | None] | None = None,
) -> None:
    """Create and position colourbars.

    This function creates two types of colourbars:
    1. Shared colourbar for ground truth and prediction (same data scale)
    2. Separate colourbar for difference data (different scale, often symmetric)

    The function works with the layout from _build_layout, using dedicated
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
        cbar_axes: Optional dict with pre-allocated colourbar axes from layout builders:
                  {"groundtruth": Axes|None, "prediction": Axes|None, "difference": Axes|None}
                  None values mean "no dedicated axis" and will fall back to automatic placement.

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
            _format_linear_ticks(
                colourbar_groundtruth,
                vmin=float(plot_spec.vmin) if plot_spec.vmin is not None else None,
                vmax=float(plot_spec.vmax) if plot_spec.vmax is not None else None,
                decimals=1,
                is_vertical=is_vertical,
            )

        # Prediction colourbar
        if image_prediction and cbar_axes and cbar_axes.get("prediction"):
            colourbar_prediction = plt.colorbar(
                image_prediction, cax=cbar_axes["prediction"], orientation=orientation
            )
            _format_linear_ticks(
                colourbar_prediction, decimals=1, is_vertical=is_vertical
            )
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
        _format_linear_ticks(
            colourbar_truth,
            vmin=float(plot_spec.vmin) if plot_spec.vmin is not None else None,
            vmax=float(plot_spec.vmax) if plot_spec.vmax is not None else None,
            decimals=1,
            is_vertical=is_vertical,
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

        # Tick formatting: symmetric for TwoSlopeNorm, otherwise linear
        if isinstance(image_difference.norm, TwoSlopeNorm):
            vmin = float(image_difference.norm.vmin or -1.0)
            vmax = float(image_difference.norm.vmax or 1.0)
            _format_symmetric_ticks(
                colourbar_diff,
                vmin=vmin,
                vmax=vmax,
                decimals=2,
                is_vertical=is_vertical,
            )
        else:
            _format_linear_ticks(colourbar_diff, decimals=2, is_vertical=is_vertical)


# --- Tick Formatting Functions ---


def _format_linear_ticks(
    colourbar: Colorbar,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    decimals: int = 1,
    is_vertical: bool,
) -> None:
    """Format a linear colourbar with 5 ticks.

    If vmin/vmax are not provided, derive them from the colourbar's mappable.
    """
    axis = colourbar.ax.yaxis if is_vertical else colourbar.ax.xaxis

    if vmin is None or vmax is None:
        mvmin, mvmax = _get_cbar_limits_from_mappable(colourbar)
        vmin = mvmin if vmin is None else vmin
        vmax = mvmax if vmax is None else vmax

    ticks = np.linspace(float(vmin), float(vmax), 5)
    colourbar.set_ticks([float(t) for t in ticks])
    axis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.{decimals}f}"))
    if not is_vertical:
        colourbar.ax.xaxis.set_tick_params(pad=1)
    _apply_monospace_to_cbar_text(colourbar)


def _format_symmetric_ticks(
    colourbar: Colorbar,
    *,
    vmin: float,
    vmax: float,
    decimals: int = 2,
    is_vertical: bool,
) -> None:
    """Format symmetric diverging ticks with a 0-centered midpoint.

    Places five ticks: [vmin, midpoint to 0, 0, 0 to midpoint, vmax].
    """
    axis = colourbar.ax.yaxis if is_vertical else colourbar.ax.xaxis
    ticks = [vmin, 0.5 * (vmin + 0.0), 0.0, 0.5 * (0.0 + vmax), vmax]
    colourbar.set_ticks([float(t) for t in ticks])
    axis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.{decimals}f}"))
    if not is_vertical:
        colourbar.ax.xaxis.set_tick_params(pad=1)
    _apply_monospace_to_cbar_text(colourbar)


def _apply_monospace_to_cbar_text(colourbar: Colorbar) -> None:
    """Set tick labels and axis labels on a colourbar to monospace family."""
    ax = colourbar.ax
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_fontfamily("monospace")
    # Ensure axis labels also use monospace if present
    ax.xaxis.label.set_fontfamily("monospace")
    ax.yaxis.label.set_fontfamily("monospace")
