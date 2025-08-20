from __future__ import annotations

import io
from datetime import date, datetime
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from PIL import Image
from PIL.ImageFile import ImageFile

from .plotting_core import (
    PlotSpec,
    compute_difference,
    validate_2d_pair,
)

# -- Constants --

N_CONTOUR_LEVELS = 100
DEFAULT_FIGSIZE_TWO_PANELS = (12, 6)
DEFAULT_FIGSIZE_THREE_PANELS = (18, 6)
DEFAULT_DPI = 200

DEFAULT_SIC_SPEC = PlotSpec(
    variable="sea_ice_concentration",
    title_groundtruth="Ground Truth",
    title_prediction="Prediction",
    title_difference="Difference",
    colourmap="viridis",
    diff_mode="signed",
    vmin=0.0,
    vmax=1.0,
    colourbar_location="vertical",
)

# ---- Main functions ----


# - Static plot -
def plot_maps(
    plot_spec: PlotSpec,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    date: date | datetime,
    include_difference: bool = True,
) -> list[ImageFile]:
    # Check the shapes of the arrays
    height, width = validate_2d_pair(ground_truth, prediction)

    # Initialise the figure and axes
    fig, axs = _init_axes(include_difference, width, height, plot_spec)
    levels = _levels_from_spec(plot_spec)

    # Draw the ground truth and prediction map images
    image_groundtruth, image_prediction, image_difference, _ = _draw_frame(
        axs,
        ground_truth,
        prediction,
        plot_spec,
        include_difference,
        diff_colour_norm=None,
        precomputed_diff_2d=None,
        levels_override=levels,
    )

    # Colourbars and title
    _add_colourbars(
        axs, image_groundtruth, image_difference, plot_spec, include_difference
    )

    _set_axes_limits(axs, width, height)
    fig.suptitle(_format_date_to_string(date))
    fig.tight_layout()

    try:
        return [_image_from_figure(fig)]
    finally:
        plt.close(fig)


# ---- Helper functions ----


def _draw_frame(
    axs: list,
    *,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    plot_spec: PlotSpec = DEFAULT_SIC_SPEC,
    include_difference: bool = True,
    diff_colour_norm: TwoSlopeNorm | None = None,
    precomputed_difference: np.ndarray | None = None,
    levels_override: np.ndarray | None = None,
) -> tuple:
    """Draw a full frame (ground truth, prediction, OPTIONAL difference)."""

    for ax in axs:
        _clear_contours(ax)

    # Use PlotSpec levels for ground_truth and prediction unless overridden
    levels = (
        _levels_from_spec(plot_spec) if levels_override is None else levels_override
    )

    image_groundtruth = axs[0].contourf(
        ground_truth,
        levels=levels,
        cmap=plot_spec.colourmap,
        vmin=plot_spec.vmin,
        vmax=plot_spec.vmax,
    )
    image_prediction = axs[1].contourf(
        prediction,
        levels=levels,
        cmap=plot_spec.colourmap,
        vmin=plot_spec.vmin,
        vmax=plot_spec.vmax,
    )

    image_difference = None
    if include_difference:
        difference = (
            precomputed_difference
            if precomputed_difference is not None
            else compute_difference(ground_truth, prediction, plot_spec.diff_mode)
        )

        if plot_spec.diff_mode == "signed":
            # Stable, symmetric colour scaling centred at 0
            if diff_colour_norm is None:
                rng = float(np.nanmax(np.abs(difference)) or 1.0)
                diff_colour_norm = TwoSlopeNorm(vmin=-rng, vcenter=0.0, vmax=rng)
            image_difference = axs[2].contourf(
                difference,
                levels=N_CONTOUR_LEVELS,
                cmap="RdBu_r",
                norm=diff_colour_norm,
            )
        else:
            # Non-negative differences
            image_difference = axs[2].contourf(
                difference, levels=N_CONTOUR_LEVELS, cmap="magma"
            )

    return image_groundtruth, image_prediction, image_difference, diff_colour_norm


def _image_from_figure(fig: Figure) -> ImageFile:
    """Convert a matplotlib figure to a PIL image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return Image.open(buf)  # type: ignore[return-value]


def _format_date_to_string(date: date | datetime) -> str:
    """Format a date to a string."""
    return (
        date.strftime(r"%Y-%m-%d %H:%M")
        if isinstance(date, datetime)
        else date.strftime(r"%Y-%m-%d")
    )


def _levels_from_spec(spec: PlotSpec) -> np.ndarray:
    """Contour levels from vmin/vmax; default to [0,1] if unspecified."""
    vmin = 0.0 if spec.vmin is None else spec.vmin
    vmax = 1.0 if spec.vmax is None else spec.vmax
    return np.linspace(vmin, vmax, N_CONTOUR_LEVELS)


def _init_axes(
    *, include_difference: bool, spec: PlotSpec
) -> tuple[Figure, list[Axes]]:
    """Create figure and 2 or 3 axes; set titles; no data-dependent limits here."""
    ncols = 3 if include_difference else 2
    figsize = (
        DEFAULT_FIGSIZE_THREE_PANELS
        if include_difference
        else DEFAULT_FIGSIZE_TWO_PANELS
    )
    fig: Figure = plt.figure(figsize=figsize)
    axs = [fig.add_subplot(1, ncols, i + 1) for i in range(ncols)]

    titles = [spec.title_groundtruth, spec.title_prediction]
    if include_difference:
        titles.append(spec.title_difference)
    for ax, title in zip(axs, titles):
        ax.set_title(title)

    _style_axes(axs)
    return fig, axs


def _style_axes(axs: Sequence[Axes]) -> None:
    """Cosmetic styling only (ticks/aspect) — no limits."""
    for ax in axs:
        ax.axis("off")
        ax.set_aspect("equal")


def _set_axes_limits(axs: Sequence[Axes], *, width: int, height: int) -> None:
    """Apply data-driven limits AFTER drawing (so contours don’t rescale unexpectedly)."""
    for ax in axs:
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)


def _clear_contours(ax) -> None:
    """Remove previous contour collections (used during animation)."""
    for coll in ax.collections[:]:
        coll.remove()


def _add_colourbars(
    axs: list,
    *,
    image_groundtruth: plt.contourf,
    image_difference: plt.contourf | None = None,
    plot_spec: PlotSpec = DEFAULT_SIC_SPEC,
    include_difference: bool = True,
) -> None:
    """Add colourbars to the axes. One colour bar shared by the
    ground truth and prediction, and (optionally) one for the difference.
    """
    if include_difference and image_difference is not None:
        plt.colorbar(
            image_groundtruth,
            ax=[axs[0], axs[1]],
            orientation=plot_spec.colourbar_location,
        )
        plt.colorbar(
            image_difference,
            ax=[axs[2]],
            orientation=plot_spec.colourbar_location,
        )
    else:
        plt.colorbar(
            image_groundtruth,
            ax=[axs[0], axs[1]],
            orientation=plot_spec.colourbar_location,
        )
