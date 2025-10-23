"""Sea ice concentration visualisation module for creating static maps and animations.

- Static map plotting with optional difference visualisation
- Animated video generation (MP4/GIF formats)
- Flexible plotting specifications and colour schemes
- Multiple difference computation strategies
- Date formatting

"""

import contextlib
import io
from collections.abc import Sequence
from datetime import date, datetime
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from PIL.ImageFile import ImageFile

from ice_station_zebra.exceptions import InvalidArrayError
from ice_station_zebra.types import DiffColourmapSpec, PlotSpec

from . import convert
from .layout import _add_colourbars, _build_layout, _set_axes_limits
from .plotting_core import (
    compute_difference,
    compute_display_ranges,
    compute_display_ranges_stream,
    levels_from_spec,
    load_land_mask,
    make_diff_colourmap,
    prepare_difference_stream,
    validate_2d_pair,
    validate_3d_streams,
)
from .range_check import compute_range_check_report

# Keep strong references to animation objects during save to avoid GC-related warnings
_ANIM_CACHE: list[animation.FuncAnimation] = []

#: Default plotting specification for sea ice concentration visualisation
DEFAULT_SIC_SPEC = PlotSpec(
    variable="sea_ice_concentration",
    title_groundtruth="Ground Truth",
    title_prediction="Prediction",
    title_difference="Difference",
    colourmap="viridis",
    diff_mode="signed",
    vmin=0.0,
    vmax=1.0,
    colourbar_location="horizontal",
    colourbar_strategy="shared",
    outside_warn=0.05,
    severe_outside=0.20,
    include_shared_range_mismatch_check=True,
)


# --- Static Map Plot ---
def plot_maps(
    plot_spec: PlotSpec,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    date: date | datetime,
) -> dict[str, list[ImageFile]]:
    """Create static maps comparing ground truth and prediction sea ice concentration data.

    Create static maps and (optional) the difference. The plots use contour mapping with customisable colour
    schemes and include proper axis scaling and colourbars.

    Args:
        plot_spec: Configuration object specifying titles, colourmaps, value ranges, and
            other visualisation parameters.
        ground_truth: 2D array of ground truth sea ice concentration values.
        prediction: 2D array of predicted sea ice concentration values. Must have
            the same shape as ground_truth.
        date: Date/datetime for the data being visualised, used in the plot title.

    Returns:
        Dictionary mapping plot names to lists of PIL ImageFile objects. Currently
        returns a single key "sea-ice_concentration-static-maps" containing a list
        with one image representing the generated plot.

    Raises:
        InvalidArrayError: If ground_truth and prediction arrays have incompatible shapes.

    """
    # Check the shapes of the arrays
    height, width = validate_2d_pair(ground_truth, prediction)

    # Load land mask if specified
    land_mask = load_land_mask(plot_spec.land_mask_path, (height, width))

    # Initialise the figure and axes
    fig, axs, cbar_axes = _build_layout(plot_spec=plot_spec, height=height, width=width)
    levels = levels_from_spec(plot_spec)

    # Prepare difference rendering parameters if needed
    diff_colour_scale = None
    if plot_spec.include_difference:
        difference = compute_difference(ground_truth, prediction, plot_spec.diff_mode)
        diff_colour_scale = make_diff_colourmap(difference, mode=plot_spec.diff_mode)

    # Draw the ground truth and prediction map images
    image_groundtruth, image_prediction, image_difference, _ = _draw_frame(
        axs,
        ground_truth,
        prediction,
        plot_spec,
        diff_colour_scale,
        precomputed_difference=difference if plot_spec.include_difference else None,
        levels_override=levels,
        land_mask=land_mask,
    )

    # Colourbars and title
    _add_colourbars(
        axs,
        image_groundtruth=image_groundtruth,
        image_prediction=image_prediction,
        image_difference=image_difference,
        plot_spec=plot_spec,
        cbar_axes=cbar_axes,
    )

    _set_axes_limits(axs, width=width, height=height)
    fig.suptitle(_format_date_to_string(date))

    # Include range_check report
    (groundtruth_min, groundtruth_max), (prediction_min, prediction_max) = (
        compute_display_ranges(ground_truth, prediction, plot_spec)
    )
    range_check_report = compute_range_check_report(
        ground_truth,
        prediction,
        vmin=groundtruth_min,
        vmax=groundtruth_max,
        outside_warn=getattr(plot_spec, "outside_warn", 0.05),
        severe_outside=getattr(plot_spec, "severe_outside", 0.20),
        include_shared_range_mismatch_check=getattr(
            plot_spec, "include_shared_range_mismatch_check", True
        ),
    )
    badge = (
        ""
        if not range_check_report.warnings
        else "Warnings: " + ", ".join(range_check_report.warnings)
    )
    if badge:
        # Place the warning just below the title
        title_artist = getattr(fig, "_suptitle", None)
        if title_artist is not None:
            _, title_y = title_artist.get_position()
            warning_y = max(title_y - 0.03, 0.0)
        else:
            warning_y = 0.93
        fig.text(
            0.5, warning_y, badge, fontsize=9, color="firebrick", ha="center", va="top"
        )

    try:
        return {"sea-ice_concentration-static-maps": [convert._image_from_figure(fig)]}
    finally:
        plt.close(fig)


# --- Video Map Plot ---
def video_maps(
    plot_spec: PlotSpec,
    ground_truth_stream: np.ndarray,
    prediction_stream: np.ndarray,
    dates: Sequence[date | datetime],
    fps: int = 2,
    video_format: Literal["mp4", "gif"] = "gif",
) -> dict[str, io.BytesIO]:
    """Generate animated visualisations showing the temporal evolution of sea ice concentration.

    Compares ground truth data with model predictions. Supports multiple video formats and
    difference computation strategies for optimal performance with large datasets.

    Args:
        plot_spec: Configuration object specifying titles, colourmaps, value ranges, and
            other visualisation parameters.
        ground_truth_stream: 3D array with shape (time, height, width) containing
            ground truth sea ice concentration values over time.
        prediction_stream: 3D array with shape (time, height, width) containing
            predicted sea ice concentration values. Must have the same shape as
            ground_truth_stream.
        dates: List of date/datetime objects corresponding to each timestep.
            Must have the same length as the time dimension of the data arrays.
        include_difference: Whether to include difference visualisation in the animation.
        fps: Frames per second for the output video. Higher values create smoother
            but larger animations.
        video_format: Output video format, either "mp4" or "gif".
        diff_strategy: Strategy for computing differences over time:
            - "precompute": Calculate all differences upfront (memory intensive)
            - "two-pass": Scan data to determine colour scale, compute per-frame
            - "per-frame": Compute differences on-demand (memory efficient)

    Returns:
        Dictionary mapping video names to BytesIO objects containing the encoded video data.
        Currently returns a single key "sea-ice_concentration-video-maps" containing the video
        data suitable for direct wandb.Video logging.

    Raises:
        InvalidArrayError: If arrays have incompatible shapes or if the number of
            dates doesn't match the number of timesteps.
        VideoRenderError: If video encoding fails.

    """
    # Check the shapes of the arrays
    shape = validate_3d_streams(ground_truth_stream, prediction_stream)
    n_timesteps, height, width = shape
    if len(dates) != n_timesteps:
        error_msg = (
            f"Number of dates ({len(dates)}) != number of timesteps ({n_timesteps})"
        )
        raise InvalidArrayError(error_msg)

    # Load land mask if specified
    land_mask = load_land_mask(plot_spec.land_mask_path, (height, width))

    # Apply land mask to data streams if provided
    if land_mask is not None:
        # Mask out land areas by setting them to NaN
        ground_truth_stream = np.where(land_mask, np.nan, ground_truth_stream)
        prediction_stream = np.where(land_mask, np.nan, prediction_stream)

    # Initialise the figure and axes
    fig, axs, cbar_axes = _build_layout(plot_spec=plot_spec, height=height, width=width)
    levels = levels_from_spec(plot_spec)

    # Stable ranges for the whole animation
    display_ranges = compute_display_ranges_stream(
        ground_truth_stream, prediction_stream, plot_spec
    )

    # Prepare the difference array according to the strategy; also ensure a colour scale exists
    difference_stream, diff_colour_scale = prepare_difference_stream(
        include_difference=plot_spec.include_difference,
        diff_mode=plot_spec.diff_mode,
        strategy=plot_spec.diff_strategy,
        ground_truth_stream=ground_truth_stream,
        prediction_stream=prediction_stream,
    )
    if plot_spec.include_difference and diff_colour_scale is None:
        # Per-frame strategy: infer a stable colour scale from the first frame to avoid breathing
        first_diff = compute_difference(
            ground_truth_stream[0], prediction_stream[0], plot_spec.diff_mode
        )
        diff_colour_scale = make_diff_colourmap(first_diff, mode=plot_spec.diff_mode)
    precomputed_diff_0 = (
        difference_stream[0]
        if (plot_spec.include_difference and difference_stream is not None)
        else None
    )

    # Initial frame
    image_groundtruth, image_prediction, image_difference, _ = _draw_frame(
        axs,
        ground_truth_stream[0],
        prediction_stream[0],
        plot_spec,
        diff_colour_scale,
        precomputed_diff_0,
        levels,
        display_ranges_override=display_ranges,
        land_mask=land_mask,
    )
    # Colourbars and title
    _add_colourbars(
        axs,
        image_groundtruth=image_groundtruth,
        image_prediction=image_prediction,
        image_difference=image_difference,
        plot_spec=plot_spec,
        diff_colour_scale=diff_colour_scale,
        cbar_axes=cbar_axes,
    )
    _set_axes_limits(axs, width=width, height=height)
    fig.suptitle(_format_date_to_string(dates[0]))

    # Animation function
    def animate(tt: int) -> tuple[()]:
        precomputed_diff_tt = (
            difference_stream[tt]
            if (plot_spec.include_difference and difference_stream is not None)
            else None
        )
        _draw_frame(
            axs,
            ground_truth_stream[tt],
            prediction_stream[tt],
            plot_spec,
            diff_colour_scale,
            precomputed_diff_tt,
            levels,
            display_ranges_override=display_ranges,
            land_mask=land_mask,
        )

        fig.suptitle(_format_date_to_string(dates[tt]))
        return ()

    # Create the animation object
    animation_object = animation.FuncAnimation(
        fig,
        animate,
        frames=n_timesteps,
        interval=1000 // fps,
        blit=False,
        repeat=True,
    )
    # Keep a strong reference without touching figure attributes (ruff-fix)
    _ANIM_CACHE.append(animation_object)

    # Save -> BytesIO and clean up temp file
    try:
        video_buffer = convert._save_animation(
            animation_object, _fps=fps, _video_format=video_format
        )
        return {"sea-ice_concentration-video-maps": video_buffer}
    finally:
        # Drop strong reference now that saving is done
        with contextlib.suppress(ValueError):
            _ANIM_CACHE.remove(animation_object)
        plt.close(fig)


# ---- Helper functions ----


def _apply_land_mask(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    difference: np.ndarray | None,
    land_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Apply land mask to data arrays by setting land areas to NaN.

    Args:
        ground_truth: Ground truth data array.
        prediction: Prediction data array.
        difference: Optional difference data array.
        land_mask: Optional land mask where True indicates land areas.

    Returns:
        Tuple of (masked_ground_truth, masked_prediction, masked_difference).

    """
    if land_mask is not None:
        ground_truth = np.where(land_mask, np.nan, ground_truth)
        prediction = np.where(land_mask, np.nan, prediction)
        if difference is not None:
            difference = np.where(land_mask, np.nan, difference)

    return ground_truth, prediction, difference


def _compute_display_ranges(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    plot_spec: PlotSpec,
    display_ranges_override: tuple[tuple[float, float], tuple[float, float]]
    | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute display ranges for ground truth and prediction plots.

    Args:
        ground_truth: Ground truth data array.
        prediction: Prediction data array.
        plot_spec: Plotting specification.
        display_ranges_override: Optional override ranges for stable animation.

    Returns:
        Tuple of ((gt_vmin, gt_vmax), (pred_vmin, pred_vmax)).

    """
    if display_ranges_override is not None:
        return display_ranges_override
    return compute_display_ranges(ground_truth, prediction, plot_spec)


def _draw_main_panels(  # noqa: PLR0913
    axs: list,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    plot_spec: PlotSpec,
    groundtruth_vmin: float,
    groundtruth_vmax: float,
    prediction_vmin: float,
    prediction_vmax: float,
    levels_override: np.ndarray | None = None,
) -> tuple:
    """Draw ground truth and prediction panels.

    Args:
        axs: List of matplotlib axes objects.
        ground_truth: Ground truth data array.
        prediction: Prediction data array.
        plot_spec: Plotting specification.
        groundtruth_vmin: Minimum value for ground truth display.
        groundtruth_vmax: Maximum value for ground truth display.
        prediction_vmin: Minimum value for prediction display.
        prediction_vmax: Maximum value for prediction display.
        levels_override: Optional custom contour levels.

    Returns:
        Tuple of (image_groundtruth, image_prediction).

    """
    # Use PlotSpec levels for ground_truth and prediction unless overridden
    levels = levels_from_spec(plot_spec) if levels_override is None else levels_override

    if plot_spec.colourbar_strategy == "separate":
        # For separate strategy, use explicit levels to prevent breathing
        groundtruth_levels = np.linspace(
            groundtruth_vmin, groundtruth_vmax, plot_spec.n_contour_levels
        )
        prediction_levels = np.linspace(
            prediction_vmin, prediction_vmax, plot_spec.n_contour_levels
        )

        image_groundtruth = axs[0].contourf(
            ground_truth,
            levels=groundtruth_levels,
            cmap=plot_spec.colourmap,
            vmin=groundtruth_vmin,
            vmax=groundtruth_vmax,
        )
        image_prediction = axs[1].contourf(
            prediction,
            levels=prediction_levels,
            cmap=plot_spec.colourmap,
            vmin=prediction_vmin,
            vmax=prediction_vmax,
        )
    else:
        # For shared strategy, use same levels for both panels
        image_groundtruth = axs[0].contourf(
            ground_truth,
            levels=levels,
            cmap=plot_spec.colourmap,
            vmin=groundtruth_vmin,
            vmax=groundtruth_vmax,
        )
        image_prediction = axs[1].contourf(
            prediction,
            levels=levels,
            cmap=plot_spec.colourmap,
            vmin=prediction_vmin,
            vmax=prediction_vmax,
        )

    return image_groundtruth, image_prediction


def _draw_frame(  # noqa: PLR0913
    axs: list,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    plot_spec: PlotSpec,
    diff_colour_scale: DiffColourmapSpec | None = None,
    precomputed_difference: np.ndarray | None = None,
    levels_override: np.ndarray | None = None,
    display_ranges_override: tuple[tuple[float, float], tuple[float, float]]
    | None = None,
    land_mask: np.ndarray | None = None,
) -> tuple:
    """Draw a complete visualisation frame with ground truth, prediction, and optional difference.

    Creates contour plots for ground truth and prediction data, and optionally computes
    and displays their difference. It handles colour normalisation, contour levels, and
    proper cleanup of previous plot elements. Can overlay grey land areas using a land mask.

    Args:
        axs: List of matplotlib axes objects where plots will be drawn. Expected to have
            at least 2 axes (ground truth, prediction) and optionally 3 (with difference).
        ground_truth: 2D array of ground truth values to plot.
        prediction: 2D array of predicted values to plot.
        plot_spec: Plotting specification containing colourmap, value ranges, and other
            display parameters.
        diff_colour_scale: Optional DiffColourmapSpec containing normalisation and colour
            mapping parameters for difference plots.
        precomputed_difference: Optional pre-computed difference array. If None and
            difference is included, the difference will be computed on-demand.
        levels_override: Optional custom contour levels. If None, levels are derived
            from plot_spec.
        display_ranges_override: Optional custom display ranges for stable animation.
            If None, ranges are computed from the data.
        land_mask: Optional 2D boolean array where True indicates land areas to overlay
            in grey. If None, no land overlay is applied.

    Returns:
        Tuple containing (image_groundtruth, image_prediction, image_difference, diff_colour_scale).
        The image objects are matplotlib contour collections, and image_difference may be None
        if include_difference is False. diff_colour_scale is returned for reuse in animations.

    """
    for ax in axs:
        _clear_contours(ax)

    # Apply land mask to data if provided
    ground_truth, prediction, difference = _apply_land_mask(
        ground_truth, prediction, precomputed_difference, land_mask
    )

    # Compute display ranges - use override if provided for stable animation
    (groundtruth_vmin, groundtruth_vmax), (prediction_vmin, prediction_vmax) = (
        _compute_display_ranges(
            ground_truth, prediction, plot_spec, display_ranges_override
        )
    )

    # Draw ground truth and prediction panels
    image_groundtruth, image_prediction = _draw_main_panels(
        axs,
        ground_truth,
        prediction,
        plot_spec,
        groundtruth_vmin,
        groundtruth_vmax,
        prediction_vmin,
        prediction_vmax,
        levels_override,
    )

    image_difference = None
    if plot_spec.include_difference:
        # Use the difference from land mask application, or compute if not available
        if difference is None:
            difference = compute_difference(
                ground_truth, prediction, plot_spec.diff_mode
            )

        # Expect colour scale to be provided by caller; no fallback here
        if diff_colour_scale is None:
            error_msg = "diff_colour_scale must be provided when including difference"
            raise InvalidArrayError(error_msg)

        if diff_colour_scale.norm is not None:
            # Signed differences with TwoSlopeNorm - use explicit levels to ensure consistency
            diff_vmin = diff_colour_scale.norm.vmin or 0.0
            diff_vmax = diff_colour_scale.norm.vmax or 1.0
            diff_levels = np.linspace(diff_vmin, diff_vmax, plot_spec.n_contour_levels)

            image_difference = axs[2].contourf(
                difference,
                levels=diff_levels,
                cmap=diff_colour_scale.cmap,
                vmin=diff_vmin,
                vmax=diff_vmax,
            )
        else:
            # Non-negative differences with vmin/vmax
            vmin = diff_colour_scale.vmin or 0.0
            vmax = diff_colour_scale.vmax or 1.0
            diff_levels = np.linspace(
                vmin,
                vmax,
                plot_spec.n_contour_levels,
            )

            image_difference = axs[2].contourf(
                difference,
                levels=diff_levels,
                cmap=diff_colour_scale.cmap,
                vmin=vmin,
                vmax=vmax,
            )

    # Optional: visually mark NaNs as semi-transparent grey overlays

    def _overlay_nans(ax: Axes, arr: np.ndarray) -> None:
        """Overlay NaNs as semi-transparent grey overlays.

        Not used currently, add the following before the return statement to enable:
        >>> _overlay_nans(axs[0], ground_truth)
        >>> _overlay_nans(axs[1], prediction)
        >>> if plot_spec.include_difference:
        >>>     _overlay_nans(axs[2], difference)
        Mask should then be visible in the plot.

        """
        if np.isnan(arr).any():
            # Create overlay for NaN areas (land mask)
            nan_mask = np.isnan(arr).astype(float)
            # Create a custom colormap: 0=transparent, 1=land color

            # Land colour options: 'white' for white land, 'grey' for grey land
            colors = ["white", "white"]  # 0=white (transparent), 1=land color
            cmap = ListedColormap(colors)
            ax.imshow(
                nan_mask,
                cmap=cmap,
                vmin=0,
                vmax=1,
                alpha=1.0,  # Fully opaque white land areas
                interpolation="nearest",
            )

    # Optional: visually mark NaNs as semi-transparent grey overlays here
    if land_mask is not None:
        _overlay_nans(axs[0], ground_truth)
        _overlay_nans(axs[1], prediction)
        if plot_spec.include_difference and difference is not None:
            _overlay_nans(axs[2], difference)

    return image_groundtruth, image_prediction, image_difference, diff_colour_scale


def _format_date_to_string(date: date | datetime) -> str:
    """Format a date or datetime object to a standardised string representation for plot titles.

    Args:
        date: Date or datetime object to format.

    Returns:
        Formatted string in "YYYY-MM-DD HH:MM" format for datetime objects,
        or "YYYY-MM-DD" format for date objects.

    Example:
        >>> from datetime import date, datetime
        >>> _format_date_to_string(date(2023, 12, 25))
        '2023-12-25'
        >>> _format_date_to_string(datetime(2023, 12, 25, 14, 30))
        '2023-12-25 14:30'

    """
    return (
        date.strftime(r"%Y-%m-%d %H:%M")
        if isinstance(date, datetime)
        else date.strftime(r"%Y-%m-%d")
    )


def _clear_contours(ax: Axes) -> None:
    """Remove all contour collections from a matplotlib axes object to prevent overlapping plots.

    Prevents overlapping plots in an animation loop.

    Args:
        ax: Matplotlib axes object to clear.

    """
    for coll in ax.collections[:]:
        coll.remove()
