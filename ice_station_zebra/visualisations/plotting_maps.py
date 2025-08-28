"""Sea ice concentration visualisation module for creating static maps and animations.

- Static map plotting with optional difference visualisation
- Animated video generation (MP4/GIF formats)
- Flexible plotting specifications and colour schemes
- Multiple difference computation strategies
- Date formatting

"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

if TYPE_CHECKING:
    import io
    from collections.abc import Sequence
    from typing import Any

    from PIL.ImageFile import ImageFile

from ice_station_zebra.exceptions import InvalidArrayError

from .convert import _image_from_figure, _save_animation
from .layout import _add_colourbars, _init_axes, _set_axes_limits
from .plotting_core import (
    DiffColourmapSpec,
    PlotSpec,
    compute_difference,
    compute_display_ranges,
    compute_display_ranges_stream,
    levels_from_spec,
    make_diff_colourmap,
    prepare_difference_stream,
    validate_2d_pair,
    validate_3d_streams,
)

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
)


# ---- Main functions ----


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

    # Initialise the figure and axes
    fig, axs, cbar_axes = _init_axes(plot_spec=plot_spec, height=height, width=width)
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

    try:
        return {"sea-ice_concentration-static-maps": [_image_from_figure(fig)]}
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

    # Initialise the figure and axes
    fig, axs, cbar_axes = _init_axes(plot_spec=plot_spec, height=height, width=width)
    levels = levels_from_spec(plot_spec)

    # Stable ranges for the whole animation
    display_ranges = compute_display_ranges_stream(
        ground_truth_stream, prediction_stream, plot_spec
    )

    # Prepare the difference array according to the strategy
    difference_stream, diff_colour_scale = prepare_difference_stream(
        include_difference=plot_spec.include_difference,
        diff_mode=plot_spec.diff_mode,
        strategy=plot_spec.diff_strategy,
        ground_truth_stream=ground_truth_stream,
        prediction_stream=prediction_stream,
    )
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

    # Save -> BytesIO and clean up temp file
    try:
        video_buffer = _save_animation(
            animation_object, fps=fps, video_format=video_format
        )
        return {"sea-ice_concentration-video-maps": video_buffer}
    finally:
        plt.close(fig)


# ---- Helper functions ----


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
) -> tuple:
    """Draw a complete visualisation frame with ground truth, prediction, and optional difference.

    Creates contour plots for ground truth and prediction data, and optionally computes
    and displays their difference. It handles colour normalisation, contour levels, and
    proper cleanup of previous plot elements.

    Args:
        axs: List of matplotlib axes objects where plots will be drawn. Expected to have
            at least 2 axes (ground truth, prediction) and optionally 3 (with difference).
        ground_truth: 2D array of ground truth values to plot.
        prediction: 2D array of predicted values to plot.
        plot_spec: Plotting specification containing colourmap, value ranges, and other
            display parameters.
        include_difference: Whether to compute and plot the difference between ground
            truth and prediction.
        diff_colour_scale: Optional DiffColourmapSpec containing normalisation and colour
            mapping parameters for difference plots.
        precomputed_difference: Optional pre-computed difference array. If None and
            difference is included, the difference will be computed on-demand.
        levels_override: Optional custom contour levels. If None, levels are derived
            from plot_spec.
        display_ranges_override: Optional custom display ranges for stable animation.
            If None, ranges are computed from the data.

    Returns:
        Tuple containing (image_groundtruth, image_prediction, image_difference, diff_colour_scale).
        The image objects are matplotlib contour collections, and image_difference may be None
        if include_difference is False. diff_colour_scale is returned for reuse in animations.

    """
    for ax in axs:
        _clear_contours(ax)

    # Compute display ranges - use override if provided for stable animation
    if display_ranges_override is not None:
        (groundtruth_vmin, groundtruth_vmax), (prediction_vmin, prediction_vmax) = (
            display_ranges_override
        )
    else:
        (groundtruth_vmin, groundtruth_vmax), (prediction_vmin, prediction_vmax) = (
            compute_display_ranges(ground_truth, prediction, plot_spec)
        )

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

    image_difference = None
    if plot_spec.include_difference:
        difference = (
            precomputed_difference
            if precomputed_difference is not None
            else compute_difference(ground_truth, prediction, plot_spec.diff_mode)
        )

        # DiffRender system tells us how to colour the difference panel
        if diff_colour_scale is None:
            # Fallback: compute render params on-demand (per-frame strategy)
            diff_colour_scale = make_diff_colourmap(
                difference, mode=plot_spec.diff_mode
            )

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


def _clear_contours(ax: Any) -> None:  # noqa: ANN401
    """Remove all contour collections from a matplotlib axes object to prevent overlapping plots.

    Prevents overlapping plots in an animation loop.

    Args:
        ax: Matplotlib axes object to clear.

    """
    for coll in ax.collections[:]:
        coll.remove()
