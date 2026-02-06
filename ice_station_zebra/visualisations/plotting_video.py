"""Raw input variable visualisation for a single timestep or animation over time.

Creates one static map per input channel with optional per-variable style overrides,
or animations showing temporal evolution of individual variables.

This module is intentionally lightweight and single-panel oriented while reusing the
layout and formatting helpers from the sea-ice plotting stack to keep appearances aligned.
"""

import logging
from datetime import date, datetime
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import TwoSlopeNorm

from ice_station_zebra.exceptions import InvalidArrayError
from ice_station_zebra.types import PlotSpec
from ice_station_zebra.visualisations.layout import (
    build_single_panel_figure,
    format_linear_ticks,
    format_symmetric_ticks,
    set_suptitle_with_box,
)
from ice_station_zebra.visualisations.plotting_core import (
    colourmap_with_bad,
    compute_difference,
    compute_display_ranges_stream,
    create_normalisation,
    levels_from_spec,
    load_land_mask,
    make_diff_colourmap,
    prepare_difference_stream,
    safe_filename,
    style_for_variable,
)

from .convert import save_animation
from .helpers import (
    _build_footer_video,
    _build_title_video,
    _draw_frame,
    _format_title,
)
from .layout import (
    LayoutConfig,
    TitleFooterConfig,
    _add_colourbars,
    _set_axes_limits,
    _set_titles,
    build_layout,
    set_footer_with_box,
)

if TYPE_CHECKING:
    from matplotlib.text import Text
import io
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)


# --- Video Map Plot ---
def plot_video_prediction(  # noqa: C901
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
    if ground_truth_stream.shape != prediction_stream.shape:
        msg = f"Prediction ({prediction_stream.shape}) has a different shape to ground truth ({ground_truth_stream.shape})."
        raise InvalidArrayError(msg)
    n_timesteps, height, width = ground_truth_stream.shape
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

    # Initialise the figure and axes with a larger footer space for videos
    layout_config = LayoutConfig(title_footer=TitleFooterConfig(footer_space=0.11))
    fig, axs, cbar_axes = build_layout(
        plot_spec=plot_spec,
        height=height,
        width=width,
        layout_config=layout_config,
    )
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
    # Restore axis titles after drawing (they were cleared in _draw_frame)
    _set_titles(axs, plot_spec)
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
    try:
        title_text = set_suptitle_with_box(fig, _build_title_video(plot_spec, dates, 0))
    except Exception:
        logger.exception("Failed to draw suptitle; continuing without title.")
        title_text = None

    # Footer metadata at the bottom (static across frames)
    if getattr(plot_spec, "include_footer_metadata", True):
        try:
            footer_text = _build_footer_video(plot_spec, dates)
            if footer_text:
                set_footer_with_box(fig, footer_text)
        except Exception:
            logger.exception("Failed to draw footer; continuing without footer.")

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
        # Restore axis titles after drawing (they were cleared in _draw_frame)
        _set_titles(axs, plot_spec)

        if title_text is not None:
            title_text.set_text(_build_title_video(plot_spec, dates, tt))
        return ()

    # Save -> BytesIO and clean up temp file
    try:
        # Create the animation object
        animation_object = animation.FuncAnimation(
            fig,
            animate,
            frames=n_timesteps,
            interval=1000 // fps,
            blit=False,
            repeat=True,
        )
        # Write to BytesIO buffer
        video_buffer = save_animation(
            animation_object, fps=fps, video_format=video_format
        )
        # Return the video buffer
        return {"sea-ice_concentration-video-maps": video_buffer}
    finally:
        # Clean up by closing figure
        plt.close(fig)


def plot_video_single_input(  # noqa: PLR0915
    *,
    variable_name: str,
    np_array_thw: np.ndarray,
    dates: Sequence[date | datetime],
    plot_spec: PlotSpec,
    land_mask: np.ndarray | None = None,
    save_path: Path | None = None,
) -> io.BytesIO:
    """Create animation showing temporal evolution of a single variable.

    This function creates a video animation of a single input variable over time,
    with consistent colour scaling across all frames to avoid visual "breathing".

    Args:
        variable_name: Name of the variable (e.g., "era5:2t", "osisaf-south:ice_conc").
        np_array_thw: 3D array of data over time [T, H, W].
        dates: Sequence of dates corresponding to each timestep (length must match T).
        plot_spec: Base plotting specification (colourmap, hemisphere, etc.).
        land_mask: Optional 2D boolean array marking land areas [H, W].
        style: Optional pre-computed VariableStyle for this variable.
        styles: Optional dictionary of style configurations for matching.
        save_path: Optional path to save the video file to disk.

    Returns:
        BytesIO buffer containing the encoded video data.

    Raises:
        InvalidArrayError: If data_stream is not 3D or dates length mismatches.
        VideoRenderError: If video encoding fails.

    """
    from . import convert  # noqa: PLC0415  # Local import to avoid circulars

    # Validate input
    if np_array_thw.ndim != 3:  # noqa: PLR2004
        msg = f"Expected 3D data [T,H,W], got shape {np_array_thw.shape}"
        raise InvalidArrayError(msg)

    n_timesteps, height, width = np_array_thw.shape
    if len(dates) != n_timesteps:
        msg = f"Number of dates ({len(dates)}) != number of timesteps ({n_timesteps})"
        raise InvalidArrayError(msg)

    # Apply land mask if provided
    if land_mask is not None:
        if land_mask.shape != (height, width):
            logger.debug(
                "Land mask shape %s doesn't match data shape (%d, %d), skipping mask",
                land_mask.shape,
                height,
                width,
            )
        else:
            # Mask out land areas by setting them to NaN
            np_array_thw = np.where(land_mask, np.nan, np_array_thw)

    # Get styling for this variable
    style = style_for_variable(variable_name, plot_spec.per_variable_styles)

    # Create stable normalisation across all frames
    # Use global min/max to prevent colour scale "breathing"
    data_min = (
        float(np.nanmin(np_array_thw)) if np.isfinite(np_array_thw).any() else 0.0
    )
    data_max = (
        float(np.nanmax(np_array_thw)) if np.isfinite(np_array_thw).any() else 1.0
    )

    # Override with style limits if provided
    effective_vmin = style.vmin if style.vmin is not None else data_min
    effective_vmax = style.vmax if style.vmax is not None else data_max

    # Create normalisation using first frame to establish type
    norm, vmin, vmax = create_normalisation(
        np_array_thw[0],
        vmin=effective_vmin,
        vmax=effective_vmax,
        centre=style.two_slope_centre,
    )

    # Build figure with single panel + colourbar
    fig, ax, cax = build_single_panel_figure(
        height=height,
        width=width,
        colourbar_location=plot_spec.colourbar_location,
    )

    # Render initial frame
    cmap_name = style.cmap or plot_spec.colourmap
    cmap = colourmap_with_bad(cmap_name, bad_color="lightgrey")
    origin = style.origin or "lower"
    image = ax.imshow(
        np_array_thw[0], cmap=cmap, norm=norm, origin=origin, interpolation="nearest"
    )

    # Create colourbar (stable across all frames)
    orientation = plot_spec.colourbar_location
    cbar = fig.colorbar(image, ax=ax, cax=cax, orientation=orientation)
    is_vertical = orientation == "vertical"

    # Format colourbar ticks based on normalisation type
    decimals = style.decimals if style.decimals is not None else 2
    use_scientific = (
        style.use_scientific_notation
        if style.use_scientific_notation is not None
        else False
    )
    if isinstance(norm, TwoSlopeNorm):
        format_symmetric_ticks(
            cbar,
            vmin=vmin,
            vmax=vmax,
            decimals=decimals,
            is_vertical=is_vertical,
            centre=norm.vcenter,
            use_scientific_notation=use_scientific,
        )
    else:
        format_linear_ticks(
            cbar,
            vmin=vmin,
            vmax=vmax,
            decimals=decimals,
            is_vertical=is_vertical,
            use_scientific_notation=use_scientific,
        )

    # Create title (will be updated each frame)
    title_text: Text | None = None
    try:
        title_text = set_suptitle_with_box(
            fig,
            _format_title(variable_name, plot_spec.hemisphere, dates[0], style.units),
        )
    except (ValueError, AttributeError, RuntimeError) as err:
        logger.debug("Failed to draw title: %s; continuing without title.", err)

    # Animation update function
    def animate(tt: int) -> tuple[()]:
        """Update function for each frame of the animation."""
        # Update image data
        image.set_data(np_array_thw[tt])
        # Update title with current date
        if title_text is not None:
            title_text.set_text(
                _format_title(
                    variable_name, plot_spec.hemisphere, dates[tt], style.units
                )
            )
        return ()

    try:
        # Create animation object
        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=n_timesteps,
            interval=1000 // plot_spec.video_fps,
            blit=False,
            repeat=True,
        )
        # Write to BytesIO buffer
        video_buffer = convert.save_animation(
            anim, fps=plot_spec.video_fps, video_format=plot_spec.video_format
        )
        # Optionally save to disk
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(video_buffer.getvalue())
            logger.debug("Saved animation to %s", save_path)
            # Reset buffer position after writing
            video_buffer.seek(0)
        # Return the video buffer
        return video_buffer
    finally:
        # Clean up by closing figure
        plt.close(fig)


def plot_video_inputs(
    *,
    channels: dict[str, np.ndarray],
    dates: Sequence[date | datetime],
    plot_spec: PlotSpec,
    land_mask: np.ndarray | None = None,
    save_dir: Path | None = None,
) -> dict[str, io.BytesIO]:
    """Create animations for multiple input variables over time.

    This is a convenience wrapper around `plot_video_single_input()` that
    processes multiple variables in a batch, applying consistent styling and
    land masking to all variables.

    Args:
        channels: Dictionary of variable name to THW 3D array of values.
        dates: Sequence of dates for each timestep (length must match T).
        plot_spec: Plotting specification.
        land_mask: Optional 2D land mask to apply to all variables.
        save_dir: Optional directory to save videos to disk.

    Returns:
        Dictionary mapping variable_name to video_buffer.

    Raises:
        InvalidArrayError: If arrays/names count mismatch or invalid shapes.

    """
    channel_names = list(channels.keys())
    channel_arrays_stream = list(channels.values())

    results: dict[str, io.BytesIO] = {}

    for data_stream, var_name in zip(channel_arrays_stream, channel_names, strict=True):
        logger.debug("Creating animation for variable: %s", var_name)

        # Determine save path if save_dir is provided
        save_path: Path | None = None
        if save_dir is not None:
            # Sanitise variable name for filename
            file_base = var_name.replace(":", "__")
            suffix = ".mp4" if plot_spec.video_format == "mp4" else ".gif"
            save_path = save_dir / f"{safe_filename(file_base)}{suffix}"

        try:
            # Create animation for this variable
            video_buffer = plot_video_single_input(
                variable_name=var_name,
                np_array_thw=data_stream,
                dates=dates,
                plot_spec=plot_spec,
                land_mask=land_mask,
                save_path=save_path,
            )

            results[var_name] = video_buffer

        except (InvalidArrayError, ValueError, MemoryError, OSError):
            logger.exception(
                "Failed to create animation for variable %s, skipping", var_name
            )
            continue

    return results
