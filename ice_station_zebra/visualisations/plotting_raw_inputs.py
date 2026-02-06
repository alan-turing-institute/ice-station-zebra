"""Raw input variable visualisation for a single timestep or animation over time.

Creates one static map per input channel with optional per-variable style overrides,
or animations showing temporal evolution of individual variables.

This module is intentionally lightweight and single-panel oriented while reusing the
layout and formatting helpers from the sea-ice plotting stack to keep appearances aligned.
"""

import logging
from datetime import date, datetime
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import TwoSlopeNorm

from ice_station_zebra.exceptions import InvalidArrayError
from ice_station_zebra.types import PlotSpec
from ice_station_zebra.visualisations.animation_helper import hold_anim, release_anim
from ice_station_zebra.visualisations.layout import (
    build_single_panel_figure,
    format_linear_ticks,
    format_symmetric_ticks,
    set_suptitle_with_box,
)
from ice_station_zebra.visualisations.plotting_core import (
    colourmap_with_bad,
    create_normalisation,
    load_land_mask,
    safe_filename,
    save_figure,
    style_for_variable,
)

if TYPE_CHECKING:
    from matplotlib.text import Text
import io
from collections.abc import Sequence
from pathlib import Path

from PIL.ImageFile import ImageFile

logger = logging.getLogger(__name__)


def _format_title(
    variable: str, hemisphere: str | None, when: date | datetime, units: str | None
) -> str:
    """Format a title string for a raw input variable plot.

    Args:
        variable: Variable name.
        hemisphere: Hemisphere ("north" or "south"), if applicable.
        when: Date or datetime of the data.
        units: Display units for the variable.

    Returns:
        Formatted title string.

    """
    hemi = f" ({hemisphere.capitalize()})" if hemisphere else ""
    units_s = f" [{units}]" if units else ""
    shown = when.date().isoformat() if isinstance(when, datetime) else when.isoformat()
    return f"{variable}{units_s}{hemi}   Shown: {shown}"


def plot_raw_inputs_for_timestep(  # noqa: C901, PLR0912, PLR0915
    *,
    channels: dict[str, np.ndarray],
    when: date | datetime,
    plot_spec_base: PlotSpec,
    land_mask: np.ndarray | None = None,
    save_dir: Path | None = None,
) -> dict[str, list[ImageFile]]:
    """Plot one image per input channel as a static map.

    Returns list of (name, PIL.ImageFile, saved_path|None).
    """
    channel_arrays = list(channels.values())
    channel_names = list(channels.keys())
    styles = plot_spec_base.per_variable_styles
    results: dict[str, list[ImageFile]] = {}
    land_mask_cache: dict[tuple[int, int], np.ndarray | None] = {}

    from . import convert  # noqa: PLC0415  # local import to avoid circulars

    expected_ndim = 2
    for arr, name in zip(channel_arrays, channel_names, strict=True):
        if arr.ndim != expected_ndim:
            msg = f"Expected 2D [H,W] for channel '{name}', got {arr.shape}"
            raise InvalidArrayError(msg)

        # Apply land mask (mask out land to NaN)
        active_mask = land_mask
        if active_mask is None and plot_spec_base.land_mask_path:
            shape = arr.shape
            if shape not in land_mask_cache:
                try:
                    land_mask_cache[shape] = load_land_mask(
                        plot_spec_base.land_mask_path, shape
                    )
                except InvalidArrayError:
                    logger.exception(
                        "Failed to load land mask for '%s' with shape %s", name, shape
                    )
                    land_mask_cache[shape] = None
            active_mask = land_mask_cache.get(shape)

        # Apply mask if available, creating a new variable to avoid overwriting loop variable
        arr_to_plot = arr
        if active_mask is not None:
            if active_mask.shape == arr.shape:
                arr_to_plot = np.where(active_mask, np.nan, arr)
            else:
                logger.debug(
                    "Skipping land mask for '%s': mask shape %s != array shape %s",
                    name,
                    active_mask.shape,
                    arr.shape,
                )

        # Prefer an explicit styles dict, otherwise fall back to the PlotSpec attribute if present
        style = style_for_variable(name, styles)
        logger.debug(
            "plot_raw_inputs: resolved style for %r => cmap=%r, vmin=%r, vmax=%r, origin=%r",
            name,
            style.cmap,
            style.vmin,
            style.vmax,
            style.origin,
        )

        # Create normalisation using shared function
        norm, vmin, vmax = create_normalisation(
            arr_to_plot,
            vmin=style.vmin,
            vmax=style.vmax,
            centre=style.two_slope_centre,
        )

        # Build figure and axis
        height, width = arr_to_plot.shape
        fig, ax, cax = build_single_panel_figure(
            height=height,
            width=width,
            colourbar_location=plot_spec_base.colourbar_location,
        )

        # Render
        cmap_name = style.cmap or plot_spec_base.colourmap
        cmap = colourmap_with_bad(cmap_name, bad_color="lightgrey")
        origin = style.origin or "lower"
        image = ax.imshow(
            arr_to_plot, cmap=cmap, norm=norm, origin=origin, interpolation="nearest"
        )

        # Colourbar
        orientation = plot_spec_base.colourbar_location
        cbar = fig.colorbar(image, ax=ax, cax=cax, orientation=orientation)
        is_vertical = orientation == "vertical"
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

        # Title
        try:
            set_suptitle_with_box(
                fig,
                _format_title(name, plot_spec_base.hemisphere, when, style.units),
            )
        except (ValueError, AttributeError, RuntimeError) as err:
            logger.debug(
                "Failed to draw raw-inputs title: %s; continuing without title.", err
            )

        # Save to disk if requested (colons in variable names replaced)
        file_base = name.replace(":", "__")
        save_figure(fig, save_dir, file_base)

        # Convert to PIL
        try:
            pil_img = convert.image_from_figure(fig)
        finally:
            plt.close(fig)

        # Add image to results dict
        if name not in results:
            results[name] = []
        results[name].append(pil_img)

    return results


def video_raw_input_for_variable(  # noqa: PLR0915
    *,
    variable_name: str,
    data_stream: np.ndarray,
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
        data_stream: 3D array of data over time [T, H, W].
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
    if data_stream.ndim != 3:  # noqa: PLR2004
        msg = f"Expected 3D data [T,H,W], got shape {data_stream.shape}"
        raise InvalidArrayError(msg)

    n_timesteps, height, width = data_stream.shape
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
            data_stream = np.where(land_mask, np.nan, data_stream)

    # Get styling for this variable
    style = style_for_variable(variable_name, plot_spec.per_variable_styles)

    # Create stable normalisation across all frames
    # Use global min/max to prevent colour scale "breathing"
    data_min = float(np.nanmin(data_stream)) if np.isfinite(data_stream).any() else 0.0
    data_max = float(np.nanmax(data_stream)) if np.isfinite(data_stream).any() else 1.0

    # Override with style limits if provided
    effective_vmin = style.vmin if style.vmin is not None else data_min
    effective_vmax = style.vmax if style.vmax is not None else data_max

    # Create normalisation using first frame to establish type
    norm, vmin, vmax = create_normalisation(
        data_stream[0],
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
        data_stream[0], cmap=cmap, norm=norm, origin=origin, interpolation="nearest"
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
        image.set_data(data_stream[tt])

        # Update title with current date
        if title_text is not None:
            title_text.set_text(
                _format_title(
                    variable_name, plot_spec.hemisphere, dates[tt], style.units
                )
            )

        return ()

    # Create animation object
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=n_timesteps,
        interval=1000 // plot_spec.video_fps,
        blit=False,
        repeat=True,
    )

    # Keep strong reference to prevent garbage collection during save
    hold_anim(anim)

    try:
        # Save to BytesIO buffer
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

        return video_buffer

    finally:
        # Clean up: remove from cache and close figure
        release_anim(anim)
        plt.close(fig)


def video_raw_inputs_for_timesteps(
    *,
    channels: dict[str, np.ndarray],
    dates: Sequence[date | datetime],
    plot_spec: PlotSpec,
    land_mask: np.ndarray | None = None,
    save_dir: Path | None = None,
) -> dict[str, io.BytesIO]:  # list[tuple[str, io.BytesIO, Path | None]]:
    """Create animations for multiple input variables over time.

    This is a convenience wrapper around `video_raw_input_for_variable()` that
    processes multiple variables in a batch, applying consistent styling and
    land masking to all variables.

    Args:
        channels: Dictionary of variable name to THW 3D array of values.
        dates: Sequence of dates for each timestep (length must match T).
        plot_spec: Plotting specification.
        land_mask: Optional 2D land mask to apply to all variables.
        save_dir: Optional directory to save videos to disk.

    Returns:
        List of tuples (variable_name, video_buffer, saved_path).

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
            video_buffer = video_raw_input_for_variable(
                variable_name=var_name,
                data_stream=data_stream,
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
