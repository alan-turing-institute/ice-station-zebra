"""Sea ice concentration visualisation module for creating static maps and animations.

- Static map plotting with optional difference visualisation
- Animated video generation (MP4/GIF formats)
- Flexible plotting specifications and colour schemes
- Multiple difference computation strategies
- Date formatting

"""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from PIL.ImageFile import ImageFile

from ice_station_zebra.exceptions import InvalidArrayError
from ice_station_zebra.types import PlotSpec

from . import convert
from .helpers import (
    _build_title_static,
    _draw_frame,
    _draw_warning_badge,
    _format_title,
    _maybe_add_footer,
    _prepare_difference,
    _prepare_static_plot,
)
from .layout import (
    _add_colourbars,
    _set_axes_limits,
    _set_titles,
    build_layout,
    build_single_panel_figure,
    format_linear_ticks,
    format_symmetric_ticks,
    set_suptitle_with_box,
)
from .plotting_core import (
    colourmap_with_bad,
    create_normalisation,
    load_land_mask,
    save_figure,
    style_for_variable,
)

logger = logging.getLogger(__name__)


def plot_static_prediction(
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
    (
        height,
        width,
        land_mask,
        layout_config,
        warnings,
        levels,
    ) = _prepare_static_plot(plot_spec, ground_truth, prediction)

    # Initialise the figure and axes with dynamic top spacing if needed
    fig, axs, cbar_axes = build_layout(
        plot_spec=plot_spec,
        height=height,
        width=width,
        layout_config=layout_config,
    )

    # Prepare difference rendering parameters if needed
    difference, diff_colour_scale = _prepare_difference(
        plot_spec, ground_truth, prediction
    )

    # Draw the ground truth and prediction map images
    image_groundtruth, image_prediction, image_difference, _ = _draw_frame(
        axs,
        ground_truth,
        prediction,
        plot_spec,
        diff_colour_scale,
        precomputed_difference=difference,
        levels_override=levels,
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
        cbar_axes=cbar_axes,
    )

    _set_axes_limits(axs, width=width, height=height)
    try:
        title_text = set_suptitle_with_box(fig, _build_title_static(plot_spec, date))
    except Exception:
        logger.exception("Failed to draw suptitle; continuing without title.")
        title_text = None

    _draw_warning_badge(fig, title_text, warnings)
    _maybe_add_footer(fig, plot_spec)

    try:
        return {"sea-ice_concentration-static-maps": [convert.image_from_figure(fig)]}
    finally:
        plt.close(fig)


def plot_static_inputs(  # noqa: C901, PLR0912, PLR0915
    *,
    channels: dict[str, np.ndarray],
    when: date | datetime,
    plot_spec_base: PlotSpec,
    land_mask: np.ndarray | None = None,
    styles: dict[str, dict[str, Any]] | None = None,
    save_dir: Path | None = None,
) -> dict[str, list[ImageFile]]:
    """Plot one image per input channel as a static map.

    Returns list of (name, PIL.ImageFile, saved_path|None).
    """
    channel_arrays = list(channels.values())
    channel_names = list(channels.keys())
    styles = styles or plot_spec_base.per_variable_styles
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
