# tests/plotting/test_layout.py
from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from ice_station_zebra.visualisations.layout import _build_layout, _set_axes_limits
from ice_station_zebra.visualisations.plotting_maps import (
    DEFAULT_SIC_SPEC,
    _draw_badge_with_box,
    _set_footer_with_box,
    _set_suptitle_with_box,
)

from .test_helper_plot_layout import axis_rectangle, rectangles_overlap

mpl.use("Agg")

# Silence Matplotlib animation warning in this test module
pytestmark = pytest.mark.filterwarnings(
    "ignore:Animation was deleted without rendering anything:UserWarning:matplotlib.animation"
)

if TYPE_CHECKING:
    # Imports used only for type annotations
    from datetime import date

    import numpy as np


@pytest.mark.parametrize(
    "colourbar_location", ["horizontal", "vertical"], ids=lambda s: f"cbar-{s}"
)
@pytest.mark.parametrize(
    "include_difference", [False, True], ids=lambda v: "with-diff" if v else "no-diff"
)
@pytest.mark.parametrize(
    "colourbar_strategy", ["shared", "separate"], ids=lambda s: f"strategy-{s}"
)
def test_no_axes_overlap(
    sic_pair_2d: tuple[np.ndarray, np.ndarray, date],
    *,
    colourbar_location: str,
    include_difference: bool,
    colourbar_strategy: str,
) -> None:
    """Panels and colourbars must not overlap for common layout configurations."""
    ground_truth, _, _ = sic_pair_2d

    spec = replace(
        DEFAULT_SIC_SPEC,
        colourbar_location=colourbar_location,  # type: ignore[arg-type]
        include_difference=include_difference,
        colourbar_strategy=colourbar_strategy,  # type: ignore[arg-type]
    )

    _, axes, colourbar_axes = _build_layout(
        plot_spec=spec, height=ground_truth.shape[0], width=ground_truth.shape[1]
    )

    # Collect rectangles for all visible axes: main panels + any colourbars that exist.
    rectangles = [axis_rectangle(ax) for ax in axes]
    rectangles.extend(
        axis_rectangle(ax) for ax in colourbar_axes.values() if ax is not None
    )

    # Pairwise non-overlap
    for rect_a, rect_b in combinations(rectangles, 2):
        assert not rectangles_overlap(rect_a, rect_b), (
            f"Found overlap between rectangles {rect_a} and {rect_b}"
        )


@pytest.mark.parametrize("colourbar_location", ["horizontal", "vertical"])
@pytest.mark.parametrize("include_difference", [False, True])
@pytest.mark.parametrize("colourbar_strategy", ["shared", "separate"])
def test_axes_have_reasonable_gaps(
    sic_pair_2d: tuple[np.ndarray, np.ndarray, date],
    *,
    colourbar_location: str,
    include_difference: bool,
    colourbar_strategy: str,
) -> None:
    """Require a small horizontal gutter between side-by-side axes."""
    ground_truth, prediction, _ = sic_pair_2d

    spec = replace(
        DEFAULT_SIC_SPEC,
        colourbar_location=colourbar_location,  # type: ignore[arg-type]
        include_difference=include_difference,
        colourbar_strategy=colourbar_strategy,  # type: ignore[arg-type]
    )

    _, axes, colourbar_axes = _build_layout(
        plot_spec=spec, height=ground_truth.shape[0], width=ground_truth.shape[1]
    )

    rectangles = [axis_rectangle(ax) for ax in axes]
    rectangles.extend(
        axis_rectangle(ax) for ax in colourbar_axes.values() if ax is not None
    )

    def _min_horizontal_gap(
        a: tuple[float, float, float, float], b: tuple[float, float, float, float]
    ) -> float:
        la, ba, ra, ta = a
        lb, bb, rb, tb = b
        if ra <= lb:  # a left of b
            return lb - ra
        if rb <= la:  # b left of a
            return la - rb
        return 0.0  # not horizontally ordered

    for rect_a, rect_b in combinations(rectangles, 2):
        # Skip overlapping pairs (covered by test_no_axes_overlap)
        if rectangles_overlap(rect_a, rect_b):
            continue

        # Only enforce horizontal spacing (rows are allowed to touch vertically)
        horizontal_gap = _min_horizontal_gap(rect_a, rect_b)
        if horizontal_gap > 0.0:
            assert horizontal_gap >= 0.005, (
                f"Expected ≥0.5% horizontal gap, got {horizontal_gap:.5f} for {rect_a} vs {rect_b}"
            )


def test_y_axis_orientation_for_geographical_data() -> None:
    """Test that y-axis is oriented correctly for geographical data following polar mapping conventions."""
    # Create a simple figure with axes
    fig, ax = plt.subplots()

    # Test with sample dimensions
    height, width = 48, 64

    # Apply the axis limits function
    _set_axes_limits([ax], width=width, height=height)

    # Check that y-axis is inverted for geographical convention
    # For polar data (both Arctic and Antarctic), higher latitude values should be
    # positioned at the top of the display, following environmental science conventions
    y_min, y_max = ax.get_ylim()

    # Check if the axis is properly oriented for geographical data
    # We want higher latitude values (pole-ward) at the top of the display
    if y_max > y_min:
        # Normal orientation: y_max at top
        assert y_max == height, f"Y-axis maximum should be {height}, got {y_max}"
        assert y_min == 0, f"Y-axis minimum should be 0 , got {y_min}"
    else:
        # Inverted orientation: y_min at top
        assert y_min == height, f"Y-axis minimum should be {height}, got {y_min}"
        assert y_max == 0, f"Y-axis maximum should be 0 , got {y_max}"

    # Check x-axis is still normal (left to right)
    x_min, x_max = ax.get_xlim()
    assert x_min == 0, f"X-axis minimum should be 0, got {x_min}"
    assert x_max == width, f"X-axis maximum should be {width}, got {x_max}"

    plt.close(fig)


def _text_rectangle(
    fig: plt.Figure, text_artist: plt.Text
) -> tuple[float, float, float, float]:
    """Return text bounding box in figure-normalised coords [0, 1]."""
    fig.canvas.draw()
    # Obtain a renderer in a backend-agnostic, mypy-friendly way
    canvas: Any = fig.canvas
    get_renderer = getattr(canvas, "get_renderer", None)
    if callable(get_renderer):
        renderer = get_renderer()
    else:
        renderer = getattr(canvas, "renderer", None)
    bbox = text_artist.get_window_extent(renderer=renderer)
    # Transform display to figure coords
    (x0, y0), (x1, y1) = fig.transFigure.inverted().transform(
        [(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)]
    )
    return (float(x0), float(y0), float(x1), float(y1))


@pytest.mark.parametrize("colourbar_location", ["horizontal", "vertical"])
@pytest.mark.parametrize("include_difference", [False, True])
def test_figure_text_boxes_do_not_overlap(
    sic_pair_2d: tuple[np.ndarray, np.ndarray, date],
    *,
    colourbar_location: str,
    include_difference: bool,
) -> None:
    """Ensure figure title, warning badge, and footer do not overlap panels or colourbars."""
    ground_truth, _, _ = sic_pair_2d

    spec = replace(
        DEFAULT_SIC_SPEC,
        colourbar_location=colourbar_location,  # type: ignore[arg-type]
        include_difference=include_difference,
    )

    fig, axes, caxes = _build_layout(
        plot_spec=spec, height=ground_truth.shape[0], width=ground_truth.shape[1]
    )

    # Add figure-level title, warning badge (synthetic), and footer
    title = _set_suptitle_with_box(fig, "Title")
    ty = title.get_position()[1]
    badge = _draw_badge_with_box(fig, 0.5, max(ty - 0.05, 0.0), "Warnings: example")
    footer = _set_footer_with_box(fig, "Footer metadata")

    # Collect rectangles: panels, colourbar axes, and figure texts
    rectangles = [axis_rectangle(ax) for ax in axes]
    rectangles.extend(axis_rectangle(ax) for ax in caxes.values() if ax is not None)
    rectangles.append(_text_rectangle(fig, title))
    rectangles.append(_text_rectangle(fig, badge))
    rectangles.append(_text_rectangle(fig, footer))

    # No overlaps among any of these elements
    for rect_a, rect_b in combinations(rectangles, 2):
        assert not rectangles_overlap(rect_a, rect_b), (
            f"Found overlap between rectangles {rect_a} and {rect_b}"
        )
