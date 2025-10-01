from __future__ import annotations

import io
from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import pytest

from ice_station_zebra.visualisations import convert
from ice_station_zebra.visualisations.plotting_core import compute_sanity_report
from ice_station_zebra.visualisations.plotting_maps import (
    DEFAULT_SIC_SPEC,
    plot_maps,
    video_maps,
)

if TYPE_CHECKING:
    # Imports used only for type annotations
    from collections.abc import Callable, Sequence
    from datetime import date

    import numpy as np


def test_plot_maps_returns_image(
    sic_pair_2d: tuple[np.ndarray, np.ndarray, date],
) -> None:
    """plot_maps should produce a dict with a PIL image of nonzero size."""
    ground_truth, prediction, date = sic_pair_2d
    spec = replace(DEFAULT_SIC_SPEC, include_difference=True)

    result = plot_maps(spec, ground_truth, prediction, date)

    assert "sea-ice_concentration-static-maps" in result
    images = result["sea-ice_concentration-static-maps"]
    assert len(images) == 1
    image = images[0]
    assert image.width > 0
    assert image.height > 0


def test_plot_maps_emits_warning_badge(
    sic_pair_warning_2d: tuple[np.ndarray, np.ndarray, date],
) -> None:
    """plot_maps should add a red warning text when sanity report warns.

    We assert by recomputing the sanity report for the same inputs and
    requiring that warnings are non-empty. The function draws the warning text
    directly onto the figure; we avoid brittle image text OCR here.
    """
    ground_truth, prediction, date = sic_pair_warning_2d
    spec = replace(
        DEFAULT_SIC_SPEC, include_difference=True, colourbar_strategy="shared"
    )

    # Sanity report should have warnings under shared [0,1]
    (gt_min, gt_max), _ = (
        (0.0, 1.0),
        (0.0, 1.0),
    )  # shared strategy uses spec range for both
    report = compute_sanity_report(
        ground_truth,
        prediction,
        vmin=gt_min,
        vmax=gt_max,
        outside_warn=spec.outside_warn,
        severe_outside=spec.severe_outside,
        include_shared_range_mismatch_check=spec.include_shared_range_mismatch_check,
    )
    assert report.warnings, (
        "Expected non-empty warnings for constructed out-of-range data"
    )

    # Ensure plot_maps runs without error and returns an image
    result = plot_maps(spec, ground_truth, prediction, date)
    images = result["sea-ice_concentration-static-maps"]
    assert len(images) == 1, "Expected 1 image"
    assert images[0].width > 0, "Image width should be greater than 0"
    assert images[0].height > 0, "Image height should be greater than 0"


@pytest.fixture
def fake_save_animation(monkeypatch: pytest.MonkeyPatch) -> Callable[..., io.BytesIO]:
    """Monkeypatch _save_animation so video_maps runs fast in tests."""

    def _fake_save(
        _anim: object, *, _fps: int, _video_format: str = "gif"
    ) -> io.BytesIO:
        return io.BytesIO(b"fake-video-data")

    monkeypatch.setattr(convert, "_save_animation", _fake_save)
    return _fake_save


@pytest.mark.parametrize("video_format", ["gif", "mp4"])
def test_video_maps_returns_buffer_fast(
    sic_pair_3d_stream: tuple[np.ndarray, np.ndarray, Sequence[date]],
    fake_save_animation: Callable[..., io.BytesIO],
    video_format: Literal["gif", "mp4"],
) -> None:
    """video_maps should produce a dict with a BytesIO video buffer (fast path)."""
    ground_truth_stream, prediction_stream, dates = sic_pair_3d_stream
    spec = replace(DEFAULT_SIC_SPEC, include_difference=True)

    result = video_maps(
        spec,
        ground_truth_stream,
        prediction_stream,
        dates,
        fps=2,
        video_format=video_format,
    )

    # Reference the fixture to avoid unused-argument lint error
    assert fake_save_animation is not None

    assert "sea-ice_concentration-video-maps" in result
    buffer = result["sea-ice_concentration-video-maps"]
    assert isinstance(buffer, io.BytesIO)
    assert buffer.getvalue()  # not empty
