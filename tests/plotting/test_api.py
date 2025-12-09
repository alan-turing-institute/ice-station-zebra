from __future__ import annotations

import io
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest

from ice_station_zebra.exceptions import InvalidArrayError
from ice_station_zebra.visualisations import convert
from ice_station_zebra.visualisations.plotting_maps import (
    DEFAULT_SIC_SPEC,
    plot_maps,
    video_maps,
)
from ice_station_zebra.visualisations.range_check import compute_range_check_report

# Silence Matplotlib animation warning in this test module
pytestmark = pytest.mark.filterwarnings(
    "ignore:Animation was deleted without rendering anything:UserWarning"
)

if TYPE_CHECKING:
    # Imports used only for type annotations
    from collections.abc import Callable, Sequence
    from datetime import date


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
    """plot_maps should add a red warning text when range_check report warns.

    We assert by recomputing the range_check report for the same inputs and
    requiring that warnings are non-empty. The function draws the warning text
    directly onto the figure; we avoid brittle image text OCR here.
    """
    ground_truth, prediction, date = sic_pair_warning_2d
    spec = replace(
        DEFAULT_SIC_SPEC, include_difference=True, colourbar_strategy="shared"
    )

    # Range Check report should have warnings under shared [0,1]
    (gt_min, gt_max), _ = (
        (0.0, 1.0),
        (0.0, 1.0),
    )  # shared strategy uses spec range for both
    report = compute_range_check_report(
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


def test_plot_maps_with_land_mask(
    sic_pair_2d: tuple[np.ndarray, np.ndarray, date],
) -> None:
    """plot_maps should work with land mask overlay."""
    ground_truth, prediction, date = sic_pair_2d
    height, width = ground_truth.shape

    # Create a simple land mask with land in the center
    land_mask = np.zeros((height, width), dtype=bool)
    center_h, center_w = height // 2, width // 2
    land_mask[center_h - 5 : center_h + 5, center_w - 5 : center_w + 5] = True

    # Save land mask to temporary file
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_file:
        np.save(tmp_file.name, land_mask)
        land_mask_path = tmp_file.name

    try:
        spec = replace(
            DEFAULT_SIC_SPEC, include_difference=True, land_mask_path=land_mask_path
        )

        result = plot_maps(spec, ground_truth, prediction, date)

        assert "sea-ice_concentration-static-maps" in result
        images = result["sea-ice_concentration-static-maps"]
        assert len(images) == 1
        image = images[0]
        assert image.width > 0
        assert image.height > 0
    finally:
        # Clean up temporary file
        Path(land_mask_path).unlink(missing_ok=True)


def test_plot_maps_with_invalid_land_mask_shape(
    sic_pair_2d: tuple[np.ndarray, np.ndarray, date],
) -> None:
    """plot_maps should raise InvalidArrayError for wrong land mask shape."""
    ground_truth, prediction, date = sic_pair_2d

    # Create land mask with wrong shape
    wrong_shape_mask = np.zeros((10, 10), dtype=bool)

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_file:
        np.save(tmp_file.name, wrong_shape_mask)
        land_mask_path = tmp_file.name

    try:
        spec = replace(DEFAULT_SIC_SPEC, land_mask_path=land_mask_path)

        with pytest.raises(InvalidArrayError):  # Should raise InvalidArrayError
            plot_maps(spec, ground_truth, prediction, date)
    finally:
        # Clean up temporary file
        Path(land_mask_path).unlink(missing_ok=True)
