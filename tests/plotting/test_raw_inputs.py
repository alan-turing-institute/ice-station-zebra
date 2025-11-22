"""Tests for raw input plotting functionality.

This module tests both static plots and animations of raw input variables,
covering ERA5 weather data, OSISAF sea ice concentration, and various
styling configurations.
"""

from __future__ import annotations

import io
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pytest
from PIL import Image

from ice_station_zebra.exceptions import InvalidArrayError
from ice_station_zebra.types import PlotSpec
from ice_station_zebra.visualisations.plotting_core import style_for_variable
from ice_station_zebra.visualisations.plotting_raw_inputs import (
    plot_raw_inputs_for_timestep,
    video_raw_input_for_variable,
    video_raw_inputs_for_timesteps,
)

# Import test constants from conftest
from .conftest import TEST_DATE, TEST_HEIGHT, TEST_WIDTH

if TYPE_CHECKING:
    from pathlib import Path

# --- Tests for Static Plotting ---


def test_plot_single_channel_basic(
    era5_temperature_2d: np.ndarray,
    base_plot_spec: PlotSpec,
) -> None:
    """Test basic single channel plotting."""
    results = plot_raw_inputs_for_timestep(
        channel_arrays=[era5_temperature_2d],
        channel_names=["era5:2t"],
        when=TEST_DATE,
        plot_spec_base=base_plot_spec,
    )

    assert len(results) == 1
    name, pil_image, saved_path = results[0]
    assert name == "era5:2t"
    assert isinstance(pil_image, Image.Image)
    assert saved_path is None  # No save_dir provided


def test_plot_with_land_mask(
    era5_temperature_2d: np.ndarray,
    land_mask_2d: np.ndarray,
    base_plot_spec: PlotSpec,
) -> None:
    """Test plotting with land mask applied."""
    results = plot_raw_inputs_for_timestep(
        channel_arrays=[era5_temperature_2d],
        channel_names=["era5:2t"],
        when=TEST_DATE,
        plot_spec_base=base_plot_spec,
        land_mask=land_mask_2d,
    )

    assert len(results) == 1
    name, pil_image, _ = results[0]
    assert name == "era5:2t"
    assert isinstance(pil_image, Image.Image)


def test_plot_with_custom_styles(
    era5_temperature_2d: np.ndarray,
    base_plot_spec: PlotSpec,
    variable_styles: dict[str, dict[str, Any]],
) -> None:
    """Test plotting with custom variable styling."""
    results = plot_raw_inputs_for_timestep(
        channel_arrays=[era5_temperature_2d],
        channel_names=["era5:2t"],
        when=TEST_DATE,
        plot_spec_base=base_plot_spec,
        styles=variable_styles,
    )

    assert len(results) == 1
    name, pil_image, _ = results[0]
    assert name == "era5:2t"
    assert isinstance(pil_image, Image.Image)


def test_plot_multiple_channels(
    multi_channel_data: tuple[list[np.ndarray], list[str]],
    base_plot_spec: PlotSpec,
) -> None:
    """Test plotting multiple channels at once."""
    channel_arrays, channel_names = multi_channel_data

    results = plot_raw_inputs_for_timestep(
        channel_arrays=channel_arrays,
        channel_names=channel_names,
        when=TEST_DATE,
        plot_spec_base=base_plot_spec,
    )

    assert len(results) == len(channel_names)
    for (name, pil_image, _), expected_name in zip(results, channel_names, strict=True):
        assert name == expected_name
        assert isinstance(pil_image, Image.Image)


def test_plot_to_disk(
    era5_temperature_2d: np.ndarray,
    base_plot_spec: PlotSpec,
    tmp_path: Path,
) -> None:
    """Test saving plots to disk."""
    results = plot_raw_inputs_for_timestep(
        channel_arrays=[era5_temperature_2d],
        channel_names=["era5:2t"],
        when=TEST_DATE,
        plot_spec_base=base_plot_spec,
        save_dir=tmp_path,
    )

    assert len(results) == 1
    _, _, saved_path = results[0]
    assert saved_path is not None
    assert saved_path.exists()
    assert saved_path.suffix == ".png"


@pytest.mark.parametrize("colourbar_location", ["vertical", "horizontal"])
def test_plot_colourbar_locations(
    era5_temperature_2d: np.ndarray,
    colourbar_location: Literal["vertical", "horizontal"],
) -> None:
    """Test plotting with different colorbar orientations."""
    plot_spec = PlotSpec(
        variable="raw_inputs",
        colourmap="viridis",
        colourbar_location=colourbar_location,
    )

    results = plot_raw_inputs_for_timestep(
        channel_arrays=[era5_temperature_2d],
        channel_names=["era5:2t"],
        when=TEST_DATE,
        plot_spec_base=plot_spec,
    )

    assert len(results) == 1


@pytest.mark.parametrize(
    ("var_name", "fixture_name"),
    [
        ("era5:2t", "era5_temperature_2d"),
        ("era5:q_10", "era5_humidity_2d"),
        ("era5:10u", "era5_wind_u_2d"),
        ("osisaf-south:ice_conc", "osisaf_ice_conc_2d"),
    ],
)
def test_plot_different_variables(
    var_name: str,
    fixture_name: str,
    base_plot_spec: PlotSpec,
    variable_styles: dict[str, dict[str, Any]],
    request: pytest.FixtureRequest,
) -> None:
    """Test plotting different types of variables with appropriate styling."""
    data = request.getfixturevalue(fixture_name)

    results = plot_raw_inputs_for_timestep(
        channel_arrays=[data],
        channel_names=[var_name],
        when=TEST_DATE,
        plot_spec_base=base_plot_spec,
        styles=variable_styles,
    )

    assert len(results) == 1
    name, pil_image, _ = results[0]
    assert name == var_name
    assert isinstance(pil_image, Image.Image)


# --- Tests for Style Resolution ---


def test_style_for_variable_exact_match(
    variable_styles: dict[str, dict[str, Any]],
) -> None:
    """Test exact variable name matching in styling."""
    style = style_for_variable("era5:2t", variable_styles)

    assert style.cmap == "RdBu_r"
    assert style.two_slope_centre == 273.15
    assert style.units == "K"
    assert style.decimals == 1


def test_style_for_variable_wildcard_match(
    variable_styles: dict[str, dict[str, Any]],
) -> None:
    """Test wildcard pattern matching in styling."""
    # Add wildcard pattern
    styles_with_wildcard = {
        **variable_styles,
        "era5:q_*": {"cmap": "viridis", "decimals": 4},
    }

    style = style_for_variable("era5:q_500", styles_with_wildcard)

    assert style.cmap == "viridis"
    assert style.decimals == 4


def test_style_for_variable_no_match() -> None:
    """Test default styling when no match found."""
    style = style_for_variable("unknown:variable", {})

    # Should use defaults
    assert style.cmap is None
    assert style.decimals is None


# --- Tests for Animations ---


def test_video_single_variable_basic(
    era5_temperature_3d: np.ndarray,
    test_dates_short: list[date],
    base_plot_spec: PlotSpec,
) -> None:
    """Test basic video creation for a single variable."""
    video_buffer = video_raw_input_for_variable(
        variable_name="era5:2t",
        data_stream=era5_temperature_3d,
        dates=test_dates_short,
        plot_spec_base=base_plot_spec,
        fps=2,
        video_format="gif",
    )

    assert isinstance(video_buffer, io.BytesIO)
    video_buffer.seek(0)
    assert len(video_buffer.read()) > 1000  # Reasonable GIF size


def test_video_with_land_mask(
    era5_temperature_3d: np.ndarray,
    test_dates_short: list[date],
    land_mask_2d: np.ndarray,
    base_plot_spec: PlotSpec,
) -> None:
    """Test video creation with land mask."""
    video_buffer = video_raw_input_for_variable(
        variable_name="era5:2t",
        data_stream=era5_temperature_3d,
        dates=test_dates_short,
        plot_spec_base=base_plot_spec,
        land_mask=land_mask_2d,
        fps=2,
        video_format="gif",
    )

    assert isinstance(video_buffer, io.BytesIO)
    video_buffer.seek(0)
    assert len(video_buffer.read()) > 1000


def test_video_with_custom_style(
    era5_temperature_3d: np.ndarray,
    test_dates_short: list[date],
    base_plot_spec: PlotSpec,
    variable_styles: dict[str, dict[str, Any]],
) -> None:
    """Test video creation with custom styling."""
    video_buffer = video_raw_input_for_variable(
        variable_name="era5:2t",
        data_stream=era5_temperature_3d,
        dates=test_dates_short,
        plot_spec_base=base_plot_spec,
        styles=variable_styles,
        fps=2,
        video_format="gif",
    )

    assert isinstance(video_buffer, io.BytesIO)
    video_buffer.seek(0)
    assert len(video_buffer.read()) > 1000


def test_video_save_to_disk(
    era5_temperature_3d: np.ndarray,
    test_dates_short: list[date],
    base_plot_spec: PlotSpec,
    tmp_path: Path,
) -> None:
    """Test saving video to disk."""
    save_path = tmp_path / "test_animation.gif"

    video_buffer = video_raw_input_for_variable(
        variable_name="era5:2t",
        data_stream=era5_temperature_3d,
        dates=test_dates_short,
        plot_spec_base=base_plot_spec,
        save_path=save_path,
        fps=2,
        video_format="gif",
    )

    assert isinstance(video_buffer, io.BytesIO)
    assert save_path.exists()
    assert save_path.stat().st_size > 1000


@pytest.mark.parametrize("video_format", ["gif", "mp4"])
def test_video_formats(
    era5_temperature_3d: np.ndarray,
    test_dates_short: list[date],
    base_plot_spec: PlotSpec,
    video_format: Literal["gif", "mp4"],
) -> None:
    """Test creating videos in different formats."""
    video_buffer = video_raw_input_for_variable(
        variable_name="era5:2t",
        data_stream=era5_temperature_3d,
        dates=test_dates_short,
        plot_spec_base=base_plot_spec,
        fps=2,
        video_format=video_format,
    )

    assert isinstance(video_buffer, io.BytesIO)
    video_buffer.seek(0)
    content = video_buffer.read()
    assert len(content) > 1000

    # Check file signature
    if video_format == "gif":
        assert content[:6] == b"GIF89a"  # GIF header
    elif video_format == "mp4":
        # MP4 typically has ftyp box early
        assert b"ftyp" in content[:100]


def test_video_multiple_variables(
    test_dates_short: list[date],
    base_plot_spec: PlotSpec,
) -> None:
    """Test batch video creation for multiple variables."""
    rng = np.random.default_rng(100)
    n_timesteps = len(test_dates_short)

    # Create data streams for multiple variables
    channel_arrays_stream = [
        rng.uniform(270, 280, size=(n_timesteps, TEST_HEIGHT, TEST_WIDTH)).astype(
            np.float32
        ),
        rng.normal(0, 5, size=(n_timesteps, TEST_HEIGHT, TEST_WIDTH)).astype(
            np.float32
        ),
        rng.uniform(0, 1, size=(n_timesteps, TEST_HEIGHT, TEST_WIDTH)).astype(
            np.float32
        ),
    ]
    channel_names = ["era5:2t", "era5:10u", "osisaf-south:ice_conc"]

    results = video_raw_inputs_for_timesteps(
        channel_arrays_stream=channel_arrays_stream,
        channel_names=channel_names,
        dates=test_dates_short,
        plot_spec_base=base_plot_spec,
        fps=2,
        video_format="gif",
    )

    assert len(results) == 3
    for (name, video_buffer, saved_path), expected_name in zip(
        results, channel_names, strict=True
    ):
        assert name == expected_name
        assert isinstance(video_buffer, io.BytesIO)
        assert saved_path is None  # No save_dir provided


# --- Error Handling Tests ---


def test_plot_mismatched_arrays_and_names(
    era5_temperature_2d: np.ndarray,
    base_plot_spec: PlotSpec,
) -> None:
    """Test error when arrays and names count mismatch."""
    with pytest.raises(InvalidArrayError, match="Channels count mismatch"):
        plot_raw_inputs_for_timestep(
            channel_arrays=[era5_temperature_2d, era5_temperature_2d],
            channel_names=["era5:2t"],  # Only one name for two arrays
            when=TEST_DATE,
            plot_spec_base=base_plot_spec,
        )


def test_plot_wrong_dimension(
    base_plot_spec: PlotSpec,
) -> None:
    """Test error when input array is not 2D."""
    rng = np.random.default_rng(42)
    wrong_dim_array = rng.random((5, 5, 5)).astype(np.float32)  # 3D instead of 2D

    with pytest.raises(InvalidArrayError, match="Expected 2D"):
        plot_raw_inputs_for_timestep(
            channel_arrays=[wrong_dim_array],
            channel_names=["era5:2t"],
            when=TEST_DATE,
            plot_spec_base=base_plot_spec,
        )


def test_video_wrong_dimension(
    test_dates_short: list[date],
    base_plot_spec: PlotSpec,
) -> None:
    """Test error when video data is not 3D."""
    rng = np.random.default_rng(42)
    wrong_dim_array = rng.random((5, 5)).astype(np.float32)  # 2D instead of 3D

    with pytest.raises(InvalidArrayError, match="Expected 3D"):
        video_raw_input_for_variable(
            variable_name="era5:2t",
            data_stream=wrong_dim_array,
            dates=test_dates_short,
            plot_spec_base=base_plot_spec,
        )


def test_video_mismatched_dates(
    era5_temperature_3d: np.ndarray,
    base_plot_spec: PlotSpec,
) -> None:
    """Test error when number of dates doesn't match timesteps."""
    wrong_dates = [
        TEST_DATE,
        TEST_DATE + timedelta(days=1),
    ]  # Only 2 dates for 4 timesteps

    with pytest.raises(
        InvalidArrayError, match="Number of dates.*!= number of timesteps"
    ):
        video_raw_input_for_variable(
            variable_name="era5:2t",
            data_stream=era5_temperature_3d,
            dates=wrong_dates,
            plot_spec_base=base_plot_spec,
        )


# --- Integration Tests ---


def test_full_workflow_static_and_video(
    multi_channel_data: tuple[list[np.ndarray], list[str]],
    test_dates_short: list[date],
    base_plot_spec: PlotSpec,
    variable_styles: dict[str, dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Test complete workflow: static plots + videos for multiple variables."""
    channel_arrays, channel_names = multi_channel_data

    # 1. Create static plots
    static_results = plot_raw_inputs_for_timestep(
        channel_arrays=channel_arrays,
        channel_names=channel_names,
        when=TEST_DATE,
        plot_spec_base=base_plot_spec,
        styles=variable_styles,
        save_dir=tmp_path / "static",
    )

    assert len(static_results) == len(channel_names)
    for _, _, saved_path in static_results:
        assert saved_path is not None
        assert saved_path.exists()

    # 2. Create 3D data streams
    rng = np.random.default_rng(100)
    n_timesteps = len(test_dates_short)
    channel_arrays_stream = [
        rng.uniform(arr.min(), arr.max(), size=(n_timesteps, *arr.shape)).astype(
            np.float32
        )
        for arr in channel_arrays
    ]

    # 3. Create videos
    video_results = video_raw_inputs_for_timesteps(
        channel_arrays_stream=channel_arrays_stream,
        channel_names=channel_names,
        dates=test_dates_short,
        plot_spec_base=base_plot_spec,
        styles=variable_styles,
        fps=2,
        video_format="gif",
        save_dir=tmp_path / "videos",
    )

    assert len(video_results) == len(channel_names)
    for _, video_buffer, saved_path in video_results:
        assert isinstance(video_buffer, io.BytesIO)
        assert saved_path is not None
        assert saved_path.exists()
        assert saved_path.suffix == ".gif"
