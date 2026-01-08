"""CLI commands for visualisation tasks."""

import logging
from pathlib import Path
from typing import Annotated, Any, cast

import hydra
import numpy as np
import typer
from omegaconf import DictConfig, OmegaConf

from ice_station_zebra.callbacks.raw_inputs_callback import (
    DEFAULT_MAX_ANIMATION_FRAMES,
    RawInputsCallback,
)
from ice_station_zebra.cli import hydra_adaptor
from ice_station_zebra.data_loaders import CombinedDataset, ZebraDataModule
from ice_station_zebra.visualisations.plotting_raw_inputs import (
    plot_raw_inputs_for_timestep,
    video_raw_inputs_for_timesteps,
)

# Create the typer app
visualisations_cli = typer.Typer(help="Visualisation commands")

log = logging.getLogger(__name__)


def _extract_channel_arrays(
    batch: dict[str, Any], test_dataset: CombinedDataset, timestep_idx: int = 0
) -> list[np.ndarray]:
    """Extract channel arrays from a dataset item.

    Args:
        batch: Dataset item (returns 4D arrays [T, C, H, W]).
        test_dataset: Test dataset instance.
        timestep_idx: Index of timestep to extract (default: 0).

    Returns:
        List of 2D channel arrays.

    """
    channel_arrays = []
    for ds in test_dataset.inputs:
        if ds.name not in batch:
            log.warning("Dataset %s not found in batch", ds.name)
            continue

        input_data = batch[ds.name]  # Shape: [T, C, H, W] - numpy array
        timestep_data = input_data[timestep_idx]  # Shape: [C, H, W]
        channel_arrays.extend([timestep_data[c] for c in range(timestep_data.shape[0])])

    return channel_arrays


def _collect_temporal_data(
    test_dataset: CombinedDataset, start_idx: int, n_frames: int
) -> tuple[list[np.ndarray], list[str], list[Any]]:
    """Collect temporal data for animation.

    Args:
        test_dataset: Test dataset instance.
        start_idx: Starting sample index.
        n_frames: Number of frames to collect.

    Returns:
        Tuple of (data_streams, channel_names, dates).

    """
    channel_names = test_dataset.input_variable_names
    n_vars = len(channel_names)

    # Initialize data collection
    temporal_data_per_var: list[list[np.ndarray]] = [[] for _ in range(n_vars)]
    dates: list[Any] = []

    for idx in range(start_idx, start_idx + n_frames):
        batch = test_dataset[idx]
        date = test_dataset.date_from_index(idx)

        # Collect data from all input datasets
        var_idx = 0
        for ds in test_dataset.inputs:
            if ds.name not in batch:
                log.warning("Dataset %s not found in batch", ds.name)
                continue

            input_data = batch[ds.name]  # Shape: [T, C, H, W]
            timestep_data = input_data[0]  # Take first timestep, Shape: [C, H, W]

            # Add each channel
            for c in range(timestep_data.shape[0]):
                var_data = timestep_data[c]
                temporal_data_per_var[var_idx].append(var_data)
                var_idx += 1

        dates.append(date)

    # Stack into list of 3D arrays [T, H, W] for each variable
    data_streams = [
        np.stack(var_frames, axis=0) for var_frames in temporal_data_per_var
    ]

    return data_streams, channel_names, dates


@visualisations_cli.command()
@hydra_adaptor
def plot_raw_inputs(
    config: DictConfig,
    forecast_date_idx: Annotated[
        int,
        typer.Option(help="Index of the forecast scenario to plot (default: 0)"),
    ] = 0,
    timestep_idx: Annotated[
        int,
        typer.Option(
            help="Index of the timestep within the history window to plot (default: 0, i.e., first timestep)"
        ),
    ] = 0,
    output_dir: Annotated[
        str | None,
        typer.Option(help="Directory to save plots (overrides config if provided)"),
    ] = None,
) -> None:
    r"""Plot raw inputs for a specific timestep from a forecast scenario.

    This command creates static plots of all input variables for a single timestep
    from a selected forecast scenario. Each sample in the dataset represents a
    forecast scenario (identified by a date) and contains multiple timesteps of
    historical data leading up to that forecast date.

    Args:
        config: Hydra config (provided via --config-name option).
        forecast_date_idx: Index of the forecast scenario to plot. This selects which
            forecast date/scenario (maps to available_dates[forecast_date_idx]).
            Default: 0.
        timestep_idx: Index of the timestep within the selected scenario's history
            window to plot. Each scenario contains n_history_steps timesteps.
            Default: 0 (first timestep in the history window).
        output_dir: Directory to save plots (overrides config if provided).

    Note:
        You must specify --config-name to use your local config file (e.g., lfrance.local.yaml).
        Settings are read from config.evaluate.callbacks.raw_inputs in that config.

    Example:
        # Plot the first timestep of the first forecast scenario
        uv run zebra visualisations plot-raw-inputs \\
            --config-name lfrance.local.yaml \\
            --sample-idx 0 \\
            --timestep-idx 0

        # Plot the last timestep of the 10th forecast scenario
        uv run zebra visualisations plot-raw-inputs \\
            --config-name lfrance.local.yaml \\
            --sample-idx 9 \\
            --timestep-idx 6

    """
    # Instantiate callback from config to get all settings
    raw_inputs_cfg = config.get("evaluate", {}).get("callbacks", {}).get("raw_inputs")
    if raw_inputs_cfg is None:
        # Create minimal config if not found
        raw_inputs_cfg = {}
    callback = (
        hydra.utils.instantiate(raw_inputs_cfg)
        if raw_inputs_cfg
        else RawInputsCallback()
    )
    callback.config = OmegaConf.to_container(config, resolve=True)

    # Create data module and prepare data
    data_module = ZebraDataModule(config)
    data_module.prepare_data()
    data_module.setup("test")

    # Get the test dataset
    test_dataloader = data_module.test_dataloader()
    test_dataset_raw = test_dataloader.dataset
    if test_dataset_raw is None:
        msg = "No test dataset available!"
        raise ValueError(msg)
    test_dataset = cast("CombinedDataset", test_dataset_raw)

    log.info("Test dataset has %d samples", len(test_dataset))

    # Validate sample index
    if forecast_date_idx < 0 or forecast_date_idx >= len(test_dataset):
        msg = f"Sample index {forecast_date_idx} out of range [0, {len(test_dataset)})"
        raise ValueError(msg)

    # Get a sample from the dataset
    batch = test_dataset[forecast_date_idx]
    date = test_dataset.date_from_index(forecast_date_idx)

    # Validate timestep index
    # Each sample contains n_history_steps timesteps
    n_history_steps = test_dataset.n_history_steps
    if timestep_idx < 0 or timestep_idx >= n_history_steps:
        msg = (
            f"Timestep index {timestep_idx} out of range [0, {n_history_steps}) "
            f"for sample {forecast_date_idx}"
        )
        raise ValueError(msg)

    log.info(
        "Plotting raw inputs for forecast scenario date: %s, timestep %d/%d",
        date,
        timestep_idx,
        n_history_steps - 1,
    )

    # Extract channel arrays
    channel_arrays = _extract_channel_arrays(batch, test_dataset, timestep_idx)
    channel_names = test_dataset.input_variable_names
    log.info("Total channels: %d", len(channel_names))

    # Use callback's settings (with command-line override for output_dir)
    plot_spec = callback.plot_spec
    variable_styles = callback.variable_styles
    save_dir = (
        Path(output_dir)
        if output_dir
        else callback.save_dir or Path("./raw_input_plots")
    )

    # Plot the raw inputs
    results = plot_raw_inputs_for_timestep(
        channel_arrays=channel_arrays,
        channel_names=channel_names,
        when=date,
        plot_spec_base=plot_spec,
        land_mask=None,  # Will be auto-loaded if available
        styles=variable_styles,
        save_dir=save_dir,
    )

    log.info("Successfully plotted %d variables to %s", len(results), save_dir)
    for var_name, _pil_img, saved_path in results:
        if saved_path:
            log.info("  - %s: %s", var_name, saved_path)


@visualisations_cli.command()
@hydra_adaptor
def animate_raw_inputs(
    config: DictConfig,
    n_frames: Annotated[
        int | None,
        typer.Option(
            help="Number of frames to include in animation (overrides config if provided)"
        ),
    ] = None,
    output_dir: Annotated[
        str | None,
        typer.Option(
            help="Directory to save animations (overrides config if provided)"
        ),
    ] = None,
    fps: Annotated[
        int | None,
        typer.Option(
            help="Frames per second for animation (overrides config if provided)"
        ),
    ] = None,
    video_format: Annotated[
        str | None,
        typer.Option(
            help="Video format: 'gif' or 'mp4' (overrides config if provided)"
        ),
    ] = None,
    start_idx: Annotated[
        int,
        typer.Option(help="Starting sample index (default: 0)"),
    ] = 0,
) -> None:
    r"""Create animations of raw inputs over time from the test dataset.

    This command creates temporal animations showing how individual input variables
    evolve over time. Settings are read from config.evaluate.callbacks.raw_inputs
    in your YAML config files.

    Args:
        config: Hydra config (provided via --config-name option).
        n_frames: Number of frames to include in animation (overrides config if provided).
        output_dir: Directory to save animations (overrides config if provided).
        fps: Frames per second for animation (overrides config if provided).
        video_format: Video format: 'gif' or 'mp4' (overrides config if provided).
        start_idx: Starting sample index (default: 0).

    Note:
        You must specify --config-name to use your local config file (e.g., lfrance.local.yaml).
        Settings are read from config.evaluate.callbacks.raw_inputs in that config.

    Example:
        uv run zebra visualisations animate-raw-inputs \\
            --config-name lfrance.local.yaml \\
            --start-idx 0

    """
    # Instantiate callback from config to get all settings
    raw_inputs_cfg = config.get("evaluate", {}).get("callbacks", {}).get("raw_inputs")
    if raw_inputs_cfg is None:
        raw_inputs_cfg = {}
    callback = (
        hydra.utils.instantiate(raw_inputs_cfg)
        if raw_inputs_cfg
        else RawInputsCallback()
    )
    callback.config = OmegaConf.to_container(config, resolve=True)

    # Get settings from callback (with command-line overrides)
    # Use callback's value, or fall back to default constant if None
    n_frames = n_frames or callback.max_animation_frames or DEFAULT_MAX_ANIMATION_FRAMES
    fps = fps or callback.video_fps
    video_format = video_format or callback.video_format
    save_dir = (
        Path(output_dir)
        if output_dir
        else callback.video_save_dir or Path("./raw_input_animations")
    )

    # Validate video format
    if video_format not in ("gif", "mp4"):
        msg = f"Video format must be 'gif' or 'mp4', got '{video_format}'"
        raise ValueError(msg)

    # Create data module and prepare data
    data_module = ZebraDataModule(config)
    data_module.prepare_data()
    data_module.setup("test")

    # Get the test dataset
    test_dataloader = data_module.test_dataloader()
    test_dataset_raw = test_dataloader.dataset
    if test_dataset_raw is None:
        msg = "No test dataset available!"
        raise ValueError(msg)
    test_dataset = cast("CombinedDataset", test_dataset_raw)

    log.info("Test dataset has %d samples", len(test_dataset))

    # Determine number of frames
    max_frames = len(test_dataset) - start_idx
    n_frames = min(n_frames, max_frames)
    if n_frames <= 0:
        msg = f"Not enough samples (start_idx={start_idx}, dataset_size={len(test_dataset)})"
        raise ValueError(msg)

    log.info(
        "Creating animations with %d frames starting from index %d", n_frames, start_idx
    )

    # Collect temporal data for all variables
    data_streams, channel_names, dates = _collect_temporal_data(
        test_dataset, start_idx, n_frames
    )

    log.info(
        "Collected data streams: %d variables x %s",
        len(data_streams),
        data_streams[0].shape if data_streams else "N/A",
    )

    # Use callback's settings
    plot_spec = callback.plot_spec
    variable_styles = callback.variable_styles

    # Create animations for all variables
    log.info("Creating batch animations...")
    results = video_raw_inputs_for_timesteps(
        channel_arrays_stream=data_streams,
        channel_names=channel_names,
        dates=dates,
        plot_spec_base=plot_spec,
        styles=variable_styles,
        fps=fps,
        video_format=video_format,  # type: ignore[arg-type]
        save_dir=save_dir,
    )

    log.info("Successfully created %d animations:", len(results))
    for var_name, video_buffer, save_path in results:
        log.info(
            "  - %s: %s (size: %d bytes)",
            var_name,
            save_path,
            len(video_buffer.getvalue()),
        )


if __name__ == "__main__":
    visualisations_cli()
