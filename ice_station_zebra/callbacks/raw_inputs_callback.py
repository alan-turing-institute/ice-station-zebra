"""Callback for plotting raw input variables during evaluation."""

# Set matplotlib backend before any plotting imports
import matplotlib as mpl

mpl.use("Agg")

import gc
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from lightning.pytorch.loggers import Logger as LightningLogger

from ice_station_zebra.data_loaders import CombinedDataset
from ice_station_zebra.exceptions import InvalidArrayError, VideoRenderError
from ice_station_zebra.types import PlotSpec
from ice_station_zebra.utils import parse_np_datetime
from ice_station_zebra.visualisations.plotting_core import (
    safe_filename,
)
from ice_station_zebra.visualisations.plotting_maps import DEFAULT_SIC_SPEC
from ice_station_zebra.visualisations.plotting_raw_inputs import (
    plot_raw_inputs_for_timestep,
    video_raw_inputs_for_timesteps,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# Constants
EXPECTED_INPUT_NDIM = 5  # Expected input data shape: [B, T, C, H, W]
DEFAULT_MAX_ANIMATION_FRAMES = (
    30  # Default frame limit for animations (≈ 1 month of daily data)
)


class RawInputsCallback(Callback):
    """A callback to plot raw input variables during evaluation."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        frequency: int | None = None,
        save_dir: str | Path | None = None,
        plot_spec: PlotSpec | None = None,
        timestep_index: int = 0,
        variable_styles: dict[str, dict[str, Any]] | None = None,
        make_video_plots: bool = False,
        video_fps: int = 2,
        video_format: Literal["mp4", "gif"] = "gif",
        video_save_dir: str | Path | None = None,
        max_animation_frames: int | None = None,
        log_to_wandb: bool = True,
    ) -> None:
        """Create raw input plots and/or animations during evaluation.

        Args:
            frequency: Create plots every `frequency` batches; `None` plots once per run.
            save_dir: Directory to save static plots to. If None and log_to_wandb=False, no plots saved.
            plot_spec: Plotting specification (colourmap, hemisphere, etc.).
            timestep_index: Which history timestep to plot (0 = most recent).
            variable_styles: Per-variable styling overrides (cmap, vmin/vmax, units, etc.).
            make_video_plots: Whether to create temporal animations of raw inputs.
            video_fps: Frames per second for animations.
            video_format: Video format ("mp4" or "gif").
            video_save_dir: Directory to save animations. If None and log_to_wandb=False, no videos saved.
            max_animation_frames: Maximum number of frames to include in animations (None = unlimited).
                                  Limits temporal accumulation to control memory and file size.
            log_to_wandb: Whether to log plots and animations to WandB (default: True).

        """
        super().__init__()
        if frequency is None:
            self.frequency = None
        else:
            self.frequency = int(max(1, frequency))

        self.save_dir: Path | None = (
            Path(save_dir) if isinstance(save_dir, str) else save_dir
        )
        self.timestep_index = timestep_index
        self.variable_styles = variable_styles or {}
        self._has_plotted = False

        # Animation settings
        self.make_video_plots = make_video_plots
        self.video_fps = video_fps
        self.video_format = video_format
        self.video_save_dir: Path | None = (
            Path(video_save_dir) if isinstance(video_save_dir, str) else video_save_dir
        )

        self.max_animation_frames = max_animation_frames

        # WandB logging control
        self.log_to_wandb = log_to_wandb

        # Ensure plot_spec is a PlotSpec instance
        if plot_spec is None:
            self.plot_spec = DEFAULT_SIC_SPEC
        else:
            self.plot_spec = plot_spec
        self._land_mask_path_detected = False
        self._land_mask_array: np.ndarray | None = None

        # Temporal data accumulation for animations
        self._temporal_data: dict[
            str, list[np.ndarray]
        ] = {}  # var_name -> list of [H,W] arrays
        self._temporal_dates: list[Any] = []
        self._dataset_ref: CombinedDataset | None = None

    def on_test_start(self, _trainer: Trainer, _module: LightningModule) -> None:
        """Called when the test loop starts."""
        logger.info("RawInputsCallback: Test loop started")

    def on_test_batch_end(  # noqa: C901, PLR0912
        self,
        trainer: Trainer,
        _module: LightningModule,
        _outputs: Any,  # noqa: ANN401
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the test batch ends."""
        logger.debug(
            "RawInputsCallback.on_test_batch_end called for batch %d", batch_idx
        )

        # Get dataset and date
        dl: DataLoader | list[DataLoader] | None = trainer.test_dataloaders
        if dl is None:
            logger.warning("No test dataloaders found, skipping raw inputs plotting.")
            return

        dataset = (dl[dataloader_idx] if isinstance(dl, Sequence) else dl).dataset
        if not isinstance(dataset, CombinedDataset):
            logger.warning(
                "Dataset is of type %s, skipping raw inputs plotting.", type(dataset)
            )
            return

        # Store dataset reference for later use
        if self._dataset_ref is None:
            self._dataset_ref = dataset

        # Get batch size from the batch data (using first dataset found)
        batch_size = 1
        for ds in dataset.inputs:
            if ds.name in batch:
                input_data = batch[ds.name]
                if input_data.ndim == EXPECTED_INPUT_NDIM:
                    batch_size = int(input_data.shape[0])
                    break

        # Calculate the sample index for the first sample in this batch
        # The callback extracts data from batch[0], so we use sample_idx = batch_size * batch_idx + 0
        sample_idx = batch_size * batch_idx

        if not dataset._available_dates:
            logger.warning(
                "No dates available for sample_idx=%d; skipping batch %d",
                sample_idx,
                batch_idx,
            )
            return

        # Get the forecast start date (as np.datetime64) for this sample
        forecast_start_date = dataset._available_dates[sample_idx]

        # Get the history steps for this forecast scenario
        history_dates = dataset.get_history_steps(forecast_start_date)

        # Get the actual date of the timestep being plotted
        timestep_date_np = history_dates[self.timestep_index]
        date = parse_np_datetime(timestep_date_np)

        # Determine if we should plot static plots this batch
        should_plot_static = False
        if self.frequency is None:
            should_plot_static = not self._has_plotted
        else:
            should_plot_static = batch_idx % self.frequency == 0

        # Always accumulate temporal data if animations are enabled
        # (regardless of static plot frequency)
        if self.make_video_plots:
            # Check if we've reached the frame limit
            if (
                self.max_animation_frames is not None
                and len(self._temporal_dates) >= self.max_animation_frames
            ):
                logger.debug(
                    "Reached max animation frames (%d), skipping further accumulation",
                    self.max_animation_frames,
                )
            else:
                try:
                    # Extract data without plotting
                    channel_arrays, channel_names = self._extract_channel_data(
                        batch, dataset
                    )
                    if channel_arrays and channel_names:
                        self._accumulate_temporal_data(
                            channel_arrays, channel_names, date
                        )
                except (ValueError, RuntimeError, IndexError, KeyError) as e:
                    logger.warning(
                        "Failed to accumulate temporal data for batch %d: %s",
                        batch_idx,
                        e,
                    )

        # Create static plots according to frequency
        if should_plot_static:
            logger.debug("Starting raw inputs plotting for batch %d", batch_idx)
            try:
                self.log_raw_inputs(batch, dataset, date, trainer.loggers, batch_idx)
                logger.info(
                    "Successfully completed raw inputs plotting for batch %d", batch_idx
                )
            except Exception:
                logger.exception("Raw inputs plotting failed for batch %d", batch_idx)
            else:
                if self.frequency is None:
                    self._has_plotted = True
        else:
            logger.debug(
                "Skipping static plots for batch %d (frequency=%s)",
                batch_idx,
                self.frequency,
            )

    def _extract_channel_data(
        self,
        batch: Mapping[str, Any],
        dataset: CombinedDataset,
    ) -> tuple[list[np.ndarray], list[str]]:
        """Extract channel data from batch without plotting.

        Returns:
            Tuple of (channel_arrays, channel_names).

        """
        # Collect all input channel arrays
        channel_arrays = []
        for ds in dataset.inputs:
            if ds.name not in batch:
                logger.warning("Dataset %s not found in batch, skipping", ds.name)
                continue

            input_data = batch[ds.name]  # Shape: [B, T, C, H, W]

            # Take first batch and specified timestep
            if input_data.ndim != EXPECTED_INPUT_NDIM:
                logger.warning(
                    "Expected 5D input data [B,T,C,H,W], got shape %s for %s",
                    input_data.shape,
                    ds.name,
                )
                continue

            timestep_data = input_data[0, self.timestep_index]  # Shape: [C, H, W]

            # Add each channel as a 2D array
            for c in range(timestep_data.shape[0]):
                channel_arr = timestep_data[c].detach().cpu().numpy()
                channel_arrays.append(channel_arr)

        if not channel_arrays:
            logger.warning("No input channels found in batch")
            return [], []

        # Get variable names from dataset
        channel_names = dataset.input_variable_names

        if len(channel_arrays) != len(channel_names):
            logger.warning(
                "Mismatch: %d channel arrays but %d channel names. Using generic names.",
                len(channel_arrays),
                len(channel_names),
            )
            channel_names = [f"channel_{i}" for i in range(len(channel_arrays))]

        return channel_arrays, channel_names

    def log_raw_inputs(
        self,
        batch: Mapping[str, Any],
        dataset: CombinedDataset,
        date: Any,  # noqa: ANN401
        lightning_loggers: list[LightningLogger],
        _batch_idx: int,
    ) -> None:
        """Extract and log raw input plots."""
        # Early return if nothing will be saved
        if not self.log_to_wandb and self.save_dir is None:
            logger.debug(
                "Skipping raw inputs plotting: log_to_wandb=False and save_dir=None"
            )
            return

        try:
            # Extract data
            channel_arrays, channel_names = self._extract_channel_data(batch, dataset)

            if not channel_arrays:
                logger.warning(
                    "No input channels found in batch, skipping raw inputs plotting"
                )
                return

            # Plot the raw inputs
            results = plot_raw_inputs_for_timestep(
                channel_arrays=channel_arrays,
                channel_names=channel_names,
                when=date,
                plot_spec_base=self.plot_spec,
                land_mask=self._land_mask_array,
                styles=self.variable_styles,
                save_dir=self.save_dir,
            )

            # Log to WandB if enabled
            if self.log_to_wandb:
                for lightning_logger in lightning_loggers:
                    if hasattr(lightning_logger, "log_image"):
                        # Group images by their name for logging
                        for var_name, pil_image, _saved_path in results:
                            safe_name = safe_filename(var_name.replace(":", "__"))
                            lightning_logger.log_image(
                                key=f"raw_inputs/{safe_name}",
                                images=[pil_image],
                            )
                    else:
                        logger.debug(
                            "Logger %s does not support images.",
                            lightning_logger.name
                            if lightning_logger.name
                            else "unknown",
                        )

            logger.debug(
                "Plotted %d raw input variables (saved to disk: %s, logged to WandB: %s)",
                len(results),
                self.save_dir is not None,
                self.log_to_wandb,
            )

        except Exception:
            logger.exception("Failed to log raw inputs")

    def _accumulate_temporal_data(
        self,
        channel_arrays: list[np.ndarray],
        channel_names: list[str],
        date: Any,  # noqa: ANN401
    ) -> None:
        """Accumulate temporal data for creating animations later.

        Args:
            channel_arrays: List of 2D arrays [H, W] for this timestep.
            channel_names: Variable names for each channel.
            date: Date/datetime for this timestep.

        """
        # Initialise temporal data storage for each variable on first call
        if not self._temporal_data:
            for name in channel_names:
                self._temporal_data[name] = []

        # Append this timestep's data for each variable
        for arr, name in zip(channel_arrays, channel_names, strict=True):
            if name in self._temporal_data:
                self._temporal_data[name].append(arr)

        # Append date
        self._temporal_dates.append(date)

        logger.debug(
            "Accumulated temporal data: %d timesteps for %d variables",
            len(self._temporal_dates),
            len(self._temporal_data),
        )

    def on_test_end(self, trainer: Trainer, _module: LightningModule) -> None:
        """Called when the test loop ends. Create and log animations."""
        if not self.make_video_plots:
            logger.debug("Video plots disabled, skipping animation creation")
            return

        if not self._temporal_data or not self._temporal_dates:
            logger.warning("No temporal data collected for animations")
            return

        logger.debug("Creating animations for %d variables", len(self._temporal_data))
        try:
            self.log_video_plots(trainer.loggers)
        finally:
            # Clear temporal data to free memory
            self._temporal_data.clear()
            self._temporal_dates.clear()
            # Force garbage collection to clean up animation resources
            gc.collect()

    def log_video_plots(self, lightning_loggers: list[LightningLogger]) -> None:  # noqa: C901
        """Create and log temporal animations of raw input variables."""
        if not self.make_video_plots:
            return

        # Early return if nothing will be saved
        if not self.log_to_wandb and self.video_save_dir is None:
            logger.debug(
                "Skipping video plotting: log_to_wandb=False and video_save_dir=None"
            )
            return

        try:
            # Convert accumulated data to 3D arrays [T, H, W]
            channel_arrays_stream = []
            channel_names = []

            for var_name, frames in self._temporal_data.items():
                if not frames:
                    continue
                # Stack frames into [T, H, W]
                data_stream = np.stack(frames, axis=0)
                channel_arrays_stream.append(data_stream)
                channel_names.append(var_name)

            if not channel_arrays_stream:
                logger.warning("No data to create animations")
                return

            logger.info(
                "Creating animations: %d variables x %d timesteps",
                len(channel_names),
                len(self._temporal_dates),
            )

            # Create animations for all variables
            results = video_raw_inputs_for_timesteps(
                channel_arrays_stream=channel_arrays_stream,
                channel_names=channel_names,
                dates=self._temporal_dates,
                plot_spec_base=self.plot_spec,
                land_mask=self._land_mask_array,
                styles=self.variable_styles,
                fps=self.video_fps,
                video_format=self.video_format,
                save_dir=self.video_save_dir,
            )

            # Log to WandB if enabled
            if self.log_to_wandb:
                for lightning_logger in lightning_loggers:
                    if hasattr(lightning_logger, "log_video"):
                        for var_name, video_buffer, _saved_path in results:
                            # Ensure buffer is at start
                            video_buffer.seek(0)
                            safe_name = safe_filename(var_name.replace(":", "__"))
                            lightning_logger.log_video(
                                key=f"raw_inputs_video/{safe_name}",
                                videos=[video_buffer],
                                format=[self.video_format],
                            )
                            logger.debug("Logged video for %s to WandB", var_name)
                    else:
                        logger.debug(
                            "Logger %s does not support videos.",
                            lightning_logger.name
                            if lightning_logger.name
                            else "unknown",
                        )

            logger.info(
                "Successfully created %d animations (saved to disk: %s, logged to WandB: %s)",
                len(results),
                self.video_save_dir is not None,
                self.log_to_wandb,
            )

        except (InvalidArrayError, VideoRenderError) as err:
            logger.warning("Video plotting skipped: %s", err)
        except (ValueError, MemoryError, OSError):
            logger.exception("Video plotting failed")
