from __future__ import annotations

import io
import logging
from datetime import date
from typing import Any, Literal

import numpy as np
import wandb
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from PIL.ImageFile import ImageFile
from torch import Tensor
from torch.utils.data import DataLoader

from ice_station_zebra.data_loaders import CombinedDataset
from ice_station_zebra.visualisations import (
    DEFAULT_SIC_SPEC,
    DiffStrategy,
    InvalidArrayError,
    PlotSpec,
    VideoRenderError,
    plot_maps,
    video_maps,
)

logger = logging.getLogger(__name__)


class PlottingCallback(Callback):
    """A callback to create plots during evaluation."""

    def __init__(
        self,
        *,
        frequency: int = 10,
        make_static_plots: bool = True,
        make_video_plots: bool = True,
        selected_timestep: int = 0,
        include_difference: bool = True,
        diff_strategy: DiffStrategy = "precompute",
        video_fps: int = 2,
        video_format: Literal["mp4", "gif"] = "mp4",
        plot_spec: PlotSpec | None = None,
    ) -> None:
        """Create plots during evaluation.

        Args:
            frequency: Create a new plot every `frequency` batches.
            plot_sea_ice_concentration: Whether to plot sea ice concentration.
            video_sea_ice_concentration: Whether to create a video of sea ice concentration.
            video_fps: The frames per second of the video.
            video_format: The format of the video.
        """
        super().__init__()
        self.frequency = int(max(1, frequency))
        self.make_static_plots = make_static_plots
        self.make_video_plots = make_video_plots
        self.selected_timestep = int(max(0, selected_timestep))
        self.include_difference = include_difference
        self.diff_strategy = diff_strategy
        self.video_fps = video_fps
        self.video_format = video_format
        self.plot_spec = plot_spec or DEFAULT_SIC_SPEC

    # --- Lightning Hook ---
    def on_test_batch_end(
        self,
        trainer: Trainer,
        module: LightningModule,
        outputs: dict[str, Tensor],  # type: ignore[override]
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the test batch ends."""
        # Run plotting every `frequency` batches
        if batch_idx % self.frequency == 0:
            # Resolve dataset
            test_dataloaders: DataLoader | list[DataLoader] | None = (
                trainer.test_dataloaders
            )
            if test_dataloaders is None:
                logger.debug("No test dataloaders found, skipping plotting.")
                return

            loader = (
                test_dataloaders[dataloader_idx]
                if isinstance(test_dataloaders, list)
                else test_dataloaders
            )
            dataset: CombinedDataset = loader.dataset  # type: ignore[assignment]

            # ----- Static Image Plots -----
            images: dict[str, list[ImageFile]] = {}
            if self.make_static_plots:
                try:
                    np_ground_truth, np_prediction, date = _extract_static_data(
                        outputs,
                        self.plot_spec.selected_timestep,
                        dataset=dataset,
                        batch_idx=batch_idx,
                    )
                    image_list = plot_maps(
                        self.plot_spec,
                        np_ground_truth,
                        np_prediction,
                        date,
                    )
                    images["sea-ice_concentration-static-maps"] = image_list
                except InvalidArrayError as err:
                    logger.warning(
                        f"Static plotting skipped due to invalid arrays: {err}"
                    )
                except (InvalidArrayError, ValueError, MemoryError, OSError) as err:
                    logger.exception(f"Static plotting failed: {err}")

            # ----- Video Plots -----
            videos: dict[str, bytes] = {}
            if self.make_video_plots:
                try:
                    ground_truth_stream, prediction_stream, dates = _extract_video_data(
                        outputs, dataset, batch_idx
                    )
                    video_bytes = video_maps(
                        self.plot_spec,
                        ground_truth_stream,
                        prediction_stream,
                        dates,
                        fps=self.video_fps,
                        format=self.video_format,
                    )
                    videos["sea-ice_concentration-video-maps"] = video_bytes
                except (InvalidArrayError, VideoRenderError) as err:
                    logger.warning(f"Video plotting skipped: {err}")
                except (
                    InvalidArrayError,
                    VideoRenderError,
                    ValueError,
                    MemoryError,
                    OSError,
                ) as err:
                    logger.exception(f"Video plotting failed: {err}")

            # ----- Log all media -----
            for lightning_logger in trainer.loggers:
                _log_media_to_wandb(
                    lightning_logger,
                    images,
                    videos,
                    self.video_format,
                )


# -------   Helper Functions -------


def _extract_static_data(
    outputs: dict[str, Tensor],
    selected_timestep: int,
    dataset: CombinedDataset,
    batch_idx: int,
) -> tuple[np.ndarray, np.ndarray, date]:
    """
    Extract the static data from the outputs.

    Args:
        outputs: The outputs of the model. Shape: [batch, time, channels, height, width]
        selected_timestep: The timestep to extract.

    Returns:
        A tuple of (ground truth, prediction) arrays.

    Raises:
        InvalidArrayError: If the arrays are not valid in shape.

    """
    # Check shape of arrays
    target, output = _require_tensors(outputs, keys=("target", "output"))
    _assert_same_shape(target, output, name_a="target", name_b="output")

    if not (0 <= selected_timestep < target.shape[1]):
        raise InvalidArrayError(
            f"Invalid timestep: {selected_timestep} outside range [0, {target.shape[1]})"
        )

    # Use the first batch, first channel -> [H,W]
    np_ground_truth = target[0, selected_timestep, 0].detach().cpu().numpy()
    np_prediction = output[0, selected_timestep, 0].detach().cpu().numpy()

    # Get the date from the dataset
    batch_size = int(target.shape[0])
    date = dataset.date_from_index(batch_size * batch_idx + selected_timestep)

    return np_ground_truth, np_prediction, date


def _extract_video_data(
    outputs: dict[str, Tensor], dataset: CombinedDataset, batch_idx: int
) -> tuple[np.ndarray, np.ndarray, list]:
    """Extract all timesteps data for video plots.

    Args:
        outputs: The outputs of the model. Shape: [batch, time, channels, height, width]
        dataset: The dataset to get dates from
        batch_idx: The batch index

    Returns:
        A tuple of (ground truth, prediction, dates) arrays.

    Raises:
        InvalidArrayError: If the arrays are not valid in shape.
    """

    # Check shape of arrays
    target, output = _require_tensors(outputs, keys=("target", "output"))
    _assert_same_shape(target, output, name_a="target", name_b="output")

    # Use the first batch, first channel -> [T,H,W]
    ground_truth_stream = target[0, :, 0].detach().cpu().numpy()
    prediction_stream = output[0, :, 0].detach().cpu().numpy()

    # Generate dates for the sequence
    batch_size = int(target.shape[0])
    n_timesteps = int(target.shape[1])
    dates = _generate_sequence_dates(dataset, batch_idx, n_timesteps, batch_size)

    return ground_truth_stream, prediction_stream, dates


def _generate_sequence_dates(
    dataset: CombinedDataset, batch_idx: int, n_timesteps: int, batch_size: int
) -> list:
    """Generate dates for a [T] sequence anchored to this batch."""
    base = batch_size * batch_idx
    return [dataset.date_from_index(base + tt) for tt in range(n_timesteps)]


def _require_tensors(
    outputs: dict[str, Tensor], keys: tuple[str, ...]
) -> tuple[Tensor, ...]:
    """Require tensors from outputs."""
    try:
        a = outputs[keys[0]]
        b = outputs[keys[1]]
    except KeyError as err:
        raise InvalidArrayError(f"Missing key in outputs: {err!s}") from err
    return a, b


def _assert_same_shape(
    a: Tensor, b: Tensor, name_a: str, name_b: str, expected_ndim: int = 5
) -> None:
    """Assert that two tensors have the same shape."""

    if expected_ndim == 5:
        expected_str = "5D tensors [B,T,C,H,W]"
    elif expected_ndim == 3:
        expected_str = "3D tensors [T,H,W]"
    else:
        raise ValueError(f"Expected 3D or 5D tensors; got {expected_ndim}D")

    if a.ndim != expected_ndim or b.ndim != expected_ndim:
        raise InvalidArrayError(
            f"Expected {expected_ndim}D {expected_str}; got {name_a}={tuple(a.shape)}, {name_b}={tuple(b.shape)}"
        )

    if a.shape != b.shape:
        raise InvalidArrayError(
            f"Shape mismatch: {name_a}={tuple(a.shape)}, {name_b}={tuple(b.shape)}"
        )


def _log_media_to_wandb(
    lightning_logger, images: dict, videos: dict, video_format: Literal["mp4", "gif"]
) -> None:
    """Log both images and videos to lightning loggers."""

    # Static images
    for key, image_list in images.items():
        if hasattr(lightning_logger, "log_image"):
            lightning_logger.log_image(key=key, images=image_list)
        else:
            logger.debug(
                f"Logger {getattr(lightning_logger, 'name', 'unknown')} does not support logging images."
            )

    # Videos
    for key, video_bytes in videos.items():
        if hasattr(lightning_logger, "experiment"):  # wandb-specific
            try:
                lightning_logger.experiment.log(
                    {key: wandb.Video(io.BytesIO(video_bytes), format=video_format)}
                )
            except ImportError:
                logger.debug("wandb not available for video logging.")
        else:
            logger.debug(
                f"Logger {getattr(lightning_logger, 'name', 'unknown')} does not support video logging."
            )
