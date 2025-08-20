import io
import logging
from typing import Any, Literal

import numpy as np
import wandb
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from torch import Tensor
from torch.utils.data import DataLoader

from ice_station_zebra.data_loaders import CombinedDataset
from ice_station_zebra.visualisations import plot_sic_comparison, video_sic_comparison


# ---- Domain exceptions ----
class PlottingError(RuntimeError): ...


class VideoRenderError(PlottingError): ...


class InvalidArrayError(PlottingError, ValueError): ...


logger = logging.getLogger(__name__)


class PlottingCallback(Callback):
    """A callback to create plots during evaluation."""

    def __init__(
        self,
        frequency: int = 10,
        plot_sea_ice_concentration: bool = True,
        video_sea_ice_concentration: bool = True,
        video_fps: int = 2,
        video_format: Literal["mp4", "gif"] = "mp4",
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
        self.frequency = frequency
        self.plot_sea_ice_concentration = plot_sea_ice_concentration
        self.video_sea_ice_concentration = video_sea_ice_concentration
        self.video_fps = video_fps
        self.video_format = video_format

        self.plot_fns = {}
        if self.plot_sea_ice_concentration:
            self.plot_fns["sea-ice-comparison"] = plot_sic_comparison

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
            # Get date for this batch
            batch_size = outputs["target"].shape[0]

            test_dataloaders: DataLoader | list[DataLoader] | None = (
                trainer.test_dataloaders
            )
            if test_dataloaders is None:
                logger.debug("No test dataloaders found, skipping plotting.")
                return

            dataset: CombinedDataset = (
                test_dataloaders[dataloader_idx]
                if isinstance(test_dataloaders, list)
                else test_dataloaders
            ).dataset  # type: ignore[assignment]

            start_date = dataset.date_from_index(batch_size * batch_idx)

            # ----- Static Image Plots -----
            images = {}
            if self.plot_sea_ice_concentration:
                np_ground_truth, np_prediction = _extract_static_data(outputs)
                images = {
                    name: plot_fn(np_ground_truth, np_prediction, start_date)
                    for name, plot_fn in self.plot_fns.items()
                }

            # ----- Video Plots -----
            videos = {}
            if self.video_sea_ice_concentration:
                ground_truth_stream, prediction_stream = _extract_video_data(outputs)
                n_timesteps = ground_truth_stream.shape[0]
                sequence_dates = _generate_sequence_dates(
                    dataset, batch_idx, batch_size, n_timesteps
                )

                videos = _create_video_plots(
                    ground_truth_stream,
                    prediction_stream,
                    sequence_dates,
                    fps=self.video_fps,
                    format=self.video_format,
                )

            # ---- Log all media ----
            for lightning_logger in trainer.loggers:
                _log_media_to_wandb(lightning_logger, images, videos, self.video_format)
            #     for key, image_list in images.items():
            #         if hasattr(lightning_logger, "log_image"):
            #             lightning_logger.log_image(key=key, images=image_list)
            #         else:
            #             logger.debug(
            #                 f"Logger {lightning_logger.name} does not support logging images."
            #             )


# -------   Helper Functions -------


def _extract_static_data(
    outputs: dict[str, Tensor], selected_timestep: int = 0
) -> tuple[np.ndarray, np.ndarray]:
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
    _validate_static_arrays(outputs, selected_timestep)

    np_ground_truth = outputs["target"].cpu().numpy()[0, selected_timestep, 0, :, :]
    np_prediction = outputs["output"].cpu().numpy()[0, selected_timestep, 0, :, :]

    return np_ground_truth, np_prediction


def _extract_video_data(outputs: dict[str, Tensor]) -> tuple[np.ndarray, np.ndarray]:
    """Extract all timesteps data for video plots.

    Args:
        outputs: The outputs of the model. Shape: [batch, time, channels, height, width]

    Returns:
        A tuple of (ground truth, prediction) arrays.

    Raises:
        InvalidArrayError: If the arrays are not valid in shape.
    """

    # Check shape of the first timestep array
    _validate_static_arrays(outputs, 0)

    video_ground_truth = outputs["target"].cpu().numpy()[0, :, 0, :, :]
    video_prediction = outputs["output"].cpu().numpy()[0, :, 0, :, :]
    return video_ground_truth, video_prediction


def _generate_sequence_dates(
    dataset: CombinedDataset, batch_idx: int, batch_size: int, n_timesteps: int
) -> list:
    """Generate list of dates for forecast sequence."""
    return [
        dataset.date_from_index(batch_size * batch_idx + i) for i in range(n_timesteps)
    ]


def _create_video_plots(
    ground_truth_stream: np.ndarray,
    prediction_stream: np.ndarray,
    dates: list,
    fps: int = 2,
    format: Literal["mp4", "gif"] = "mp4",
) -> dict[str, bytes]:
    """Create video plots for sea ice concentration.

    Args:
        ground_truth_stream: Ground truth array with shape [T,H,W]
        prediction_stream: Prediction array with shape [T,H,W]
        dates: List of dates corresponding to each timestep
        fps: The frames per second of the video.
        format: The format of the video.

    Returns:
        A dictionary of video bytes.

    Raises:
        VideoRenderError: If the video cannot be rendered.
        InvalidArrayError: If the arrays are not valid in shape.

    """

    # Check shape of arrays
    _validate_video_arrays(ground_truth_stream, prediction_stream, dates)

    videos = {}
    try:
        videos["sea-ice-comparison-video"] = video_sic_comparison(
            ground_truth_stream, prediction_stream, dates, fps=fps, format=format
        )
    except (OSError, MemoryError) as err:
        raise VideoRenderError(f"System/encoder error: {err!s}") from err
    except (ValueError, TypeError) as err:
        raise InvalidArrayError(f"Invalid values for video render: {err!s}") from err
    return videos


def _log_media_to_wandb(
    lightning_logger, images: dict, videos: dict, video_format: Literal["mp4", "gif"]
) -> None:
    """Log both images and videos to lightning loggers."""

    # Log static images
    for key, image_list in images.items():
        if hasattr(lightning_logger, "log_image"):
            lightning_logger.log_image(key=key, images=image_list)
        else:
            logger.debug(
                f"Logger {lightning_logger.name} does not support logging images."
            )

    # Log videos
    for key, video_bytes in videos.items():
        if hasattr(lightning_logger, "experiment"):  # wandb-specific
            try:
                lightning_logger.experiment.log(
                    {key: wandb.Video(io.BytesIO(video_bytes), format=video_format)}
                )
            except ImportError:
                logger.debug("wandb not available for video logging")
        else:
            logger.debug(
                f"Logger {lightning_logger.name} does not support video logging."
            )


# ---- Validation ----

EXPECTED_MIN_BATCHES = 1


def _validate_array_shape(
    array: Tensor, name: str, expected_ndim: int = 5
) -> tuple[int, int]:
    """Validate shape of a single array and return batch size and timesteps."""
    if array.ndim != expected_ndim:
        raise InvalidArrayError(
            f"Expected {expected_ndim}D tensor for {name}, got shape {array.shape}"
        )

    n_batches, n_timesteps, *_ = array.shape
    if n_batches < EXPECTED_MIN_BATCHES:
        raise InvalidArrayError(f"Too few batches in {name}: {n_batches}")

    return n_batches, n_timesteps


def _validate_timestep(array: Tensor, name: str, selected_timestep: int) -> None:
    """Validate timestep of a single array."""
    n_timesteps, *_ = array.shape
    if not (0 <= selected_timestep < n_timesteps):
        raise InvalidArrayError(
            f"Invalid timestep: {selected_timestep} outside range [0, {n_timesteps})"
        )


def _validate_static_arrays(
    outputs: dict[str, Tensor], selected_timestep: int = 0
) -> None:
    """Raise InvalidArrayError if arrays are not valid."""

    try:
        target = outputs["target"]
        output = outputs["output"]
    except KeyError as err:
        raise InvalidArrayError(f"Missing key in outputs: {err!s}") from err

    if target.shape != output.shape:
        raise InvalidArrayError(
            "The target and output arrays must have the same shape."
        )

    # Validate both arrays
    # -- Checks the Tensor size and at least multiple batches
    _validate_array_shape(target, "target")
    _validate_array_shape(output, "output")

    # -- Checks the timestep is within the range of the array
    _validate_timestep(target, "target", selected_timestep)
    _validate_timestep(output, "output", selected_timestep)


def _validate_video_array_shape(
    array: np.ndarray, name: str, expected_ndim: int = 3
) -> int:
    """Validate shape of a single video array and return number of timesteps."""
    if array.ndim != expected_ndim:
        raise InvalidArrayError(
            f"Expected {expected_ndim}D array ([T,H,W]) for {name}, got shape {array.shape}"
        )

    n_timesteps, *_ = array.shape
    return n_timesteps


def _validate_video_arrays(
    ground_truth_stream: np.ndarray,
    prediction_stream: np.ndarray,
    dates: list,
) -> None:
    """Raise InvalidArrayError if arrays or dates are inconsistent.

    Args:
        ground_truth_stream: Ground truth array with shape [T,H,W]
        prediction_stream: Prediction array with shape [T,H,W]
        dates: List of dates corresponding to each timestep

    Raises:
        InvalidArrayError: If arrays or dates are inconsistent
    """
    # Validate array shapes
    gt_timesteps = _validate_video_array_shape(ground_truth_stream, "ground_truth")
    pred_timesteps = _validate_video_array_shape(prediction_stream, "prediction")

    # Check arrays have matching shapes
    if ground_truth_stream.shape != prediction_stream.shape:
        raise InvalidArrayError(
            f"Shape mismatch: ground_truth={ground_truth_stream.shape}, "
            f"prediction={prediction_stream.shape}"
        )

    # Check dates list matches timesteps
    if len(dates) != gt_timesteps or len(dates) != pred_timesteps:
        raise InvalidArrayError(
            f"Dates length {len(dates)} != timesteps {gt_timesteps}"
        )
