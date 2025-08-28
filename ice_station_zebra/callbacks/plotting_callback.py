import logging
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from lightning.pytorch.loggers import Logger as LightningLogger
from torch import Tensor

from ice_station_zebra.data_loaders import CombinedDataset
from ice_station_zebra.exceptions import InvalidArrayError, VideoRenderError
from ice_station_zebra.types import ModelTestOutput, TensorDimensions
from ice_station_zebra.visualisations import (
    DEFAULT_SIC_SPEC,
    PlotSpec,
    plot_maps,
    video_maps,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class PlottingCallback(Callback):
    """A callback to create plots during evaluation."""

    def __init__(
        self,
        *,
        frequency: int = 10,
        make_static_plots: bool = True,
        make_video_plots: bool = True,
        video_fps: int = 2,
        video_format: Literal["mp4", "gif"] = "mp4",
        plot_spec: PlotSpec | None = None,
    ) -> None:
        """Create plots during evaluation.

        Args:
            frequency: Create a new plot every `frequency` batches.
            make_static_plots: Whether to create static plots.
            make_video_plots: Whether to create video plots.
            video_fps: Frames per second for video plots.
            video_format: Format for video plots (mp4 or gif).
            plot_spec: Plotting specification to use (contains difference settings,
                      timestep selection, etc.).

        """
        super().__init__()
        self.frequency = int(max(1, frequency))
        self.make_static_plots = make_static_plots
        self.make_video_plots = make_video_plots
        self.video_fps = video_fps
        self.video_format = video_format
        self.plot_spec = plot_spec or DEFAULT_SIC_SPEC

    # --- Lightning Hook ---
    def on_test_batch_end(
        self,
        trainer: Trainer,
        _module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        _batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the test batch ends."""
        # Only run plotting every `frequency` batches
        if batch_idx % self.frequency:
            return

        # Check that outputs is a ModelTestOutput
        if not isinstance(outputs, ModelTestOutput):
            msg = f"Output is of type {type(outputs)}, skipping plotting."
            logger.warning(msg)
            return

        # Get date for this batch
        dl: DataLoader | list[DataLoader] | None = trainer.test_dataloaders
        if dl is None:
            logger.warning("No test dataloaders found, skipping plotting.")
            return
        dataset = (dl[dataloader_idx] if isinstance(dl, Sequence) else dl).dataset
        if not isinstance(dataset, CombinedDataset):
            msg = f"Dataset is of type {type(dataset)}, skipping plotting."
            logger.warning(msg)
            return

        # Get sequence dates for static and video plots
        batch_size = int(outputs.target.shape[0])
        n_timesteps = int(outputs.target.shape[1])
        dates = [
            dataset.date_from_index(batch_size * batch_idx + tt)
            for tt in range(n_timesteps)
        ]

        # Log static and video plots
        self.log_static_plots(outputs, dates, trainer.loggers)
        self.log_video_plots(outputs, dates, trainer.loggers)

    def log_static_plots(
        self,
        outputs: ModelTestOutput,
        dates: list[datetime],
        lightning_loggers: list[LightningLogger],
    ) -> None:
        """Create and log static image plots."""
        if not self.make_static_plots:
            return
        try:
            np_ground_truth, np_prediction, date = _extract_static_data(
                outputs, self.plot_spec.selected_timestep, dates
            )
            images = plot_maps(self.plot_spec, np_ground_truth, np_prediction, date)
            # Log static images
            for lightning_logger in lightning_loggers:
                if hasattr(lightning_logger, "log_image"):
                    for key, image_list in images.items():
                        lightning_logger.log_image(key=key, images=image_list)
                else:
                    logger.debug(
                        "Logger %s does not support images.",
                        lightning_logger.name if lightning_logger.name else "unknown",
                    )
        except InvalidArrayError as err:
            logger.warning("Static plotting skipped due to invalid arrays: %s", err)
        except (ValueError, MemoryError, OSError):
            logger.exception("Static plotting failed")

    def log_video_plots(
        self,
        outputs: ModelTestOutput,
        dates: list[datetime],
        lightning_loggers: list[LightningLogger],
    ) -> None:
        """Create and log video plots."""
        if not self.make_video_plots:
            return
        try:
            ground_truth_stream, prediction_stream = _extract_video_data(outputs)
            video_data = video_maps(
                self.plot_spec,
                ground_truth_stream,
                prediction_stream,
                dates,
                fps=self.video_fps,
                video_format=self.video_format,
            )
            for lightning_logger in lightning_loggers:
                if hasattr(lightning_logger, "log_video"):
                    for key, video_buffer in video_data.items():
                        video_buffer.seek(0)
                        lightning_logger.log_video(
                            key=key,
                            videos=[video_buffer],
                            format=[self.video_format],
                        )
                else:
                    logger.debug(
                        "Logger %s does not support videos.",
                        lightning_logger.name if lightning_logger.name else "unknown",
                    )
        except (InvalidArrayError, VideoRenderError) as err:
            logger.warning("Video plotting skipped: %s", err)
        except (ValueError, MemoryError, OSError):
            logger.exception("Video plotting failed")


def _extract_static_data(
    outputs: ModelTestOutput,
    selected_timestep: int,
    dates: list,
) -> tuple[np.ndarray, np.ndarray, date]:
    """Extract the static data from the outputs.

    Args:
        outputs: The outputs of the model. Shape: [batch, time, channels, height, width]
        selected_timestep: The timestep to extract.
        dates: Pre-generated list of dates for the sequence.

    Returns:
        A tuple of (ground truth, prediction) arrays.

    Raises:
        InvalidArrayError: If the arrays are not valid in shape.

    """
    # Check shape of arrays
    _assert_same_shape(
        outputs.target, outputs.prediction, name_a="target", name_b="prediction"
    )

    if not (0 <= selected_timestep < outputs.target.shape[1]):
        error_msg = f"Invalid timestep: {selected_timestep} outside range [0, {outputs.target.shape[1]})"
        raise InvalidArrayError(error_msg)

    # Use the first batch, first channel -> [H,W]
    np_ground_truth = outputs.target[0, selected_timestep, 0].detach().cpu().numpy()
    np_prediction = outputs.prediction[0, selected_timestep, 0].detach().cpu().numpy()

    # Get the date from the pre-generated dates list
    date = dates[selected_timestep]

    return np_ground_truth, np_prediction, date


def _extract_video_data(outputs: ModelTestOutput) -> tuple[np.ndarray, np.ndarray]:
    """Extract all timesteps data for video plots.

    Args:
        outputs: The outputs of the model. Shape: [batch, time, channels, height, width]

    Returns:
        A tuple of (ground truth, prediction) arrays.

    Raises:
        InvalidArrayError: If the arrays are not valid in shape.

    """
    # Check shape of arrays
    _assert_same_shape(
        outputs.target, outputs.prediction, name_a="target", name_b="prediction"
    )

    # Use the first batch, first channel -> [T,H,W]
    ground_truth_stream = outputs.target[0, :, 0].detach().cpu().numpy()
    prediction_stream = outputs.prediction[0, :, 0].detach().cpu().numpy()

    return ground_truth_stream, prediction_stream


def _assert_same_shape(
    a: Tensor,
    b: Tensor,
    name_a: str,
    name_b: str,
    expected_ndim: TensorDimensions = TensorDimensions.BTCHW,
) -> None:
    """Assert that two tensors have the same shape."""
    if expected_ndim == TensorDimensions.BTCHW:
        expected_str = "5D tensors [B,T,C,H,W]"
    elif expected_ndim == TensorDimensions.THW:
        expected_str = "3D tensors [T,H,W]"
    else:
        msg = f"Expected 3D or 5D tensors; got {expected_ndim}D"
        raise ValueError(msg)

    if a.ndim != expected_ndim or b.ndim != expected_ndim:
        msg = f"Expected {expected_ndim}D {expected_str}; got {name_a}={tuple(a.shape)}, {name_b}={tuple(b.shape)}"
        raise InvalidArrayError(msg)

    if a.shape != b.shape:
        msg = f"Shape mismatch: {name_a}={tuple(a.shape)}, {name_b}={tuple(b.shape)}"
        raise InvalidArrayError(msg)
