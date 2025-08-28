from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from lightning.pytorch import Callback

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import date, datetime
    from typing import Any

    import numpy as np
    from lightning import LightningModule, Trainer
    from torch import Tensor
    from torch.utils.data import DataLoader

from ice_station_zebra.data_loaders import CombinedDataset
from ice_station_zebra.types import ModelTestOutput, TensorDimensions
from ice_station_zebra.visualisations import (
    DEFAULT_SIC_SPEC,
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
    def on_test_batch_end(  # noqa: C901, PLR0912
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
            logger.warning("Dataset is not a CombinedDataset, skipping plotting.")
            return

        # Get sequence dates for static and video plots
        target, prediction = _require_tensors(outputs, keys=("target", "prediction"))
        batch_size = int(target.shape[0])
        n_timesteps = int(target.shape[1])
        dates = _generate_sequence_dates(dataset, batch_idx, n_timesteps, batch_size)

        # ----- Static Image Plots -----
        if self.make_static_plots:
            try:
                np_ground_truth, np_prediction, date = _extract_static_data(
                    outputs,
                    self.plot_spec.selected_timestep,
                    dates,
                )
                images = plot_maps(
                    self.plot_spec,
                    np_ground_truth,
                    np_prediction,
                    date,
                )
                # Log static images
                for lightning_logger in trainer.loggers:
                    if hasattr(lightning_logger, "log_image"):
                        for key, image_list in images.items():
                            lightning_logger.log_image(key=key, images=image_list)
                    else:
                        logger.debug(
                            "Logger %s does not support images.",
                            lightning_logger.name
                            if lightning_logger.name
                            else "unknown",
                        )
            except InvalidArrayError as err:
                logger.warning("Static plotting skipped due to invalid arrays: %s", err)
            except (ValueError, MemoryError, OSError):
                logger.exception("Static plotting failed")

        # ----- Video Plots -----
        if self.make_video_plots:
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
                for lightning_logger in trainer.loggers:
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
                            lightning_logger.name
                            if lightning_logger.name
                            else "unknown",
                        )
            except (InvalidArrayError, VideoRenderError) as err:
                logger.warning("Video plotting skipped: %s", err)
            except (
                ValueError,
                MemoryError,
                OSError,
            ):
                logger.exception("Video plotting failed")


# -------   Helper Functions -------


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
    target, prediction = _require_tensors(outputs, keys=("target", "prediction"))
    _assert_same_shape(target, prediction, name_a="target", name_b="prediction")

    if not (0 <= selected_timestep < target.shape[1]):
        error_msg = f"Invalid timestep: {selected_timestep} outside range [0, {target.shape[1]})"
        raise InvalidArrayError(error_msg)

    # Use the first batch, first channel -> [H,W]
    np_ground_truth = target[0, selected_timestep, 0].detach().cpu().numpy()
    np_prediction = prediction[0, selected_timestep, 0].detach().cpu().numpy()

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
    target, prediction = _require_tensors(outputs, keys=("target", "prediction"))
    _assert_same_shape(target, prediction, name_a="target", name_b="prediction")

    # Use the first batch, first channel -> [T,H,W]
    ground_truth_stream = target[0, :, 0].detach().cpu().numpy()
    prediction_stream = prediction[0, :, 0].detach().cpu().numpy()

    return ground_truth_stream, prediction_stream


def _generate_sequence_dates(
    dataset: CombinedDataset, batch_idx: int, n_timesteps: int, batch_size: int
) -> list[datetime]:
    """Generate dates for a [T] sequence anchored to this batch."""
    base = batch_size * batch_idx
    return [dataset.date_from_index(base + tt) for tt in range(n_timesteps)]


def _require_tensors(
    outputs: ModelTestOutput, keys: tuple[str, ...]
) -> tuple[Tensor, ...]:
    """Require tensors from outputs."""
    try:
        a = outputs[keys[0]]
        b = outputs[keys[1]]
    except KeyError as err:
        msg = f"Missing key in outputs: {err!s}"
        raise InvalidArrayError(msg) from err
    return a, b


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
