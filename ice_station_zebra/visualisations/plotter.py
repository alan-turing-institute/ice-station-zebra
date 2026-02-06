import logging
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from omegaconf import DictConfig
from torch import Tensor

from ice_station_zebra.exceptions import InvalidArrayError, VideoRenderError
from ice_station_zebra.types import ModelTestOutput, PlotSpec, TensorDimensions

from .metadata import build_metadata, format_metadata_subtitle
from .plotting_maps import plot_maps, video_maps

logger = logging.getLogger(__name__)


class Plotter:
    def __init__(self, base_path: str, plot_spec: PlotSpec) -> None:
        """A helper class to create and log plots."""
        self.base_path = Path(base_path)
        self.plot_spec = plot_spec

        # Find land mask paths for both hemispheres
        self.land_masks: dict[Literal["north", "south"], Path] = {}
        if land_masks_south := list(
            self.base_path.glob(
                "data/preprocessing/*/IceNetSIC/data/masks/south/masks/land_mask.npy"
            )
        ):
            self.land_masks["south"] = land_masks_south[0].resolve()
        if land_masks_north := list(
            self.base_path.glob(
                "data/preprocessing/*/IceNetSIC/data/masks/north/masks/land_mask.npy"
            )
        ):
            self.land_masks["north"] = land_masks_north[0].resolve()

    def set_hemisphere(self, hemisphere: Literal["north", "south"]) -> None:
        """Set the hemisphere and update the plot spec accordingly."""
        self.plot_spec.hemisphere = hemisphere
        self.plot_spec.land_mask_path = str(self.land_masks[hemisphere])

    def set_metadata(self, config: DictConfig, model_name: str) -> None:
        """Set metadata for the plotter based on the model test output."""
        metadata = build_metadata(config, model_name)
        self.plot_spec.metadata_subtitle = format_metadata_subtitle(metadata)

    def log_static_plots(
        self, outputs: ModelTestOutput, dates: list[datetime], lightning_loggers: list
    ) -> None:
        """Create and log static image plots."""
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
        self, outputs: ModelTestOutput, dates: list[datetime], lightning_loggers: list
    ) -> None:
        """Create and log video plots."""
        try:
            ground_truth_stream, prediction_stream = _extract_video_data(outputs)
            video_data = video_maps(
                self.plot_spec,
                ground_truth_stream,
                prediction_stream,
                dates,
                fps=self.plot_spec.video_fps,
                video_format=self.plot_spec.video_format,
            )
            for lightning_logger in lightning_loggers:
                if hasattr(lightning_logger, "log_video"):
                    for key, video_buffer in video_data.items():
                        video_buffer.seek(0)
                        lightning_logger.log_video(
                            key=key,
                            videos=[video_buffer],
                            format=[self.plot_spec.video_format],
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
