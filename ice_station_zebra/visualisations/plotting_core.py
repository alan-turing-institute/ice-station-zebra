from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


# -- Exceptions shared by all plotting functions --
class PlottingError(RuntimeError): ...


class VideoRenderError(PlottingError): ...


class InvalidArrayError(PlottingError, ValueError): ...


DiffMode = Literal["signed", "absolute", "smape"]


@dataclass(frozen=True)
class PlotSpec:
    variable: str
    title_groundtruth: str = "Ground Truth"
    title_prediction: str = "Prediction"
    title_difference: str = "Difference"
    colourmap: str = "viridis"
    diff_mode: DiffMode = "signed"
    vmin: float | None = 0.0
    vmax: float | None = 1.0
    colourbar_location: Literal["vertical", "horizontal"] = "vertical"


def compute_difference(
    ground_truth: np.ndarray, prediction: np.ndarray, diff_mode: DiffMode
) -> np.ndarray:
    """Compute the difference between the ground truth and prediction.

    Args:
        ground_truth: The ground truth array. [T,H,W]
        prediction: The prediction array. [T,H,W]
        diff_mode: Method to compute the difference.

    Returns:
        Difference array. [T,H,W]

    Raises:
        ValueError: If the difference mode is invalid.
    """
    if diff_mode == "signed":
        return ground_truth - prediction
    elif diff_mode == "absolute":
        return np.abs(ground_truth - prediction)
    elif diff_mode == "smape":
        denom = np.clip((np.abs(ground_truth) + np.abs(prediction)) / 2.0, 1e-6, None)
        return np.abs(prediction - ground_truth) / denom
    else:
        raise ValueError(f"Invalid difference mode: {diff_mode}")


# --- Validation ---


def validate_2d_pair(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> tuple[int, int]:
    """
    Validate that both arrays are 2D [H,W] and have the same shape.

    Args:
        ground_truth: The ground truth array. [H,W]
        prediction: The prediction array. [H,W]

    Returns:
        The shape of the arrays. (H, W)

    Raises:
        InvalidArrayError: If the arrays are not 2D or have different shapes.
    """
    if ground_truth.ndim != 2 or prediction.ndim != 2:
        raise InvalidArrayError(
            f"Expected 2D [H,W]; got ground truth={ground_truth.shape}, prediction={prediction.shape}"
        )
    if ground_truth.shape != prediction.shape:
        raise InvalidArrayError(
            f"Shape mismatch: ground truth={ground_truth.shape}, prediction={prediction.shape}"
        )
    return ground_truth.shape  # (H, W)


def validate_3d_streams(
    ground_truth_stream: np.ndarray,
    prediction_stream: np.ndarray,
) -> tuple[int, int]:
    """
    Validate that both arrays are 3D [T,H,W] and have the same shape.

    Args:
        ground_truth_stream: The ground truth array. [T,H,W]
        prediction_stream: The prediction array. [T,H,W]

    Returns:
        The shape of the arrays. (T,H,W)

    Raises:
        InvalidArrayError: If the arrays are not 2D or have different shapes.
    """
    if ground_truth_stream.ndim != 3 or prediction_stream.ndim != 3:
        raise InvalidArrayError(
            f"Expected 3D [T,H,W]; got ground truth={ground_truth_stream.shape}, prediction={prediction_stream.shape}"
        )
    if ground_truth_stream.shape != prediction_stream.shape:
        raise InvalidArrayError(
            f"Shape mismatch: ground truth={ground_truth_stream.shape}, prediction={prediction_stream.shape}"
        )
    return ground_truth_stream.shape  # (T,H,W)
