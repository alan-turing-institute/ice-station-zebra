from pathlib import Path

import numpy as np
from matplotlib.colors import TwoSlopeNorm

from ice_station_zebra.exceptions import InvalidArrayError
from ice_station_zebra.types import DiffColourmapSpec, DiffMode, DiffStrategy, PlotSpec

# Constants for land mask validation
EXPECTED_LAND_MASK_DIMENSIONS = 2


def levels_from_spec(spec: PlotSpec) -> np.ndarray:
    """Generate contour levels from a plotting specification.

    Creates evenly-spaced contour levels based on the minimum and maximum values
    specified in the PlotSpec. If vmin or vmax are not specified, defaults to
    0.0 and 1.0.

    Args:
        spec: PlotSpec object containing vmin and vmax parameters.

    Returns:
        NumPy array of contour levels spanning from vmin to vmax with
        n_contour_levels steps.

    """
    vmin = 0.0 if spec.vmin is None else spec.vmin
    vmax = 1.0 if spec.vmax is None else spec.vmax
    return np.linspace(vmin, vmax, spec.n_contour_levels)


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
    if diff_mode == "absolute":
        return np.abs(ground_truth - prediction)
    if diff_mode == "smape":
        denom = np.clip((np.abs(ground_truth) + np.abs(prediction)) / 2.0, 1e-6, None)
        return np.abs(prediction - ground_truth) / denom
    msg = f"Invalid difference mode: {diff_mode}"
    raise ValueError(msg)


def make_diff_colourmap(
    sample: np.ndarray | float,
    *,
    mode: DiffMode,
) -> DiffColourmapSpec:
    """Construct colour mapping settings for a difference panel.

    Behaviour depends on the difference mode:

    - "signed": symmetric diverging scale centred on 0,
      useful for showing positive vs negative bias.
    - "absolute" / "smape": sequential scale from 0 to max,
      useful for showing error magnitude.

    Args:
        sample: Either a full array of differences (for precompute mode)
                or a scalar maximum difference (for two-pass mode).
        mode: Difference mode ("signed", "absolute", or "smape").

    Returns:
        DiffRenderParams: Normalisation, colour limits, and colourmap.

    """
    if mode == "signed":
        # Force symmetric limits around zero so 0 is the literal midpoint
        if isinstance(sample, (float, int)):
            max_abs = max(1.0, float(abs(sample)))
            vmin, vmax = -max_abs, max_abs
        else:
            # Find the min and max values of the sample array
            vmin_data = float(
                np.nanmin(sample) if np.nanmin(sample) is not None else -1.0
            )
            vmax_data = float(
                np.nanmax(sample) if np.nanmax(sample) is not None else 1.0
            )
            # Find the maximum absolute value of the sample array
            max_abs = max(1.0, abs(vmin_data), abs(vmax_data))
            vmin, vmax = -max_abs, max_abs

        return DiffColourmapSpec(
            norm=TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax),
            vmin=None,
            vmax=None,
            cmap="RdBu_r",
        )

    if mode in ("absolute", "smape"):
        # Positive-only scale
        if isinstance(sample, (float, int)):
            vmax = max(1e-6, float(sample))
        else:
            vmax = max(1e-6, float(np.nanmax(sample) or 0.0))

        return DiffColourmapSpec(
            norm=None,
            vmin=0.0,
            vmax=vmax,
            cmap="magma",
        )

    msg = f"Unknown difference mode: {mode}"
    raise ValueError(msg)


# ---- Range check for colourmap ----
"""
The range_check-report API has been moved to visualisations/range_check.py.
This module imports and re-exports the symbols for backward compatibility.
"""


# ---- Handling the difference stream ----


def prepare_difference_stream(
    *,
    include_difference: bool,
    diff_mode: DiffMode,
    strategy: DiffStrategy,
    ground_truth_stream: np.ndarray,
    prediction_stream: np.ndarray,
) -> tuple[np.ndarray | None, DiffColourmapSpec | None]:
    """General, reusable planner for animations (maps or time-series).

    This function implements three different strategies for handling
    difference between ground truth and prediction in animations, each with
    different memory and computational trade-offs. The choice of strategy affects
    both memory usage and animation performance.

    Args:
        include_difference: Whether difference visualisation is requested.
        diff_mode: Type of difference computation (signed/absolute/smape).
        strategy: Strategy for difference computation:
            - "precompute": Calculate all differences upfront
            - "two-pass": Scan data to determine colour scale, then compute per-frame
            - "per-frame": Compute differences on-demand
        ground_truth_stream: 3D array of ground truth data over time.
        prediction_stream: 3D array of prediction data over time.

    Returns:
        Tuple of (difference_stream, colour_scale):
        - difference_stream: None unless strategy == 'precompute'
        - colour_scale: DiffColourmapSpec describing colourmap/norm/range

    """
    if not include_difference:
        return None, None

    n_timesteps = ground_truth_stream.shape[0]
    if strategy == "precompute":
        difference_stream = compute_difference(
            ground_truth_stream, prediction_stream, diff_mode
        )
        colour_scale = make_diff_colourmap(difference_stream, mode=diff_mode)
        return difference_stream, colour_scale

    if strategy == "two-pass":
        if diff_mode == "signed":
            # find max |diff| without storing full stream
            max_abs = 0.0
            for tt in range(n_timesteps):
                difference = compute_difference(
                    ground_truth_stream[tt], prediction_stream[tt], "signed"
                )
                max_abs = max(max_abs, float(np.nanmax(np.abs(difference)) or 0.0))
            colour_scale = make_diff_colourmap(max_abs, mode="signed")
            return None, colour_scale
        # find max diff for sequential scale without storing full stream
        max_val = 0.0
        for tt in range(n_timesteps):
            difference = compute_difference(
                ground_truth_stream[tt], prediction_stream[tt], diff_mode
            )
            max_val = max(max_val, float(np.nanmax(difference) or 0.0))
        colour_scale = make_diff_colourmap(max_val, mode=diff_mode)
        return None, colour_scale

    if strategy == "per-frame":
        # no precomputation; each frame will call compute_difference and
        # may choose to infer its own params if desired (less consistent look)
        return None, None

    msg = f"Unknown DiffStrategy: {strategy}"
    raise ValueError(msg)


# --- Validation ---


def validate_2d_pair(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> tuple[int, int]:
    """Validate that both arrays are 2D [H,W] and have the same shape.

    Args:
        ground_truth: The ground truth array. [H,W]
        prediction: The prediction array. [H,W]

    Returns:
        The shape of the arrays. (H, W)

    Raises:
        InvalidArrayError: If the arrays are not 2D or have different shapes.

    """
    if not (ground_truth.ndim == prediction.ndim == 2):  # noqa: PLR2004
        msg = f"Expected 2D [H,W]; got ground truth={ground_truth.shape}, prediction={prediction.shape}"
        raise InvalidArrayError(msg)
    if ground_truth.shape != prediction.shape:
        msg = f"Shape mismatch: ground truth={ground_truth.shape}, prediction={prediction.shape}"
        raise InvalidArrayError(msg)
    return ground_truth.shape  # (H, W)


def validate_3d_streams(
    ground_truth_stream: np.ndarray,
    prediction_stream: np.ndarray,
) -> tuple[int, int, int]:
    """Validate that both arrays are 3D [T,H,W] and have the same shape.

    Args:
        ground_truth_stream: The ground truth array. [T,H,W]
        prediction_stream: The prediction array. [T,H,W]

    Returns:
        The shape of the arrays. (T,H,W)

    Raises:
        InvalidArrayError: If the arrays are not 2D or have different shapes.

    """
    if not (ground_truth_stream.ndim == prediction_stream.ndim == 3):  # noqa: PLR2004
        msg = f"Expected 3D [T,H,W]; got ground truth={ground_truth_stream.shape}, prediction={prediction_stream.shape}"
        raise InvalidArrayError(msg)
    if ground_truth_stream.shape != prediction_stream.shape:
        msg = f"Shape mismatch: ground truth={ground_truth_stream.shape}, prediction={prediction_stream.shape}"
        raise InvalidArrayError(msg)
    return ground_truth_stream.shape  # type: ignore[return-value]


def compute_display_ranges(
    ground_truth: np.ndarray, prediction: np.ndarray, plot_spec: PlotSpec
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute vmin/vmax for ground truth and prediction based on strategy.

    Args:
        ground_truth: The ground truth array. [H,W]
        prediction: The prediction array. [H,W]
        plot_spec: The plotting specification.

    Returns:
        The display ranges. (vmin, vmax)

    Raises:
        InvalidArrayError: If the arrays are not 2D or have different shapes.

    """
    if plot_spec.colourbar_strategy == "shared":
        # Both panels use spec vmin/vmax (current behavior)
        vmin = plot_spec.vmin if plot_spec.vmin is not None else 0.0
        vmax = plot_spec.vmax if plot_spec.vmax is not None else 1.0
        spec_range = (vmin, vmax)
        return spec_range, spec_range

    if plot_spec.colourbar_strategy == "separate":
        # Each panel uses its own data range
        groundtruth_range = (
            float(np.nanmin(ground_truth)),
            float(np.nanmax(ground_truth)),
        )
        prediction_range = (float(np.nanmin(prediction)), float(np.nanmax(prediction)))
        return groundtruth_range, prediction_range

    # Fallback - should not reach here but required for type checking
    vmin = plot_spec.vmin if plot_spec.vmin is not None else 0.0
    vmax = plot_spec.vmax if plot_spec.vmax is not None else 1.0
    spec_range = (vmin, vmax)
    return spec_range, spec_range


def compute_display_ranges_stream(
    ground_truth_stream: np.ndarray, prediction_stream: np.ndarray, plot_spec: PlotSpec
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute stable vmin/vmax for Ground Truth and Prediction over the entire video.

    Args:
        ground_truth_stream: The ground truth array. [T,H,W]
        prediction_stream: The prediction array. [T,H,W]
        plot_spec: The plotting specification.

    Returns:
        The display ranges. (vmin, vmax)

    Raises:
        InvalidArrayError: If the arrays are not 3D or have different shapes.

    """
    if plot_spec.colourbar_strategy == "shared":
        vmin = plot_spec.vmin if plot_spec.vmin is not None else 0.0
        vmax = plot_spec.vmax if plot_spec.vmax is not None else 1.0
        spec_range = (vmin, vmax)
        return spec_range, spec_range
    # "separate"
    groundtruth_min = float(np.nanmin(ground_truth_stream))
    groundtruth_max = float(np.nanmax(ground_truth_stream))
    prediction_min = float(np.nanmin(prediction_stream))
    prediction_max = float(np.nanmax(prediction_stream))
    return (groundtruth_min, groundtruth_max), (prediction_min, prediction_max)


def detect_land_mask_path(
    base_path: str | Path,
    dataset_name: str | None = None,
    hemisphere: str | None = None,
) -> str | None:
    """Automatically detect the land mask path based on dataset configuration.

    This function looks for land mask files in the expected locations based on
    the dataset name and hemisphere. It follows the pattern:
    - {base_path}/data/preprocessing/{dataset_name}/IceNetSIC/data/masks/{hemisphere}/masks/land_mask.npy
    - {base_path}/data/preprocessing/IceNetSIC/data/masks/{hemisphere}/masks/land_mask.npy

    Args:
        base_path: Base path to the data directory.
        dataset_name: Name of the dataset (e.g., 'samp-sicsouth-osisaf-25k-2017-2019-24h-v1').
        hemisphere: Hemisphere ('north' or 'south').

    Returns:
        Path to the land mask file if found, None otherwise.

    """
    base_path = Path(base_path)

    # Try to infer hemisphere from dataset name if not provided
    if hemisphere is None and dataset_name is not None:
        if "south" in dataset_name.lower():
            hemisphere = "south"
        elif "north" in dataset_name.lower():
            hemisphere = "north"

    if hemisphere is None:
        return None

    # Try dataset-specific path first
    if dataset_name is not None:
        dataset_specific_path = (
            base_path
            / "data"
            / "preprocessing"
            / dataset_name
            / "IceNetSIC"
            / "data"
            / "masks"
            / hemisphere
            / "masks"
            / "land_mask.npy"
        )
        if dataset_specific_path.exists():
            return str(dataset_specific_path)

    # Try general IceNetSIC path
    general_path = (
        base_path
        / "data"
        / "preprocessing"
        / "IceNetSIC"
        / "data"
        / "masks"
        / hemisphere
        / "masks"
        / "land_mask.npy"
    )
    if general_path.exists():
        return str(general_path)

    return None


def load_land_mask(
    land_mask_path: str | None, expected_shape: tuple[int, int]
) -> np.ndarray | None:
    """Load and validate a land mask from a numpy file.

    Args:
        land_mask_path: Path to the land mask .npy file. If None, returns None.
        expected_shape: Expected shape (height, width) of the land mask.

    Returns:
        Land mask array with shape (height, width) where True indicates land areas,
        or None if no land mask path is provided.

    Raises:
        InvalidArrayError: If the land mask file cannot be loaded or has wrong shape.

    """
    if land_mask_path is None:
        return None

    try:
        land_mask = np.load(land_mask_path)
    except (OSError, ValueError) as e:
        msg = f"Failed to load land mask from {land_mask_path}: {e}"
        raise InvalidArrayError(msg) from e

    if land_mask.ndim != EXPECTED_LAND_MASK_DIMENSIONS:
        msg = f"Land mask must be 2D, got shape {land_mask.shape}"
        raise InvalidArrayError(msg)

    if land_mask.shape != expected_shape:
        msg = f"Land mask shape {land_mask.shape} does not match expected shape {expected_shape}"
        raise InvalidArrayError(msg)

    # Convert to boolean if it's not already
    if land_mask.dtype != bool:
        land_mask = land_mask.astype(bool)

    return land_mask
