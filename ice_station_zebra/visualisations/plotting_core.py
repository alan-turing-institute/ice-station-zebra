import numpy as np
from matplotlib.colors import TwoSlopeNorm
from dataclasses import dataclass

from ice_station_zebra.exceptions import InvalidArrayError
from ice_station_zebra.types import DiffColourmapSpec, DiffMode, DiffStrategy, PlotSpec


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
            vmin_data = float(
                np.nanmin(sample) if np.nanmin(sample) is not None else -1.0
            )
            vmax_data = float(
                np.nanmax(sample) if np.nanmax(sample) is not None else 1.0
            )
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


# ---- Sanity check for colourmap ----

@dataclass(frozen=True)
class SanityReport:
    """Compact sanity report: ranges and fraction outside display range."""

    groundtruth_min: float
    groundtruth_max: float
    prediction_min: float
    prediction_max: float

    outside_fraction_groundtruth: float
    outside_fraction_prediction: float

    warnings: tuple[str, ...]

def _frac_outside(data: np.ndarray, *, vmin: float, vmax: float) -> float:
    """Return fraction of finite values outside [vmin, vmax]."""
    data = data.astype(float)
    finite = np.isfinite(data)
    n = int(np.count_nonzero(finite))
    if n == 0:
        return 0.0
    values = data[finite]
    return float(np.count_nonzero((values < vmin) | (values > vmax))) / n

def _robust_p2_p98(data: np.ndarray) -> tuple[float, float]:
    data = data.astype(float)
    finite = np.isfinite(data)
    if not np.any(finite):
        return 0.0, 0.0
    values = data[finite]
    return float(np.percentile(values, 2)), float(np.percentile(values, 98))

def compute_sanity_report(
    groundtruth: np.ndarray,
    prediction: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    outside_warn: float = 0.05,
    severe_outside: float = 0.20,
    include_shared_range_mismatch_check: bool = True,
) -> SanityReport:
    """Clarity-first sanity using a single display range [vmin, vmax].

    Reports actual numeric ranges and the fraction of values outside [vmin, vmax]
    for ground truth and prediction, plus optional magnitude mismatch nudges.
    """
    # Numeric ranges
    groundtruth_min = float(np.nanmin(groundtruth)) if np.isfinite(np.nanmin(groundtruth)) else float("nan")
    groundtruth_max = float(np.nanmax(groundtruth)) if np.isfinite(np.nanmax(groundtruth)) else float("nan")
    prediction_min = float(np.nanmin(prediction)) if np.isfinite(np.nanmin(prediction)) else float("nan")
    prediction_max = float(np.nanmax(prediction)) if np.isfinite(np.nanmax(prediction)) else float("nan")

    # Fractions outside display
    outside_fraction_groundtruth = _frac_outside(groundtruth, vmin=vmin, vmax=vmax)
    outside_fraction_prediction = _frac_outside(prediction, vmin=vmin, vmax=vmax)

    warnings_list: list[str] = []

    # 1) Outside-range warnings
    if outside_fraction_prediction > severe_outside:
        warnings_list.append(
            f"Prediction range {prediction_min:.2f}-{prediction_max:.2f} lies far outside display [{vmin:.2f}, {vmax:.2f}] "
            f"({outside_fraction_prediction:.0%} of values clipped)."
        )
    elif outside_fraction_prediction > outside_warn:
        warnings_list.append(
            f"Prediction has values outside display [{vmin:.2f}, {vmax:.2f}] "
            f"({outside_fraction_prediction:.0%} clipped; range {prediction_min:.2f}-{prediction_max:.2f})."
        )

    if outside_fraction_groundtruth > severe_outside:
        warnings_list.append(
            f"Ground truth range {groundtruth_min:.2f}-{groundtruth_max:.2f} lies far outside display [{vmin:.2f}, {vmax:.2f}] "
            f"({outside_fraction_groundtruth:.0%} of values clipped)."
        )
    elif outside_fraction_groundtruth > outside_warn:
        warnings_list.append(
            f"Ground truth has values outside display [{vmin:.2f}, {vmax:.2f}] "
            f"({outside_fraction_groundtruth:.0%} clipped; range {groundtruth_min:.2f}-{groundtruth_max:.2f})."
        )

    # 2) Optional shared-range mismatch nudge
    if include_shared_range_mismatch_check:
        span = vmax - vmin
        if span > 0 and np.isfinite(span):
            gt_p2, gt_p98 = _robust_p2_p98(groundtruth)
            pr_p2, pr_p98 = _robust_p2_p98(prediction)
            if (pr_p98 < vmin + 0.05 * span) and (gt_p2 > vmin + 0.15 * span):
                warnings_list.append(
                    "Prediction magnitude appears much lower than ground truth under the shared scale "
                    f"(pred 2-98%: {pr_p2:.2f}-{pr_p98:.2f}, gt 2-98%: {gt_p2:.2f}-{gt_p98:.2f})."
                )
            if (pr_p2 > vmax - 0.05 * span) and (gt_p98 < vmax - 0.15 * span):
                warnings_list.append(
                    "Prediction magnitude appears much higher than ground truth under the shared scale "
                    f"(pred 2-98%: {pr_p2:.2f}-{pr_p98:.2f}, gt 2-98%: {gt_p2:.2f}-{gt_p98:.2f})."
                )

    return SanityReport(
        groundtruth_min=groundtruth_min,
        groundtruth_max=groundtruth_max,
        prediction_min=prediction_min,
        prediction_max=prediction_max,
        outside_fraction_groundtruth=outside_fraction_groundtruth,
        outside_fraction_prediction=outside_fraction_prediction,
        warnings=tuple(warnings_list),
    )


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
