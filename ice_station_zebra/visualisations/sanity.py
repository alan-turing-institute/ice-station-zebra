from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SanityReport:
    """Summarise basic range checks for ground truth and prediction.

    Attributes:
        groundtruth_min: Minimum value observed in the ground truth array.
        groundtruth_max: Maximum value observed in the ground truth array.
        prediction_min: Minimum value observed in the prediction array.
        prediction_max: Maximum value observed in the prediction array.
        outside_fraction_groundtruth: Fraction of finite ground truth values outside the display range.
        outside_fraction_prediction: Fraction of finite prediction values outside the display range.
        warnings: Tuple of short, user-facing warning strings suitable for a plot badge.

    """

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
    """Estimate a robust central band using the 2nd and 98th percentiles.

    This provides a quick sense of the typical magnitude while remaining
    tolerant to heavy tails and a modest amount of outliers.
    """
    data = data.astype(float)
    finite = np.isfinite(data)
    if not np.any(finite):
        return 0.0, 0.0
    values = data[finite]
    return float(np.percentile(values, 2)), float(np.percentile(values, 98))


def compute_sanity_report(  # noqa: PLR0913
    groundtruth: np.ndarray,
    prediction: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    outside_warn: float = 0.05,
    severe_outside: float = 0.20,
    include_shared_range_mismatch_check: bool = True,
) -> SanityReport:
    """Produce a concise sanity report under a shared display range.

    Uses a single display interval [vmin, vmax] to:
    - Report numeric minima and maxima for both arrays
    - Compute fractions of values falling outside the display interval
    - Optionally add gentle nudges when magnitudes appear mismatched

    Args:
        groundtruth: Ground-truth data array.
        prediction: Prediction data array.
        vmin: Lower bound of the shared display range.
        vmax: Upper bound of the shared display range.
        outside_warn: Threshold for issuing a soft "outside range" warning.
        severe_outside: Higher threshold that triggers a stronger warning.
        include_shared_range_mismatch_check: If True, add hints when the prediction
            appears systematically lower or higher than ground truth under the shared scale.

    Returns:
        A `SanityReport` containing basic ranges, outside fractions, and short warnings
        suitable for rendering in a figure badge.

    """
    # Numeric ranges
    groundtruth_min = (
        float(np.nanmin(groundtruth))
        if np.isfinite(np.nanmin(groundtruth))
        else float("nan")
    )
    groundtruth_max = (
        float(np.nanmax(groundtruth))
        if np.isfinite(np.nanmax(groundtruth))
        else float("nan")
    )
    prediction_min = (
        float(np.nanmin(prediction))
        if np.isfinite(np.nanmin(prediction))
        else float("nan")
    )
    prediction_max = (
        float(np.nanmax(prediction))
        if np.isfinite(np.nanmax(prediction))
        else float("nan")
    )

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
