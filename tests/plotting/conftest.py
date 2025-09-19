from datetime import date, timedelta

import numpy as np
import pytest


def _make_circular_arctic(
    height: int,
    width: int,
    *,
    rng: np.random.Generator,
    ring_width: int = 6,
    noise: float = 0.05,
) -> np.ndarray:
    """Make a simple sea-ice map using a circle in the centre.

    - Inside the circle: land (set to 0 but should be NaN eventually)
    - Along the circle edge: high ice
    - Further outside: fades down to 0 (open water)
    """
    # Create a grid of distances from the centre
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # Choose a radius so the land takes most of the centre
    radius = min(height, width) * 0.25

    # Distance from the coastline (ring at the circle): 0 on the ring, >0 outside
    d_outside = np.maximum(0.0, dist - radius)

    # Ice strength: 1 at the ring, then smoothly falls to 0 with distance
    # Use a gentle exponential falloff controlled by ring_width
    falloff = np.exp(-(d_outside / max(1.0, float(ring_width))))

    # Add a tiny bit of texture so the ring looks less perfect
    texture = rng.normal(0.0, noise, size=(height, width))
    sic = np.clip(falloff + texture, 0.0, 1.0)

    # Land mask: everything strictly inside the circle is NaN
    # Note: When land masking is implemented, uncomment the line below and remove the 0.0 line
    # Should be: sic[dist < radius] = np.nan
    sic[dist < radius] = (
        0.0  # Temporary: set land to 0 SIC to match current code behavior
    )
    return sic.astype(np.float32)


@pytest.fixture
def sic_pair_2d() -> tuple[np.ndarray, np.ndarray, date]:
    """Small 2D ground-truth/prediction arrays and a date for static plots.

    Shape (48, 48), values in [0, 1]. Prediction is ground truth with a
    different ice distribution noise but same circle shape.
    """
    rng = np.random.default_rng(123)
    height, width = 48, 48
    ground_truth = _make_circular_arctic(height, width, rng=rng)

    # Prediction: same circle shape as ground truth, but different ice distribution noise
    prediction = _make_circular_arctic(
        height, width, rng=rng, noise=0.08
    )  # Different noise level

    current_date = date(2020, 1, 16)
    return ground_truth.astype(np.float32), prediction.astype(np.float32), current_date


@pytest.fixture
def sic_pair_3d_stream() -> tuple[np.ndarray, np.ndarray, list[date]]:
    """Short 3D streams (time, height, width) and dates for animations.

    Shape (4, 48, 48), values in [0, 1]. Frames drift slightly over time
    with a bit of noise to mimic day-to-day change.
    """
    rng = np.random.default_rng(54321)
    timesteps, height, width = 4, 48, 48

    groundtruth_frames = []
    prediction_frames = []
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    base_radius = min(height, width) * 0.25
    for t in range(timesteps):
        # Ground truth: small daily wiggle of the coastline
        radius_t = base_radius + 0.5 * np.sin(0.7 * t)
        d_outside_t = np.maximum(0.0, dist - radius_t)
        groundtruth_t = np.exp(-(d_outside_t / 6.0)) + rng.normal(
            0.0, 0.03, size=(height, width)
        )
        groundtruth_t = np.clip(groundtruth_t, 0.0, 1.0)
        # Note: When land masking is implemented, uncomment the line below and remove the 0.0 line
        # Sould be:groundtruth_t[dist < radius_t] = np.nan
        groundtruth_t[dist < radius_t] = (
            0.0  # Temporary: set land to 0 SIC to match current code behavior
        )

        # Prediction: same circle shape as ground truth, but different ice distribution noise
        prediction_t = _make_circular_arctic(
            height, width, rng=rng, noise=0.08
        )  # Different noise level

        groundtruth_frames.append(groundtruth_t.astype(np.float32))
        prediction_frames.append(prediction_t.astype(np.float32))

    ground_truth_stream = np.stack(groundtruth_frames, axis=0)
    prediction_stream = np.stack(prediction_frames, axis=0)
    dates = [date(2020, 1, 15) + timedelta(days=int(d)) for d in range(timesteps)]
    return ground_truth_stream, prediction_stream, dates
