from datetime import date

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
