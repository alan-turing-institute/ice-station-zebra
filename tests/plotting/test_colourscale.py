from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import matplotlib as mpl
import pytest

if TYPE_CHECKING:
    # Imports used only for type annotations
    from datetime import date

    import numpy as np

mpl.use("Agg")

from ice_station_zebra.visualisations.plotting_core import (
    compute_display_ranges,
    make_diff_colourmap,
)
from ice_station_zebra.visualisations.plotting_maps import DEFAULT_SIC_SPEC


@pytest.mark.parametrize(
    "colourbar_strategy", ["shared", "separate"], ids=lambda s: f"cbar-{s}"
)
def test_sic_display_range_respects_strategy(
    sic_pair_2d: tuple[np.ndarray, np.ndarray, date],
    colourbar_strategy: Literal["shared", "separate"],
) -> None:
    """Test that the display range respects the colourbar strategy."""
    ground_truth, prediction, _ = sic_pair_2d
    plot_spec = replace(DEFAULT_SIC_SPEC, colourbar_strategy=colourbar_strategy)
    (gt_vmin, gt_vmax), (p_vmin, p_vmax) = compute_display_ranges(
        ground_truth, prediction, plot_spec
    )
    if colourbar_strategy == "shared":
        assert (gt_vmin, gt_vmax) == (0.0, 1.0), (
            "Shared colourbar strategy should have vmin=0.0 and vmax=1.0 for Ground Truth"
        )
        assert (p_vmin, p_vmax) == (0.0, 1.0), (
            "Shared colourbar strategy should have vmin=0.0 and vmax=1.0 for Prediction"
        )
    else:
        assert gt_vmin <= gt_vmax, (
            "Separate colourbar strategy should have vmin<=vmax for Ground Truth"
        )
        assert p_vmin <= p_vmax, (
            "Separate colourbar strategy should have vmin<=vmax for Prediction"
        )


@pytest.mark.parametrize(
    "diff_mode", ["signed", "absolute", "smape"], ids=lambda s: f"diff-{s}"
)
def test_difference_colour_scale_modes(
    sic_pair_2d: tuple[np.ndarray, np.ndarray, date],
    diff_mode: Literal["signed", "absolute", "smape"],
) -> None:
    """Test that the difference colour scale mode is correct."""
    ground_truth, prediction, _ = sic_pair_2d
    plot_spec = make_diff_colourmap(ground_truth - prediction, mode=diff_mode)
    if diff_mode == "signed":
        assert plot_spec.norm is not None, "Signed difference mode should have a norm"
        assert getattr(plot_spec.norm, "vcenter", None) == 0.0, (
            "Signed difference mode should have vcenter=0.0"
        )
    else:
        assert plot_spec.norm is None, "Difference mode should have a norm"
        assert plot_spec.vmin == 0.0, "Difference mode should have vmin=0.0"
        assert (plot_spec.vmax or 0) >= 0.0, "Difference mode should have vmax>=0.0"
