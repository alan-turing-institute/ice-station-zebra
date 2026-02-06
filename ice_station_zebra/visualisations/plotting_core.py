import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib as mpl
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

from ice_station_zebra.exceptions import InvalidArrayError
from ice_station_zebra.types import DiffColourmapSpec, DiffMode, DiffStrategy, PlotSpec

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

# Constants for land mask validation
EXPECTED_LAND_MASK_DIMENSIONS = 2


# --- Variable Styling ---


@dataclass
class VariableStyle:
    """Styling configuration for individual variables.

    Attributes:
        cmap: Matplotlib colourmap name (e.g., "viridis", "RdBu_r").
        colourbar_strategy: "shared" or "separate" (kept for compatibility).
        vmin: Minimum value for colour scale.
        vmax: Maximum value for colour scale.
        two_slope_centre: Centre value for diverging colourmap (TwoSlopeNorm).
        units: Display units for the variable (e.g., "K", "m/s").
        origin: Imshow origin override ("upper" keeps north-up, "lower" keeps south-up).
        decimals: Number of decimal places for colourbar tick labels (default: 2).
        use_scientific_notation: Whether to format colourbar tick labels in scientific notation (default: False).

    """

    cmap: str | None = None
    colourbar_strategy: str | None = None
    vmin: float | None = None
    vmax: float | None = None
    two_slope_centre: float | None = None
    units: str | None = None
    origin: Literal["upper", "lower"] | None = None
    decimals: int | None = None
    use_scientific_notation: bool | None = None


def colourmap_with_bad(
    cmap_name: str | None, bad_color: str = "#dcdcdc"
) -> mpl.colors.Colormap:
    """Create a colourmap copy with a specified color for bad (NaN) values.

    This function copies the specified colourmap and sets the 'bad' color to handle
    NaN values consistently, preventing white artifacts in visualisations.

    Args:
        cmap_name: Name of the matplotlib colourmap (e.g., "viridis", "RdBu_r").
                   If None, defaults to "viridis".
        bad_color: Color to use for NaN/bad values. Default is light grey (#dcdcdc).

    Returns:
        A copy of the colourmap with set_bad() configured.

    """
    if cmap_name is None:
        cmap = mpl.colormaps.get_cmap("viridis")
    else:
        cmap = mpl.colormaps.get_cmap(cmap_name)

    try:
        cmap = cmap.copy()
    except (AttributeError, TypeError):
        # Some matplotlib versions return non-copyable colourmap; create new
        cmap = mpl.colormaps.get_cmap(cmap.name)

    cmap.set_bad(bad_color)
    return cmap


def safe_nanmin(arr: np.ndarray, default: float = 0.0) -> float:
    """Safely compute nanmin with fallback for empty or all-NaN arrays.

    Args:
        arr: Array to compute minimum from.
        default: Default value if array is empty or all NaN.

    Returns:
        Minimum value or default.

    """
    if np.isfinite(arr).any():
        result = np.nanmin(arr)
        return float(result) if np.isfinite(result) else default
    return default


def safe_nanmax(arr: np.ndarray, default: float = 1.0) -> float:
    """Safely compute nanmax with fallback for empty or all-NaN arrays.

    Args:
        arr: Array to compute maximum from.
        default: Default value if array is empty or all NaN.

    Returns:
        Maximum value or default.

    """
    if np.isfinite(arr).any():
        result = np.nanmax(arr)
        return float(result) if np.isfinite(result) else default
    return default


def style_for_variable(  # noqa: C901, PLR0911
    var_name: str, styles: dict[str, dict[str, Any]] | None
) -> VariableStyle:
    """Return best matching style for a variable from config styles dict.

    Matching priority:
      1) exact key
      2) wildcard prefix key ending with '*'
      3) _default
      4) empty style
    Accepts any Mapping (so OmegaConf DictConfig works).
    """

    def _normalise_name(name: str) -> str:
        # Convert double-underscore to colon (this maps 'era5__2t' -> 'era5:2t')
        name = name.replace("__", ":")
        # Treat hyphens as separators too: 'era5-2t' -> 'era5:2t'
        name = name.replace("-", ":")
        # Collapse accidental repeated '::' to single ':'
        while "::" in name:
            name = name.replace("::", ":")
        # Keep single underscores (they are meaningful in some variable names)
        return name

    if not styles:
        return VariableStyle()

    # Accept Mapping-like configs (Dict, DictConfig, etc.)
    if not isinstance(styles, Mapping):
        logger.info("style_for_variable: styles is not a Mapping; ignoring styles")
        return VariableStyle()

    # Quick exact match first (try raw var_name)
    spec = styles.get(var_name)
    if isinstance(spec, Mapping):
        return VariableStyle(**{k: spec.get(k) for k in VariableStyle.__annotations__})

    # Try normalised exact match
    norm_var = _normalise_name(var_name)
    if norm_var != var_name:
        spec = styles.get(norm_var)
        if isinstance(spec, Mapping):
            return VariableStyle(
                **{k: spec.get(k) for k in VariableStyle.__annotations__}
            )

    # Wildcard prefix match: scan keys ending with '*' (normalise the key before comparing)
    # We iterate keys so keep original order (OmegaConf preserves insertion order).
    for key in styles:
        if isinstance(key, str) and key.endswith("*"):
            prefix = key[:-1]
            prefix_norm = _normalise_name(prefix)
            # If prefix_norm is empty (user wrote '*' only) skip it
            if not prefix_norm:
                continue
            # Compare against both raw and normalised var names
            if var_name.startswith(prefix) or norm_var.startswith(prefix_norm):
                spec = styles.get(key)
                if isinstance(spec, Mapping):
                    return VariableStyle(
                        **{k: spec.get(k) for k in VariableStyle.__annotations__}
                    )
                logger.info(
                    "style_for_variable: wildcard candidate %r not a dict (type=%s)",
                    key,
                    type(spec),
                )

    # Fallback to _default
    spec = styles.get("_default")
    if isinstance(spec, Mapping):
        return VariableStyle(**{k: spec.get(k) for k in VariableStyle.__annotations__})

    return VariableStyle()


# --- Colour Scale Generation ---


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


def create_normalisation(
    data: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    centre: float | None = None,
) -> tuple[Normalize | TwoSlopeNorm, float, float]:
    """Create appropriate normalisation for data with optional centring.

    This function creates either a linear Normalize or a diverging TwoSlopeNorm
    based on whether a centre value is provided. When a centre is specified,
    the normalisation will be symmetric around that value.

    Args:
        data: 2D array of data to normalise.
        vmin: Minimum value for colour scale. If None, inferred from data.
        vmax: Maximum value for colour scale. If None, inferred from data.
        centre: Centre value for diverging colourmap. If provided, creates
            a symmetric TwoSlopeNorm around this value.

    Returns:
        Tuple of (normalisation, vmin, vmax) where:
        - normalisation: Normalize or TwoSlopeNorm object
        - vmin: Computed minimum value
        - vmax: Computed maximum value

    """
    # Compute data range with robust handling of NaN/inf
    data_min = float(np.nanmin(data)) if np.isfinite(data).any() else 0.0
    data_max = float(np.nanmax(data)) if np.isfinite(data).any() else 1.0

    if centre is not None:
        # Diverging colourmap centred at specified value
        low = vmin if vmin is not None else data_min
        high = vmax if vmax is not None else data_max

        # Make symmetric around the centre where possible
        span_low = abs(centre - low)
        span_high = abs(high - centre)
        span = max(span_low, span_high, 1e-6)

        final_vmin = float(centre - span)
        final_vmax = float(centre + span)

        norm: Normalize | TwoSlopeNorm = TwoSlopeNorm(
            vmin=final_vmin, vcenter=float(centre), vmax=final_vmax
        )
        return norm, final_vmin, final_vmax

    # Linear colourmap
    final_vmin = float(vmin if vmin is not None else data_min)
    final_vmax = float(vmax if vmax is not None else data_max)

    norm_linear: Normalize | TwoSlopeNorm = Normalize(vmin=final_vmin, vmax=final_vmax)
    return norm_linear, final_vmin, final_vmax


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
            # Find the min and max values of the sample array using safe helpers
            vmin_data = safe_nanmin(sample, default=-1.0)
            vmax_data = safe_nanmax(sample, default=1.0)
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
            vmax = max(1e-6, safe_nanmax(sample, default=0.0))

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


def detect_land_mask_path(  # noqa: C901
    base_path: str | Path,
    dataset_name: str | None = None,
    hemisphere: str | None = None,
) -> str | None:
    """Automatically detect the land mask path based on dataset configuration.

    This function looks for land mask files in the expected locations based on
    the dataset name and hemisphere. It accepts either the repository root
    (containing a ``data`` directory) or the ``data`` directory itself as
    ``base_path``. It follows the pattern:
    - {base_path}/data/preprocessing/{dataset_name}/IceNetSIC/data/masks/{hemisphere}/masks/land_mask.npy
    - {base_path}/data/preprocessing/IceNetSIC/data/masks/{hemisphere}/masks/land_mask.npy

    Args:
        base_path: Base path to the project root **or** the ``data`` directory.
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

    # Support repo-root, data-directory, and nested data/data layouts
    candidate_data_roots: list[Path] = []
    if base_path.name != "data":
        candidate_data_roots.append(base_path / "data")
    candidate_data_roots.append(base_path)

    seen_roots: set[Path] = set()
    for data_root in candidate_data_roots:
        resolved_root = data_root.resolve()
        if resolved_root in seen_roots:
            continue
        seen_roots.add(resolved_root)

        # Some deployments keep datasets in data/data/,
        # so we probe both the root and an extra nested data/ layer.
        preprocessing_roots = {
            resolved_root / "preprocessing",
            resolved_root / "data" / "preprocessing",
        }

        for preproc_root in preprocessing_roots:
            # Try dataset-specific path first
            if dataset_name is not None:
                dataset_specific_path = (
                    preproc_root
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
                preproc_root
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


# --- File Utilities ---


def safe_filename(name: str) -> str:
    """Sanitise a string for use as a filename.

    Replaces non-alphanumeric characters (except hyphens and underscores)
    with hyphens and strips leading/trailing hyphens.

    Args:
        name: Input string to sanitise.

    Returns:
        Sanitised filename string.

    Examples:
        >>> safe_filename("era5:2t")
        'era5-2t'
        >>> safe_filename("My Variable Name!")
        'My-Variable-Name'

    """
    keep = [c if c.isalnum() or c in ("-", "_") else "-" for c in name.strip()]
    return "".join(keep).strip("-") or "var"


def save_figure(
    fig: "plt.Figure", save_dir: Path | None, base_name: str
) -> Path | None:
    """Save a matplotlib figure to disk as PNG.

    Creates the save directory if it doesn't exist. Sanitises the filename
    to avoid filesystem issues.

    Args:
        fig: Matplotlib Figure object to save.
        save_dir: Directory to save the figure in. If None, does not save.
        base_name: Base name for the file (will be sanitised).

    Returns:
        Path to the saved file, or None if save_dir was None.

    """
    if save_dir is None:
        return None

    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{safe_filename(base_name)}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
