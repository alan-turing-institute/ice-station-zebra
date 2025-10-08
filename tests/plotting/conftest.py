import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Protocol, cast

import hydra
import matplotlib as mpl
import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf import errors as oc_errors

from ice_station_zebra.data_loaders import ZebraDataModule
from tests.conftest import make_varying_sic_stream

mpl.use("Agg")

# Suppress Matplotlib animation warning during tests; we intentionally do not keep
# long-lived references to animation objects beyond saving to buffer.
warnings.filterwarnings(
    "ignore",
    message="Animation was deleted without rendering anything",
    category=UserWarning,
)

TEST_DATE = date(2020, 1, 15)


@pytest.fixture
def sic_pair_2d(
    sic_pair_3d_stream: tuple[np.ndarray, np.ndarray, list[date]],
) -> tuple[np.ndarray, np.ndarray, date]:
    """Extract a single frame from the 3D stream for static plots.

    Returns the first timestep from the stream as 2D arrays.
    """
    ground_truth_stream, prediction_stream, dates = sic_pair_3d_stream
    return ground_truth_stream[0], prediction_stream[0], dates[0]


@pytest.fixture
def sic_pair_warning_2d() -> tuple[np.ndarray, np.ndarray, date]:
    """Construct arrays that should trigger sanity-report warnings.

    Ground truth stays in [0,1]. Prediction has a stripe with values > 1.5 to
    ensure >5% of values are outside the display range when using shared 0..1.
    """
    height, width = 64, 64
    gt = np.clip(np.random.default_rng(42).random((height, width)), 0.0, 1.0).astype(
        np.float32
    )
    pred = gt.copy()
    # Make a vertical stripe out-of-range ~25% of pixels
    stripe_cols = slice(width // 4, width // 2)
    pred[:, stripe_cols] = 1.6
    return gt, pred, TEST_DATE


class MakeCircularArctic(Protocol):
    def __call__(
        self,
        height: int,
        width: int,
        *,
        rng: np.random.Generator,
        ring_width: int = ...,
        noise: float = ...,
    ) -> np.ndarray: ...


def make_central_distance_grid(height: int, width: int) -> np.ndarray:
    """Return per-pixel Euclidean distance from the image centre.

    The centre is defined as ((H-1)/2, (W-1)/2) so that distances are symmetric
    for even-sized grids.
    """
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)


@pytest.fixture
def sic_pair_3d_stream(
    make_circular_arctic: MakeCircularArctic,
) -> tuple[np.ndarray, np.ndarray, list[date]]:
    """Short 3D streams (time, height, width) and dates for animations.

    Shape (4, 48, 48), values in [0, 1]. Frames drift slightly over time
    with a bit of noise to mimic day-to-day change.
    """
    rng = np.random.default_rng(100)
    timesteps, height, width = 4, 48, 48

    dist = make_central_distance_grid(height, width)

    coastline_base_radius = min(height, width) * 0.25

    # Generate ground truth stream using oscillating sea-ice radius.
    ground_truth_stream = make_varying_sic_stream(
        dist_grid=dist,
        timesteps=timesteps,
        base_radius=coastline_base_radius,
        rng=rng,
        ring_width=6.0,
        noise_std=0.03,
        radius_oscillation_amplitude=0.5,
        radius_oscillation_frequency=0.7,
    )

    # Prediction: same circle shape as ground truth, but randomised ice distribution noise
    prediction_frames = []
    for _ in range(timesteps):
        prediction_t = make_circular_arctic(height, width, rng=rng, noise=0.08)
        prediction_frames.append(prediction_t.astype(np.float32))

    prediction_stream = np.stack(prediction_frames, axis=0)
    dates = [TEST_DATE + timedelta(days=int(d)) for d in range(timesteps)]
    return ground_truth_stream, prediction_stream, dates


@pytest.fixture
def checkpoint_data(
    *, use_checkpoint: bool, example_checkpoint_path: Path | None
) -> tuple[np.ndarray, np.ndarray, date] | None:
    """Load data from a checkpoint for plotting tests.

    Returns (ground_truth, prediction, date) if checkpoint is available, else None.
    """
    if not use_checkpoint or example_checkpoint_path is None:
        return None

    try:
        # Load config (try checkpoint dir first, fallback to default)
        config_path = example_checkpoint_path.parent.parent / "model_config.yaml"
        if config_path.exists():
            config = cast("DictConfig", OmegaConf.load(config_path))
        else:
            # Fallback to default config
            # Look for a file ending with local.yaml
            config_dir = Path("ice_station_zebra/config/")
            yaml_iter = config_dir.glob("*.local.yaml")
            local_yaml = next(yaml_iter)
            config = cast("DictConfig", OmegaConf.load(local_yaml))

        # Load model from checkpoint
        model_dict = cast(
            "DictConfig", config["model"]
        )  # ensure DictConfig for nested access
        model_target = cast("str", model_dict["_target_"])
        model_cls = cast("Any", hydra.utils.get_class(model_target))
        model = model_cls.load_from_checkpoint(checkpoint_path=example_checkpoint_path)
        model.eval()

        # Load data module
        # Ensure DictConfig type for constructor
        dm_config = cast("DictConfig", config)
        data_module = ZebraDataModule(dm_config)
        data_module.setup("test")
        test_dataloader = data_module.test_dataloader()

        # Get a single batch
        batch = next(iter(test_dataloader))

        # Run model inference
        with torch.no_grad():
            target = batch.pop("target")
            prediction = model(batch)

        # Extract first sample, first timestep, first channel -> [H, W]
        ground_truth = target[0, 0, 0].detach().cpu().numpy()
        prediction_array = prediction[0, 0, 0].detach().cpu().numpy()

        # Get date
        current_date = TEST_DATE

    except (
        FileNotFoundError,
        KeyError,
        AttributeError,
        RuntimeError,
        ValueError,
        StopIteration,
        ImportError,
        OSError,
        oc_errors.OmegaConfBaseException,
    ) as e:
        pytest.skip(f"Could not load checkpoint data: {e}")
    else:
        return ground_truth, prediction_array, current_date


@pytest.fixture
def plotting_data(
    checkpoint_data: tuple[np.ndarray, np.ndarray, date] | None,
    sic_pair_2d: tuple[np.ndarray, np.ndarray, date],
) -> tuple[np.ndarray, np.ndarray, date]:
    """Choose between checkpoint data or fake data data for plotting tests.

    Returns checkpoint data if available and --use-checkpoint is set, otherwise fake data data.
    """
    if checkpoint_data is not None:
        return checkpoint_data
    return sic_pair_2d


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command-line options for plotting tests.

    --use-checkpoint: prefer loading a model checkpoint instead of fake data data
    --checkpoint-path: explicit path to a checkpoint (.ckpt)
    """
    parser.addoption(
        "--use-checkpoint",
        action="store_true",
        default=False,
        help="Use a model checkpoint instead of fake data SIC arrays.",
    )
    parser.addoption(
        "--checkpoint-path",
        action="store",
        default="",
        help="Path to a .ckpt file to use in plotting tests.",
    )


@pytest.fixture
def use_checkpoint(pytestconfig: pytest.Config) -> bool:
    """Whether tests should use a checkpoint instead of fake data data."""
    return bool(pytestconfig.getoption("--use-checkpoint"))


@pytest.fixture
def example_checkpoint_path(pytestconfig: pytest.Config) -> Path | None:
    """Return a checkpoint path if available, else None.

    Resolution order:
    1) --checkpoint-path if provided and exists
    2) tests/plotting/checkpoints/example.ckpt if present in repo
    3) data/training/wandb/.../checkpoints/*.ckpt (first found) [optional scan]
    """
    # 1) explicit
    arg_path = str(pytestconfig.getoption("--checkpoint-path") or "").strip()
    if arg_path:
        p = Path(arg_path).expanduser().resolve()
        return p if p.exists() else None

    # 2) bundled example (recommended to add to repo if you want deterministic tests)
    bundled = Path(__file__).parent / "checkpoints" / "example.ckpt"
    if bundled.exists():
        return bundled.resolve()

    # 3) optional: try a typical wandb path if present locally
    default_wandb = Path(__file__).resolve().parents[2] / "data" / "training" / "wandb"
    if default_wandb.exists():
        for ckpt in default_wandb.rglob("*.ckpt"):
            return ckpt.resolve()

    return None
