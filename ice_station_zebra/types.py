from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Any, Literal, NamedTuple, Self, TypedDict

from jaxtyping import Float
from matplotlib.colors import Normalize
from numpy import float32
from numpy.typing import NDArray
from omegaconf import DictConfig
from torch import Tensor

ArrayCHW = Float[NDArray[float32], "channels height width"]
ArrayTCHW = Float[NDArray[float32], "time channels height width"]
TensorNCHW = Float[Tensor, "batch channels height width"]
TensorNTCHW = Float[Tensor, "batch time channels height width"]

# DiffMode → what you compute
# - "signed": target - prediction (can be ±, so symmetric colour scale around 0)
# - "absolute": |target - prediction| (≥ 0, sequential scale)
# - "smape": |pred - target| / ((|pred|+|target|)/2) (≥ 0, sequential scale)
DiffMode = Literal["signed", "absolute", "smape"]

# DiffStrategy → when you compute (for animations)
# - "precompute": compute full diff stream once (fast playback, more RAM)
# - "two-pass": scan once to figure the scale, compute per-frame (balanced)
# - "per-frame": compute per-frame (low RAM, more CPU)
DiffStrategy = Literal["precompute", "two-pass", "per-frame"]


class BetaSchedule(StrEnum):
    LINEAR = "linear"
    COSINE = "cosine"


class TensorDimensions(IntEnum):
    """Enum for tensor dimensions used throughout the codebase."""

    THW = 3  # Time, Height, Width
    BTCHW = 5  # Batch, Time, Channels, Height, Width


@dataclass
class AnemoiCreateArgs:
    config: DictConfig
    path: str
    command: str = "unused"
    overwrite: bool = False
    processes: int = 0
    threads: int = 0


@dataclass
class AnemoiInspectArgs:
    detailed: bool
    path: str
    progress: bool
    size: bool
    statistics: bool


class DataloaderArgs(TypedDict):
    batch_size: int
    sampler: None
    batch_sampler: None
    drop_last: bool
    worker_init_fn: None


class DataSpace:
    channels: int
    name: str
    shape: tuple[int, int]

    def __init__(self, channels: int, name: str, shape: Sequence[int]) -> None:
        """Initialise a DataSpace from channels, name and shape."""
        self.channels = int(channels)
        self.name = name
        self.shape = (int(shape[0]), int(shape[1]))

    @property
    def chw(self) -> tuple[int, int, int]:
        """Return a tuple of [channels, height, width]."""
        return (self.channels, *self.shape)

    @classmethod
    def from_dict(cls, config: DictConfig | dict[str, Any]) -> Self:
        return cls(
            channels=config["channels"], name=config["name"], shape=config["shape"]
        )

    def to_dict(self) -> DictConfig:
        """Return the DataSpace as a DictConfig."""
        return DictConfig(
            {"channels": self.channels, "name": self.name, "shape": self.shape}
        )


class DiffColourmapSpec(NamedTuple):
    """Colour scale specification for visualising a difference panel.

    Attributes:
        norm: Normalisation object for mapping data values to colours.
              Typically TwoSlopeNorm for signed differences (centred at 0).
              None for non-signed modes (absolute, smape).
        vmin: Lower bound for colour scale (used if norm is None).
        vmax: Upper bound for colour scale (used if norm is None).
        cmap: Name of the matplotlib colourmap to use.

    """

    norm: Normalize | None
    vmin: float | None
    vmax: float | None
    cmap: str


@dataclass
class ModelTestOutput(Mapping[str, Tensor]):
    """Output of a model test step."""

    prediction: TensorNTCHW
    target: TensorNTCHW
    loss: Tensor

    def __getitem__(self, key: str) -> Tensor:
        """Get a tensor by key."""
        if key == "prediction":
            return self.prediction
        if key == "target":
            return self.target
        if key == "loss":
            return self.loss
        msg = f"Key {key} not found in ModelTestOutput"
        raise KeyError(msg)

    def __iter__(self) -> Iterator[str]:
        """Iterate over the keys of ModelTestOutput."""
        yield "prediction"
        yield "target"
        yield "loss"

    def __len__(self) -> int:
        """Return ModelTestOutput length."""
        return 3


@dataclass(frozen=True)
class PlotSpec:
    variable: str
    title_groundtruth: str = "Ground Truth"
    title_prediction: str = "Prediction"
    title_difference: str = "Difference"
    n_contour_levels: int = 100
    colourmap: str = "viridis"
    include_difference: bool = True
    diff_mode: DiffMode = "signed"
    diff_strategy: DiffStrategy = "precompute"
    selected_timestep: int = 0
    vmin: float | None = 0.0
    vmax: float | None = 1.0
    colourbar_location: Literal["vertical", "horizontal"] = "vertical"
    colourbar_strategy: Literal["shared", "separate"] = "shared"
