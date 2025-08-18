from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self, TypedDict

from jaxtyping import Float
from numpy import float32
from numpy.typing import NDArray
from omegaconf import DictConfig
from torch import Tensor

ArrayCHW = Float[NDArray[float32], "channels height width"]
ArrayTCHW = Float[NDArray[float32], "time channels height width"]
TensorNCHW = Float[Tensor, "batch channels height width"]
TensorNTCHW = Float[Tensor, "batch time channels height width"]


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
