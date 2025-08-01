from collections.abc import Sequence
from typing import Any, Self, TypedDict
from torch import Tensor

from omegaconf import DictConfig

LightningBatch = tuple[Tensor, Tensor]


class DataloaderArgs(TypedDict):
    batch_size: int
    sampler: None
    batch_sampler: None
    drop_last: bool
    worker_init_fn: None


class DataSpace:
    channels: int
    shape: tuple[int, int]

    def __init__(self, channels: int, shape: Sequence[int]) -> None:
        self.channels = int(channels)
        self.shape = (int(shape[0]), int(shape[1]))

    @classmethod
    def from_dict(cls, config: DictConfig | dict[str, Any]) -> Self:
        return cls(channels=config["channels"], shape=config["shape"])

    def to_dict(self) -> DictConfig:
        """Return the DataSpace as a DictConfig."""
        return DictConfig({"channels": self.channels, "shape": self.shape})
