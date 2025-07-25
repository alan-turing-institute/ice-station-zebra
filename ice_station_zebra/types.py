from collections.abc import Iterable
from typing import TypedDict

from omegaconf import DictConfig


class DataloaderArgs(TypedDict):
    batch_size: int
    sampler: None
    batch_sampler: None
    drop_last: bool
    worker_init_fn: None


class DataSpace:
    channels: int
    shape: tuple[int, int]

    def __init__(self, channels: int, shape: Iterable[int]) -> None:
        self.channels = int(channels)
        self.shape = (int(shape[0]), int(shape[1]))

    def as_dict(self) -> DictConfig:
        """Return the DataSpace as a DictConfig."""
        return DictConfig({"channels": self.channels, "shape": self.shape})
