from dataclasses import dataclass
from typing import TypedDict

from pydantic import validate_arguments


class DataloaderArgs(TypedDict):
    batch_size: int
    sampler: None
    batch_sampler: None
    drop_last: bool
    worker_init_fn: None


@validate_arguments
@dataclass
class ZebraDataSpace:
    channels: int
    shape: tuple[int, int]

    def __post_init__(self):
        if not isinstance(self.shape, tuple):
            self.shape = tuple(self.shape)
