from typing import TypedDict

class DataloaderArgs(TypedDict):
    batch_size: int
    sampler: None
    batch_sampler: None
    drop_last: bool
    worker_init_fn: None