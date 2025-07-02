from typing import Any

# from lightning.pytorch.strategies import SingleDeviceStrategy
from pytorch_lightning.strategies import SingleDeviceStrategy


class ZebraSingleDeviceStrategy(SingleDeviceStrategy):
    def __init__(self, static_graph: bool, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
