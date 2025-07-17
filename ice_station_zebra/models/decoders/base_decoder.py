from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor, nn


class BaseDecoder(nn.Module, ABC):
    """
    Base decoder

    Transform data from the reduced latent space back to the full phase space
    """

    def __init__(self, output_shape: tuple[int, int], output_channels: int) -> None:
        super().__init__()
        self.output_shape = output_shape
        self.output_channels = output_channels

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...
