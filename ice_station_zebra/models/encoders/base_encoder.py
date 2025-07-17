from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor, nn


class BaseEncoder(nn.Module, ABC):
    """
    Base encoder

    Transform data from the full phase space into a reduced latent space
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...
