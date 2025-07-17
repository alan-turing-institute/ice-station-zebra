from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class BaseProcessor(nn.Module, ABC):
    """
    Base processor

    Transform data within the reduced latent space
    """

    def __init__(self, n_latent_channels: int) -> None:
        super().__init__()
        self.n_latent_channels = n_latent_channels

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...
