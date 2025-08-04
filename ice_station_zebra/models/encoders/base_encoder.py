from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class BaseEncoder(nn.Module, ABC):
    """
    Encoder that takes data in an input space and translates it to a smaller latent space

    Input:
        Tensor of (batch_size, input_channels, input_height, input_width)

    Output:
        Tensor of (batch_size, latent_channels, latent_height, latent_width)
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Transformation summary

        Input:
            x: Tensor of (batch_size, input_channels, input_height, input_width)

        Output:
            Tensor of (batch_size, latent_channels, latent_height, latent_width)
        """
