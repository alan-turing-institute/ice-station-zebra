from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class BaseEncoder(nn.Module, ABC):
    """
    Encoder that takes data in an input space and translates it to a smaller latent space

    Input space:
        Tensor[NTCHW] with (batch_size, n_history_steps, input_channels, input_height, input_width)

    Latent space:
        Tensor[NCHW] with (batch_size, latent_channels, latent_height, latent_width)
    """

    def __init__(self, *, name: str, n_history_steps: int) -> None:
        super().__init__()
        self.name = name
        self.n_history_steps = n_history_steps

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Transformation summary

        Args:
            x: Tensor[NTCHW] with (batch_size, n_history_steps, input_channels, input_height, input_width)

        Returns:
            Tensor[NCHW] with (batch_size, latent_channels, latent_height, latent_width)
        """
